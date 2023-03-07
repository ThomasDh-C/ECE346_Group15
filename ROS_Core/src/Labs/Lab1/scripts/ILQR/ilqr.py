from typing import Tuple, Optional, Dict, Union
from jaxlib.xla_extension import DeviceArray
import time
import os
import numpy as np
import jax
from .dynamics import Bicycle5D
from .cost import Cost, CollisionChecker, Obstacle
from .ref_path import RefPath
from .config import Config
import time
from jax.lax import fori_loop
from functools import partial
import jax.numpy as jnp

status_lookup = ['Iteration Limit Exceed',
                 'Converged',
                 'Failed Line Search']


class ILQR():
    def __init__(self, config_file=None) -> None:

        self.config = Config()  # Load default config.
        if config_file is not None:
            self.config.load_config(config_file)  # Load config from file.

        self.load_parameters()
        print('ILQR setting:', self.config)

        # Set up Jax parameters
        jax.config.update('jax_platform_name', self.config.platform)
        print('Jax using Platform: ', jax.lib.xla_bridge.get_backend().platform)

        # If you want to use GPU, lower the memory fraction from 90% to avoid OOM.
        os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '20'

        self.dyn = Bicycle5D(self.config)
        self.cost = Cost(self.config)
        self.ref_path = None

        # collision checker
        # Note: This will not be used until lab2.
        self.collision_checker = CollisionChecker(self.config)
        self.obstacle_list = []

        # Do a dummy run to warm up the jitted functions.
        self.warm_up()

    def load_parameters(self):
        '''
        This function defines ILQR parameters from <self.config>.
        '''
        # ILQR parameters
        self.dim_x = self.config.num_dim_x
        self.dim_u = self.config.num_dim_u
        self.T = int(self.config.T)
        self.dt = float(self.config.dt)
        self.max_iter = int(self.config.max_iter)
        self.tol = float(self.config.tol)  # ILQR update tolerance.

        # line search parameters.
        self.alphas = self.config.line_search_base**(
            np.arange(self.config.line_search_a,
                      self.config.line_search_b,
                      self.config.line_search_c)
        )

        print('Line Search Alphas: ', self.alphas)

        # regularization parameters
        self.reg_min = float(self.config.reg_min)
        self.reg_max = float(self.config.reg_max)
        self.reg_init = float(self.config.reg_init)
        self.reg_scale_up = float(self.config.reg_scale_up)
        self.reg_scale_down = float(self.config.reg_scale_down)
        self.max_attempt = self.config.max_attempt

    def warm_up(self):
        '''
        Warm up the jitted functions.
        '''
        # Build a fake path as a 1 meter radius circle.
        theta = np.linspace(0, 2 * np.pi, 100)
        centerline = np.zeros([2, 100])
        centerline[0, :] = 1 * np.cos(theta)
        centerline[1, :] = 1 * np.sin(theta)

        self.ref_path = RefPath(centerline, 0.5, 0.5, 1, True)

        # add obstacle
        obs = np.array([[0, 0, 0.5, 0.5], [1, 1.5, 1, 1.5]]).T
        obs_list = [[obs for _ in range(self.T)]]
        self.update_obstacles(obs_list)

        x_init = np.array([0.0, -1.0, 1, 0, 0])
        print('Start warm up ILQR...')
        self.plan(x_init)
        print('ILQR warm up finished.')

        self.ref_path = None
        self.obstacle_list = []

    def update_ref_path(self, ref_path: RefPath):
        '''
        Update the reference path.
        Args:
                ref_path: RefPath: reference path.
        '''
        self.ref_path = ref_path

    def update_obstacles(self, vertices_list: list):
        '''
        Update the obstacle list for a list of vertices.
        Args:
                vertices_list: list of np.ndarray: list of vertices for each obstacle.
        '''
        # Note: This will not be used until lab2.
        self.obstacle_list = []
        for vertices in vertices_list:
            self.obstacle_list.append(Obstacle(vertices))

    def get_references(self, trajectory: Union[np.ndarray, DeviceArray]):
        '''
        Given the trajectory, get the path reference and obstacle information.
        Args:
                trajectory: [num_dim_x, T] trajectory.
        Returns:
                path_refs: [num_dim_x, T] np.ndarray: references.
                obs_refs: [num_dim_x, T] np.ndarray: obstacle references.
        '''
        trajectory = np.asarray(trajectory)
        path_refs = self.ref_path.get_reference(trajectory[:2, :])
        obs_refs = self.collision_checker.check_collisions(
            trajectory, self.obstacle_list)
        return path_refs, obs_refs

    def compute_new_cost(self, trajectory, controls):
        # Get path and obstacle references based on your current nominal trajectory.
        # Note: you will NEED TO call this function and get new references at each iteration.
        path_refs, obs_refs = self.get_references(trajectory)

        # Get the initial cost of the trajectory.
        return self.cost.get_traj_cost(trajectory, controls, path_refs, obs_refs)

    def backward_pass(self, x_nom, u_nom, path_refs, obs_refs, lambda_val=1):
        ''' Run backward LQR pass
                x_nom: 5xN matrix of trajectory
                u_nom: 2xN matrix of nominal controls
                target_state: 5d matrix of target state

                Returns:
                K: State feedback matrices
                k: feedforward offsets
                lambda_val: regularisation parameter
        '''
        # Note: Largely based on handout and python example code
        # ported to jax + variables according to handout and slide names

        # Get path and obstacle references based on your current nominal trajectory.
        q, r, Q, R, H = self.cost.get_derivatives_np(
            x_nom, u_nom, path_refs, obs_refs)
        A, B = self.dyn.get_jacobian_np(x_nom, u_nom)
        k_open_loop = np.zeros((2, self.T))
        K_closed_loop = np.zeros((2, 5, self.T))

        # Derivative of value function at final step
        end_t_idx = self.T - 1
        p = q[:, end_t_idx]
        P = Q[:, :, end_t_idx]
        t = end_t_idx - 1
        lambda_a = 5  # Arbitrary - just has to be over 1
        while t >= 0:
            A_curr = A[:, :, t]
            B_curr = B[:, :, t]
            Q_x = q[:, t] + np.matmul(A_curr.T, p)
            Q_u = r[:, t] + np.matmul(B_curr.T, p)
            Q_xx = Q[:, :, t] + \
                np.matmul(A_curr.T, np.matmul(P, A_curr))
            Q_uu = R[:, :, t] + \
                np.matmul(B_curr.T, np.matmul(P, B_curr))
            Q_ux = H[:, :, t] + \
                np.matmul(B_curr.T, np.matmul(P, A_curr))
            # Add regularization
            reg_matrix = lambda_val * np.eye(5)
            Q_uu_reg = R[:, :, t] + \
                np.matmul(B_curr.T, np.matmul(
                    (P+reg_matrix), B_curr))
            Q_ux_reg = H[:, :, t] + \
                np.matmul(B_curr.T, np.matmul(
                    (P+reg_matrix), A_curr))
            # Check if Q_uu_reg is positive definite
            if not np.all(np.linalg.eigvals(Q_uu_reg) > 0) and lambda_val < 1e5:
                lambda_val *= lambda_a
                t = end_t_idx - 1  # restart from end of trajectory
                p = q[:, end_t_idx]
                P = Q[:, :, end_t_idx]
                continue

            Q_uu_reg_inv = np.linalg.inv(Q_uu_reg)

            # Calculate policy

            k = np.matmul(-Q_uu_reg_inv, Q_u)
            k_open_loop[:, t] = k

            K = np.matmul(-Q_uu_reg_inv, Q_ux_reg)
            K_closed_loop[:, :, t] = K

            # Update value function derivative for the previous time step
            p = Q_x + np.matmul(K.T, np.matmul(Q_uu, k)) + \
                np.matmul(K.T, Q_u) + np.matmul(Q_ux.T, k)
            P = Q_xx + \
                np.matmul(K.T, np.matmul(Q_uu, K)) + \
                np.matmul(K.T, Q_ux) + \
                np.matmul(Q_ux.T, K)
            t -= 1

        lambda_val = max(1e-5, lambda_val*0.5)
        return K_closed_loop, k_open_loop, lambda_val

    def line_search(self, x, x_nom, u_nom, controls, alpha_curr, k, K):

        for t in range(self.T - 1):
            dx = x[:, t] - x_nom[:, t]
            dx[3] = np.arctan2(np.sin(dx[3]), np.cos(dx[3]))
            controls_temp = u_nom[:, t] + np.matmul(
                K[:, :, t], dx) + (alpha_curr * k[:, t])
            x[:, t+1], u_clip = self.dyn.integrate_forward_np(
                x[:, t], controls_temp)

            # ran controls with clipped controls anyways
            controls[:, t] = u_clip
        return np.copy(x), np.copy(controls)

        # ########## x.at[idx].set(y)
    @partial(jax.jit, static_argnums=(0,))
    def line_search_jax(self, x: np.ndarray, x_nom: np.ndarray, u_nom: np.ndarray, controls: np.ndarray, alpha_curr: float, k: np.ndarray, K: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        @jax.jit
        def body_line_search_jax(t, args):
            x, x_nom, u_nom, controls, alpha_curr, k, K = args
            dx = x[:, t] - x_nom[:, t]
            dx = dx.at[3].set(jnp.arctan2(jnp.sin(dx[3]), jnp.cos(dx[3])))
            controls_temp = u_nom[:, t] + jnp.matmul(
                K[:, :, t], dx) + (alpha_curr * k[:, t])
            x_temp, u_clip = self.dyn.integrate_forward_jax(
                x[:, t], controls_temp)
            x = x.at[:, t+1].set(x_temp)
            controls = controls.at[:, t].set(u_clip)
            # print('hello from deeeeeeep inside')
            return x, x_nom, u_nom, controls, alpha_curr, k, K

        x[:, 0] = jnp.copy(x_nom[:, 0])
        _, _, _, _, _, _, _ = jax.lax.fori_loop(0, self.T-1, body_line_search_jax,
                                                (x, x_nom, u_nom, controls, alpha_curr, k, K))
        # print('in between world')
        # x = x.set(temp_x)
        # print('hello from the inside')
        return x, controls

    def plan(self, init_state: np.ndarray,
             u_nom: Optional[np.ndarray] = None) -> Dict:
        '''
        Main ILQR loop.
        Args:
                init_state: [num_dim_x] np.ndarray: initial state.
                u_nom: [num_dim_u, T] np.ndarray: optional initial control.
        Returns:
                A dictionary with the following keys:
                        status: int: -1 for failure, 0 for success. You can add more status if you want.
                        t_process: float: time spent on planning.
                        x_nom: [num_dim_x, T] np.ndarray: ILQR planned x_nom.
                        controls: [num_dim_u, T] np.ndarray: ILQR planned controls sequence.
                        K_closed_loop: [num_dim_u, num_dim_x, T] np.ndarray: closed loop gain.
                        k_closed_loop: [num_dim_u, T] np.ndarray: closed loop bias.
        '''

        # We first check if the planner is ready
        if self.ref_path is None:
            print('No reference path is provided.')
            return dict(status=-1)

        # if no initial control sequence is provided, we assume it is all zeros.
        if u_nom is None:
            u_nom = np.zeros((self.dim_u, self.T))
        else:
            assert u_nom.shape[1] == self.T

        # Start timing
        t_start = time.time()

        # 1. Rolls out the nominal x_nom (implemented) and gets the initial cost.
        x_nom, u_nom = self.dyn.rollout_nominal_np(init_state, u_nom)
        controls = np.copy(u_nom)  # initialize for later :)
        improved_flag = True  # again init for later

        # 2. Get the initial cost of the x_nom.
        path_refs, obs_refs = self.get_references(x_nom)
        Jorig = self.cost.get_traj_cost(x_nom, u_nom, path_refs, obs_refs)

        ##########################################################################
        # TODO 1: Implement the ILQR algorithm. Feel free to add any helper functions.
        # You will find following implemented functions useful:
        # https://colab.research.google.com/drive/1Svs5mOh-2WPcUbjGc_DEs2tnez3gRwei?usp=sharinghttps%3A%2F%2Fcolab.research.google.com%2Fdrive%2F1Svs5mOh-2WPcUbjGc_DEs2tnez3gRwei%3Fusp%3Dsharing#scrollTo=1JAhQdsMt7gV
        # from canvas was useful :)
        # use jax so can use gpu and ta happy

        steps = 0
        converged_flag = False
        K, k = None, None
        lambda_val = 1

        x = np.zeros_like(x_nom)
        while steps <= self.max_iter:
            # Backward pass
            # JAX arrays are immutable. Instead of ``x[idx] = y``, use ``x = x.at[idx].set(y)`` or another .at[] method: https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.ndarray.at.html
            # t_backward = time.time()
            if not improved_flag:
                lambda_val *= 5
                if lambda_val > 1000:
                    break
            K, k, lambda_val = self.backward_pass(
                x_nom, u_nom, path_refs, obs_refs, lambda_val)
            # print(f't_backward pass: {1000*(time.time() - t_backward):.2f}ms')

            # Forward pass
            t_forward = time.time()
            improved_flag = False
            for alpha_curr in self.alphas:
                # print('hello')
                # x, controls = self.line_search(
                #     x, x_nom, u_nom, controls, alpha_curr, k, K)
                # # 1. Rolls out the nominal x_nom
                # # update controls
                x[:, 0] = x_nom[:, 0]
                for t in range(self.T - 1):
                    dx = x[:, t] - x_nom[:, t]
                    dx[3] = np.arctan2(np.sin(dx[3]), np.cos(dx[3]))
                    controls_temp = u_nom[:, t] + np.matmul(
                        K[:, :, t], dx) + (alpha_curr * k[:, t])
                    x[:, t+1], u_clip = self.dyn.integrate_forward_np(
                        x[:, t], controls_temp)

                    # ran controls with clipped controls anyways
                    controls[:, t] = u_clip

                # 2. Get the initial cost of the x_nom.
                # ?? - save references of objects you may hit/ needed for compute for backward pass
                # @partial(jax.jit, static_argnums=(0,)) --> for things you wanna jit
                # regular functions that aren't in your class just @jax.jit decorator
                # make sure your inptus are always same shape
                # while, for and if are special in jax --> have conditional function
                # no slicing outside of jitted functions
                # if line search fail, can try larger lambda value in backward pass (multiply and try again)
                # check self parameters for backward pass
                # get speed for plan one step to 0.1s
                path_refs, obs_refs = self.get_references(x)
                Jnew = self.cost.get_traj_cost(
                    x, controls, path_refs, obs_refs)
                if Jnew < Jorig:
                    improved_flag = True
                    if np.abs(Jnew - Jorig) < self.config.tol:
                        converged_flag = True
                    x_nom = np.copy(x)
                    u_nom = np.copy(controls)
                    Jorig = Jnew
                    break

            steps += 1
            if converged_flag:
                break
            # break if feedforward terms are sufficiently small
            # print(f't_forward pass: {1000*(time.time() - t_forward):.2f}ms')
            # print(f'Converged in {steps} iterations')
        ########################### #END of TODO 1 #####################################
        t_process = time.time() - t_start
        # print(f't_process: {t_process*1000:.2f}ms')
        solver_info = dict(
            t_process=t_process,  # Time spent on planning
            trajectory=x_nom,
            controls=controls,
            status=(-1, 0)[converged_flag],
            K_closed_loop=K,
            k_open_loop=k
        )
        return solver_info
