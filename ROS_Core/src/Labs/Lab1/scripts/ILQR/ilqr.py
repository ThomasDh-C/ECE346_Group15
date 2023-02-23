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

import jax.numpy as jnp

status_lookup = ['Iteration Limit Exceed',
                'Converged',
                'Failed Line Search']

class ILQR():
	def __init__(self, config_file = None) -> None:

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
		centerline[0,:] = 1 * np.cos(theta)
		centerline[1,:] = 1 * np.sin(theta)

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
		obs_refs = self.collision_checker.check_collisions(trajectory, self.obstacle_list)
		return path_refs, obs_refs

	def compute_new_cost(self,trajectory, controls):
		# Get path and obstacle references based on your current nominal trajectory.
		# Note: you will NEED TO call this function and get new references at each iteration.
		path_refs, obs_refs = self.get_references(trajectory)

		# Get the initial cost of the trajectory.
		return self.cost.get_traj_cost(trajectory, controls, path_refs, obs_refs)

	def backward_pass(self, x_nom, u_nom, lambda_val=1):
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
		path_refs, obs_refs = self.get_references(x_nom)
		q, r, Q, R, H = self.cost.get_derivatives_jax(x_nom, u_nom, path_refs, obs_refs)
		
		A, B = self.dyn.get_jacobian_np(x_nom, u_nom)

		k_open_loop = jnp.zeros((2, self.T))
		K_closed_loop = jnp.zeros((2, 4, self.T))
		
		# Derivative of value function at final step
		end_t_idx = self.T - 1
		p = q[:, end_t_idx]
		P = Q[:,:, end_t_idx]
		t = end_t_idx - 1

		lambda_a = 5 # Arbitrary - just has to be over 1
		while t >= 0:
			Q_x = q[:,t] + jnp.matmul(A[:,:,t].T, p)
			Q_u = r[:,t] + jnp.matmul(B[:,:,t].T, p)
			Q_xx = Q[:,:,t] + jnp.matmul(A[:,:,t].T, jnp.matmul(P, A[:,:,t]))
			Q_uu = R[:,:,t] + jnp.matmul(B[:,:,t].T, jnp.matmul(P, B[:,:,t]))
			Q_ux = H[:,:,t] + jnp.matmul(B[:,:,t].T, jnp.matmul(P, A[:,:,t]))

			# Add regularization
			reg_matrix = reg * jnp.eye(5)
			Q_uu_reg = R[:,:,t] + jnp.matmul(B[:,:,t].T, jnp.matmul((P+reg_matrix), B[:,:,t]))
			Q_ux_reg = H[:,:,t] + jnp.matmul(B[:,:,t].T, jnp.matmul( (P+reg_matrix) , A[:,:,t]))
        
			# Check if Q_uu_reg is positive definite
			if not jnp.all(jnp.linalg.eigvals(Q_uu_reg) > 0) and lambda_val < 1e5:
				lambda_val *= lambda_a
				t = end_t_idx - 1 # restart from end of trajectory
				p = q[:,end_t_idx]
				P = Q[:,:, end_t_idx]
				continue

			Q_uu_reg_inv = jnp.linalg.inv(Q_uu_reg)

			# Calculate policy
			k = jnp.matmul(-Q_uu_reg_inv, Q_u)
			K = jnp.matmul(-Q_uu_reg_inv, Q_ux_reg)
			k_open_loop[:,t] = k          
			K_closed_loop[:, :, t] = K

			# Update value function derivative for the previous time step
			p = Q_x + jnp.matmul(K.T, jnp.matmul(Q_uu, k)) + jnp.matmul(K.T, Q_u) + jnp.matmul(Q_ux.T, k)
			P = Q_xx + jnp.matmul(K.T @ Q_uu, jnp.matmul(K + K.T, Q_ux)) + jnp.matmul(Q_ux.T, K)
			t -= 1

		lambda_val = max(1e-5, K*0.5)
		return K_closed_loop, k_open_loop, lambda_val

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
		if controls is None:
			controls =np.zeros((self.dim_u, self.T))
		else:
			assert controls.shape[1] == self.T

		# Start timing
		t_start = time.time()

		# 1. Rolls out the nominal x_nom (implemented) and gets the initial cost.
		x_nom, u_nom = self.dyn.rollout_nominal_np(init_state, u_nom)
		controls = u_nom	# initialize for later :)

		# 2. Get the initial cost of the x_nom.
		Jorig = self.compute_new_cost(x_nom, u_nom)

		##########################################################################
		# TODO 1: Implement the ILQR algorithm. Feel free to add any helper functions.
		# You will find following implemented functions useful:
		# https://colab.research.google.com/drive/1Svs5mOh-2WPcUbjGc_DEs2tnez3gRwei?usp=sharinghttps%3A%2F%2Fcolab.research.google.com%2Fdrive%2F1Svs5mOh-2WPcUbjGc_DEs2tnez3gRwei%3Fusp%3Dsharing#scrollTo=1JAhQdsMt7gV
		# from canvas was useful :)
		# use jax so can use gpu and ta happy

		steps = 0
		converged_flag = False
		K, k = None, None
		while steps <= 100:
			# Backward pass
			K, k, lambda_val = self.backward_pass(x_nom, u_nom, lambda_val=1)
			
			# Forward pass
			for alpha_curr in self.alphas:
				# 1. Rolls out the nominal x_nom (implemented) and gets the initial cost.
				# update controls
				x = jnp.zeros_like(x_nom)
				for t in range(self.T - 1):
					controls[:,t] = u_nom[:,t] + jnp.matmul(K[:,:,t], (x[:,t] - x_nom[:, t])) + (alpha_curr * k[:,:,t])
					x[:, t + 1] = self.dyn.integrate_forward_jax(x[:, t], controls[:, t])

				# 2. Get the initial cost of the x_nom.
				Jnew = self.compute_new_cost(x, controls)

				if jnp.abs(Jnew - Jorig) < 0.001: # arbitrary value
					x_nom = x
					k = k * alpha_curr
					converged_flag = True
					break
				
			steps += 1
			if converged_flag: break
			# break if feedforward terms are sufficiently small

		########################### #END of TODO 1 #####################################

		t_process = time.time() - t_start
		solver_info = dict(
				t_process=t_process, # Time spent on planning
				trajectory = x_nom,
				controls = controls,
				status= (-1,0)[converged_flag],
				K_closed_loop=K,
				k_open_loop=k
		)
		return solver_info



