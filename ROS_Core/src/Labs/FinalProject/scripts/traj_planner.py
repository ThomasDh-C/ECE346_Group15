#!/usr/bin/env python

import threading
import rospy
import numpy as np
import os
import time
import queue
import yaml
import sys
sys.path.append(os.getcwd())

from task2_world import RealtimeBuffer, Policy, GeneratePwm
# from ILQR import RefPath
from ILQR import ILQR_np as ILQR

from racecar_msgs.msg import ServoMsg
from racecar_planner.cfg import plannerConfig

from dynamic_reconfigure.server import Server
from tf.transformations import euler_from_quaternion
from nav_msgs.msg import Odometry

# used to display the trajectory on RVIZ
from nav_msgs.msg import Path as PathMsg
from std_srvs.srv import Empty, EmptyResponse

# You will use those for lab2
from racecar_msgs.msg import OdometryArray
from utils import frs_to_obstacle, frs_to_msg, get_obstacle_vertices, get_ros_param
from visualization_msgs.msg import MarkerArray
from racecar_obs_detection.srv import GetFRS, GetFRSResponse

# To get nav points
from racecar_routing.srv import Plan, PlanResponse, PlanRequest
from task2_world.util import RefPath


class TrajectoryPlanner():
    '''
    Main class for the Receding Horizon trajectory planner
    '''

    def __init__(self):
        # Indicate if the planner is used to generate a new trajectory
        self.update_lock = threading.Lock()
        self.latency = 0.0
        self.static_obstacle_dict = {}

        self.read_parameters()

        # Initialize the PWM converter
        self.pwm_converter = GeneratePwm()

        rospy.loginfo("about to start ILQR warmup...")

        # set up the optimal control solver
        self.setup_planner()

        self.setup_publisher()

        self.setup_subscriber()

        self.setup_service()
        
        # Grab waypoints from yaml file
        config_file_name = __file__.replace("scripts/traj_planner.py", "task1.yaml")
        self.load_config(config_file_name)

        rospy.loginfo("About to wait for /obstacles/get_frs service...")

        # start nav thread, control and planning thread
        threading.Thread(target=self.nav_thread).start()

        threading.Thread(target=self.control_thread).start()
        threading.Thread(target=self.receding_horizon_planning_thread).start()


        print("\t All threads started from traj_planner.py...")



    def load_config(self, config_path):
        '''
        This function loads parameters from a yaml file.
        '''
        with open(config_path, 'r') as f:
            config_dict = yaml.load(f, Loader=yaml.FullLoader)
            
        for key, val in config_dict.items():
                setattr(self, key, val)



    def read_parameters(self):
        '''
        This function reads the parameters from the parameter server
        '''
        # Required parameters
        self.package_path = rospy.get_param('~package_path')

        self.receding_horizon = get_ros_param('~receding_horizon', False)

        # Read ROS topic names to subscribe
        self.odom_topic = get_ros_param('~odom_topic', '/slam_pose')
        self.path_topic = get_ros_param('~path_topic', '/Routing/Path')
        self.obstacle_topic = get_ros_param(
            '~obstacle_topic', '/prediction/obstacles')

        # Read ROS topic names to publish
        self.control_topic = get_ros_param(
            '~control_topic', '/control/servo_control')
        self.traj_topic = get_ros_param('~traj_topic', '/Planning/Trajectory')

        # Read the simulation flag,
        # if the flag is true, we are in simulation
        # and no need to convert the throttle and steering angle to PWM
        self.simulation = get_ros_param('~simulation', True)

        # Read Planning parameters
        # if true, the planner will load a path from a file rather than subscribing to a path topic
        self.replan_dt = get_ros_param('~replan_dt', 0.1)

        # Read the ILQR parameters file, if empty, use the default parameters
        ilqr_params_file = get_ros_param('~ilqr_params_file', '')
        if ilqr_params_file == '':
            self.ilqr_params_abs_path = None
        elif os.path.isabs(ilqr_params_file):
            self.ilqr_params_abs_path = ilqr_params_file
        else:
            self.ilqr_params_abs_path = os.path.join(
                self.package_path, ilqr_params_file)

        # OUR STUFF
        self.static_obs_topic = get_ros_param(
            '~static_obs_topic', '/Obstacles/Static')

    def setup_planner(self):
        '''
        This function setup the ILQR solver
        '''
        # Initialize ILQR solver
        self.planner = ILQR(self.ilqr_params_abs_path)

        # create buffers to handle multi-threading
        self.plan_state_buffer = RealtimeBuffer()
        self.control_state_buffer = RealtimeBuffer()
        self.control_state_buffer2 = RealtimeBuffer()
        self.policy_buffer = RealtimeBuffer()
        self.path_buffer = RealtimeBuffer()
        # Indicate if the planner is ready to generate a new trajectory
        self.planner_ready = True

    def setup_publisher(self):
        '''
        This function sets up the publisher for the trajectory
        '''
        # Publisher for the planned nominal trajectory for visualization
        self.trajectory_pub = rospy.Publisher(
            self.traj_topic, PathMsg, queue_size=1)
        
        self.path_pub = rospy.Publisher(
            self.path_topic, PathMsg, queue_size=1)

        # Publisher for the control command
        self.control_pub = rospy.Publisher(
            self.control_topic, ServoMsg, queue_size=1)

    def setup_subscriber(self):
        '''
        This function sets up the subscriber for the odometry and path
        '''
        self.pose_sub = rospy.Subscriber(
            self.odom_topic, Odometry, self.odometry_callback, queue_size=10)

        # OUR STUFF 2/34/23
        self.static_obs_sub = rospy.Subscriber(
            self.static_obs_topic, MarkerArray, self.static_obs_callback,
            queue_size=10)

    def setup_service(self):
        '''
        Set up ros service
        '''
        self.start_srv = rospy.Service(
            '/Planning/Start', Empty, self.start_planning_cb)
        self.stop_srv = rospy.Service(
            '/Planning/Stop', Empty, self.stop_planning_cb)

        self.dyn_server = Server(plannerConfig, self.reconfigure_callback)

    def start_planning_cb(self, req):
        '''
        ros service callback function for start planning
        '''
        rospy.loginfo('Start planning!')
        self.planner_ready = True
        return EmptyResponse()

    def stop_planning_cb(self, req):
        '''
        ros service callback function for stop planning
        '''
        rospy.loginfo('Stop planning!')
        self.planner_ready = False
        self.policy_buffer.reset()
        return EmptyResponse()

    def reconfigure_callback(self, config, level):
        self.update_lock.acquire()
        self.latency = config['latency']
        rospy.loginfo(f"Latency Updated to {self.latency} s")
        if self.latency < 0.0:
            rospy.logwarn(
                f"Negative latency compensation {self.latency} is not a good idea!")
        self.update_lock.release()
        return config

    def odometry_callback(self, odom_msg):
        '''
        Subscriber callback function of the robot pose
        '''
        # Add the current state to the buffer
        # Controller thread will read from the buffer
        # Then it will be processed and add to the planner buffer
        # inside the controller thread
        self.control_state_buffer.writeFromNonRT(odom_msg)
        self.control_state_buffer2.writeFromNonRT(odom_msg)

    def path_callback(self, path_msg):
        x = []
        y = []
        width_L = []
        width_R = []
        speed_limit = []

        for waypoint in path_msg.poses:
            x.append(waypoint.pose.position.x)
            y.append(waypoint.pose.position.y)
            width_L.append(waypoint.pose.orientation.x)
            width_R.append(waypoint.pose.orientation.y)
            speed_limit.append(waypoint.pose.orientation.z)

        centerline = np.array([x, y])

        try:
            ref_path = RefPath(centerline, width_L, width_R,
                               speed_limit, loop=False)
            self.path_buffer.writeFromNonRT(ref_path)
            rospy.loginfo('Path received!')
        except:
            rospy.logwarn('Invalid path received! Move your robot and retry!')

    def static_obs_callback(self, markers_msg):
        '''3/24/23: our stuff, callback function for static obstacle topic'''
        self.static_obstacle_dict = {}  # reset dict every call
        for marker in markers_msg.markers:
            idx, verts = get_obstacle_vertices(marker)
            self.static_obstacle_dict[idx] = verts

    @staticmethod
    def compute_control(x, x_ref, u_ref, K_closed_loop):
        '''
        Given the current state, reference trajectory, control command
        and closed loop gain, compute the control command

        Args:
            x: np.ndarray, [dim_x] current state
            x_ref: np.ndarray, [dim_x] reference trajectory
            u_ref: np.ndarray, [dim_u] reference control command
            K_closed_loop: np.ndarray, [dim_u, dim_x] closed loop gain

        Returns:
            accel: float, acceleration command [m/s^2]
            steer_rate: float, steering rate command [rad/s]
        '''

        ###############################
        #### TODO: Task 2 #############
        ###############################
        # Implement your control law here using ILQR policy
        # Hint: make sure that the difference in heading is between [-pi, pi]

        dx = x - x_ref
        dx[3] = np.arctan2(np.sin(dx[3]), np.cos(dx[3]))  # heading thing
        u_t = u_ref + K_closed_loop @ dx

        accel = u_t[0]
        steer_rate = u_t[1]

        ##### END OF TODO ##############

        return accel, steer_rate

    def control_thread(self):
        '''
        Main control thread to publish control command
        '''
        print("Entered control_thread...")
        rate = rospy.Rate(40)
        u_queue = queue.Queue()

        # values to keep track of the previous control command
        prev_state = None  # [x, y, v, psi, delta]
        prev_u = np.zeros(3)  # [accel, steer, t]

        # helper function to compute the next state
        def dyn_step(x, u, dt):
            dx = np.array([x[2]*np.cos(x[3]),
                           x[2]*np.sin(x[3]),
                           u[0],
                           x[2]*np.tan(u[1]*1.1)/0.257,
                           0
                           ])
            x_new = x + dx*dt
            x_new[2] = max(0, x_new[2])  # do not allow negative velocity
            x_new[3] = np.mod(x_new[3] + np.pi, 2 * np.pi) - np.pi
            x_new[-1] = u[1]
            return x_new

        while not rospy.is_shutdown():
            # initialize the control command
            accel = -5
            steer = 0
            state_cur = None
            policy = self.policy_buffer.readFromRT()

            # take the latency of publish into the account
            if self.simulation:
                t_act = rospy.get_rostime().to_sec()
            else:
                self.update_lock.acquire()
                t_act = rospy.get_rostime().to_sec() + self.latency
                self.update_lock.release()

            # check if there is new state available
            if self.control_state_buffer.new_data_available:
                odom_msg = self.control_state_buffer.readFromRT()
                t_slam = odom_msg.header.stamp.to_sec()

                u = np.zeros(3)
                u[-1] = t_slam
                while not u_queue.empty() and u_queue.queue[0][-1] < t_slam:
                    u = u_queue.get()  # remove old control commands

                # get the state from the odometry message
                q = [odom_msg.pose.pose.orientation.x, odom_msg.pose.pose.orientation.y,
                     odom_msg.pose.pose.orientation.z, odom_msg.pose.pose.orientation.w]
                # get the heading angle from the quaternion
                psi = euler_from_quaternion(q)[-1]

                state_cur = np.array([
                    odom_msg.pose.pose.position.x,
                    odom_msg.pose.pose.position.y,
                    odom_msg.twist.twist.linear.x,
                    psi,
                    u[1]
                ])

                # predict the current state use past control command
                for i in range(u_queue.qsize()):
                    u_next = u_queue.queue[i]
                    dt = u_next[-1] - u[-1]
                    state_cur = dyn_step(state_cur, u, dt)
                    u = u_next

                # predict the cur state with the most recent control command
                state_cur = dyn_step(state_cur, u, t_act - u[-1])

                # update the state buffer for the planning thread
                plan_state = np.append(state_cur, t_act)
                self.plan_state_buffer.writeFromNonRT(plan_state)

            # if there is no new state available, we do one step forward integration to predict the state
            elif prev_state is not None:
                t_prev = prev_u[-1]
                dt = t_act - t_prev
                # predict the state using the last control command is executed
                state_cur = dyn_step(prev_state, prev_u, dt)

            # Generate control command from the policy
            if policy is not None:
                # get policy
                if not self.receding_horizon:
                    state_ref, u_ref, K = policy.get_policy_by_state(state_cur)
                else:
                    state_ref, u_ref, K = policy.get_policy(t_act)

                if state_ref is not None:
                    accel, steer_rate = self.compute_control(
                        state_cur, state_ref, u_ref, K)
                    steer = max(-0.37, min(0.37, prev_u[1] + steer_rate*dt))
                else:
                    # reset the policy buffer if the policy is not valid
                    rospy.logwarn(
                        "Try to retrieve a policy beyond the horizon! Reset the policy buffer!")
                    self.policy_buffer.reset()

            # generate control command
            if not self.simulation and state_cur is not None:
                # If we are using robot,
                # the throttle and steering angle needs to convert to PWM signal
                throttle_pwm, steer_pwm = self.pwm_converter.convert(
                    accel, steer, state_cur[2])
            else:
                throttle_pwm = accel
                steer_pwm = steer

            # publish control command
            servo_msg = ServoMsg()
            # use the current time to avoid synchronization issue
            servo_msg.header.stamp = rospy.get_rostime()
            servo_msg.throttle = throttle_pwm
            servo_msg.steer = steer_pwm
            self.control_pub.publish(servo_msg)

            # Record the control command and state for next iteration
            u_record = np.array([accel, steer, t_act])
            u_queue.put(u_record)
            prev_u = u_record
            prev_state = state_cur

            # end of while loop
            rate.sleep()

    def nav_thread(self):
        '''
        basically just the code from the readme lol
        '''

        # How close it gets to nav goal before it is good
        EPS = 0.2

        print("Entered nav thread...")
        rospy.wait_for_service('/routing/plan')
        plan_client = rospy.ServiceProxy('/routing/plan', Plan)

        # print("Done waiting for service: /routing/plan!")

        # initialize goals to far, far away
        x_goal = 5 # x coordinate of the goal ## temp values
        y_goal = 7 # y coordinate of the goal
        first_time = True
        current_waypoint = 0

        path_msg = None
        
        while not rospy.is_shutdown():
            if self.control_state_buffer2.new_data_available:
                # print("new control state buffer data available")
                odom_msg = self.control_state_buffer2.readFromRT()

                x_start = odom_msg.pose.pose.position.x # x coordinate of the start
                y_start = odom_msg.pose.pose.position.y # y coordinate of the start

                # print(f'nav_thread: x_start: {x_start}, y_start: {y_start}')


                if first_time or (abs(x_start-x_goal) < EPS and abs(y_start - y_goal) < EPS):
                    first_time = False
                    current_waypoint += 1
                    print(f'Getting the {current_waypoint}th waypoint: {self.goals[self.goal_order[current_waypoint]-1]}')
                    # Get new goal locations
                    # TODO: handle last waypoint!
                    x_goal = self.goals[self.goal_order[current_waypoint]-1][0]
                    y_goal = self.goals[self.goal_order[current_waypoint]-1][1]

                    plan_request = PlanRequest([x_start, y_start], [x_goal, y_goal])
                    plan_response = plan_client(plan_request)
                    path_msg = plan_response.path

                    path_msg.header.frame_id='map'
                    self.path_pub.publish(path_msg)

                    self.path_callback(path_msg)
                    
                    print(f'New reference path written from ({x_start}, {y_start}) to ({x_goal}, {y_goal})')

    def receding_horizon_planning_thread(self):
        '''
        This function is the main thread for receding horizon planning
        We repeatedly call ILQR to replan the trajectory (policy) once the new state is available
        '''
        
        rospy.loginfo('Receding Horizon Planning thread started waiting for ROS service calls...')
        t_last_replan = 0
        while not rospy.is_shutdown():
            # determine if we need to replan
            if self.plan_state_buffer.new_data_available:
                state_cur = self.plan_state_buffer.readFromRT()
                
                t_cur = state_cur[-1] # the last element is the time
                dt = t_cur - t_last_replan
                
                # Do replanning
                if dt >= self.replan_dt:
                    # Get the initial controls for hot start
                    init_controls = None

                    original_policy = self.policy_buffer.readFromRT()
                    if original_policy is not None:
                        init_controls = original_policy.get_ref_controls(t_cur)

                    # Update the path
                    if self.path_buffer.new_data_available:
                        new_path = self.path_buffer.readFromRT()
                        self.planner.update_ref_path(new_path)
                    
                    # Update the static obstacles
                    obstacles_list = []
                    for vertices in self.static_obstacle_dict.values():
                        obstacles_list.append(vertices)
                    
                    # update dynamic obstacles
                    # try:
                    #     t_list= t_cur + np.arange(self.planner.T)*self.planner.dt
                    #     frs_respond = self.get_frs(t_list)
                    #     obstacles_list.extend(frs_to_obstacle(frs_respond))
                    #     self.frs_pub.publish(frs_to_msg(frs_respond))
                    # except:
                    #     rospy.logwarn_once('FRS server not available!')
                    #     frs_respond = None
                        
                    # self.planner.update_obstacles(obstacles_list)
                    
                    # Replan use ilqr
                    t
                    new_plan = self.planner.plan(state_cur[:-1], init_controls)
                    
                    plan_status = new_plan['status']
                    if plan_status == -1:
                        rospy.logwarn_once('No path specified!')
                        continue
                    
                    if self.planner_ready:
                        rospy.loginfo(f"replan after {dt}")
                        # If stop planning is called, we will not write to the buffer
                        new_policy = Policy(X = new_plan['trajectory'], 
                                            U = new_plan['controls'],
                                            K = new_plan['K_closed_loop'], 
                                            t0 = t_cur, 
                                            dt = self.planner.dt,
                                            T = self.planner.T)
                        
                        self.policy_buffer.writeFromNonRT(new_policy)
                        
                        # publish the new policy for RVIZ visualization
                        self.trajectory_pub.publish(new_policy.to_msg())        
                        t_last_replan = t_cur

    