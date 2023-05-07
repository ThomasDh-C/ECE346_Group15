#!/usr/bin/env python
import rospy
from std_msgs.msg import String
from nav_msgs.msg import Odometry
from task2_world import RealtimeBuffer, RefPath
from utils import frs_to_obstacle, frs_to_msg, get_obstacle_vertices, get_ros_param
from visualization_msgs.msg import MarkerArray, Marker
import yaml
from racecar_routing.srv import Plan, PlanRequest
from nav_msgs.msg import Path as PathMsg
import time
import numpy as np
from sklearn.neighbors import KDTree
from copy import deepcopy
from threading import Thread

from tf.transformations import euler_from_quaternion
from final_project.srv import Task, TaskRequest, TaskResponse, Reward, RewardRequest, RewardResponse, Schedule, ScheduleRequest, ScheduleResponse
from std_srvs.srv import Empty, EmptyRequest, EmptyResponse


control_state_buffer = RealtimeBuffer()

# A subscriber callback for odom
def odometry_callback(odom_msg):
    '''
    Subscriber callback function of the robot pose
    '''
    control_state_buffer.writeFromNonRT(odom_msg)



def msg_to_ref_path(poses):
    '''
    create a ref path from message 
    '''
    x = []
    y = []
    width_L = [] ## bounds on road
    width_R = []
    speed_limit = []
    
    for point in poses:
        x.append(point.pose.position.x)
        y.append(point.pose.position.y)
        width_L.append(point.pose.orientation.x)
        width_R.append(point.pose.orientation.y)
        speed_limit.append(point.pose.orientation.z)
    centerline = np.array([x, y])
    return RefPath(centerline, width_L, width_R, speed_limit, loop=False)

def boss_obstacle_target():
    boss_state_buffer = RealtimeBuffer()
    
    def boss_odometry_callback(odom_msg):
        '''Subscriber to boss pose'''
        boss_state_buffer.writeFromNonRT(odom_msg)
    # get warehouse positions from yaml file
    config_path = __file__.replace("scripts/path_planner_task2_node.py", "task2.yaml")
    warehouse_positions = []
    with open(config_path, 'r') as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)
        for warehouse_letter in ['A', 'B', 'C', 'D', 'E']:
            curr_warehouse = config_dict[f'warehouse_{warehouse_letter}']
            warehouse_positions.append(curr_warehouse['location'])

    # get boss pose
    boss_odom_topic = get_ros_param('~boss_odom_topic', '/Boss/Pose')
    boss_odom_subscriber = rospy.Subscriber(boss_odom_topic, Odometry, boss_odometry_callback, queue_size=10)
    static_obs_topic = get_ros_param('~static_obs_topic', '/Obstacles/Static')
    boss_obstacle_publisher = rospy.Publisher(static_obs_topic, MarkerArray, queue_size=1)

    start_time = rospy.get_time()
    boss_schedule_client = rospy.ServiceProxy('/SwiftHaul/BossSchedule', Schedule)
    boss_schedule = boss_schedule_client(ScheduleRequest()) # start_warehouse (list of start indexes)

    def response_idx_curr_boss_warehouse(start_time, boss_schedule):
        """Get idx in response object of next boss warehouse based on current time"""
        # local start_time, boss_schedule, warehouse_positions
        curr_time = rospy.get_time() - start_time

        # iterate through boss schedule to find current interval of time
        response_idx = -1
        for t_leave in boss_schedule.schedule:
            if t_leave > curr_time:
                break
            response_idx+=1
        return response_idx
    
    def create_boss_obstacle(x,y,z, yaw, point_idx):
        marker = Marker()
        marker.id = point_idx
        marker.header = boss_msg.header
        marker.ns = 'static_obs'
        marker.type = 1 # CUBE
        marker.action = 0 # ADD/modify
        marker.scale.x = 0.42
        marker.scale.y = 0.19
        marker.scale.z = 0.188

        marker.pose = deepcopy(boss_msg.pose.pose)
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = z

        # TODO: not sure how this yaw code is working ... hey ho
        # q = [marker.pose.orientation.x, marker.pose.orientation.y,
        #         marker.pose.orientation.z, marker.pose.orientation.w]
        # yaw = euler_from_quaternion(q)[-1]
        marker.pose.position.x += 0.1285*np.cos(yaw)
        marker.pose.position.y += 0.1285*np.sin(yaw)
        marker.pose.position.z = 0 
        
        # a modest green
        marker.color.r = 0
        marker.color.g = 200
        marker.color.b = 0
        marker.color.a = 0.5
        marker.lifetime = rospy.Duration(0)

        return marker
    curr_response_idx, boss_ref_path_positions, curr_boss_ref_path_kdtree = -1, None, None
    while not rospy.is_shutdown():
        if boss_state_buffer.new_data_available:
            # update boss ref path if out of date
            response_idx = response_idx_curr_boss_warehouse(start_time, boss_schedule)
            if response_idx != curr_response_idx:
                curr_response_idx = response_idx
                start_warehouse_idx = boss_schedule.start_warehouse_index[response_idx]
                end_warehouse_idx = boss_schedule.goal_warehouse_index[response_idx]
                print(f'updating boss start ({start_warehouse_idx}) and end warehouse {end_warehouse_idx}')
                boss_x_start, boss_y_start = warehouse_positions[start_warehouse_idx]
                boss_x_goal, boss_y_goal = warehouse_positions[end_warehouse_idx]
                boss_plan_request = PlanRequest([boss_x_start, boss_y_start], [boss_x_goal, boss_y_goal])
                curr_boss_ref_path = plan_client(boss_plan_request).path
                boss_ref_path_positions = []
                for point in curr_boss_ref_path.poses:
                    x = point.pose.position.x # centerline x component
                    y = point.pose.position.y
                    boss_ref_path_positions.append([x,y])
                curr_boss_ref_path_kdtree = KDTree(np.array(boss_ref_path_positions), leaf_size=2)
            
            boss_msg = boss_state_buffer.readFromRT()
            boss_position = boss_msg.pose.pose.position
            boss_x, boss_y = boss_position.x, boss_position.y
            _, closest_ref_path_idx = curr_boss_ref_path_kdtree.query([[boss_x, boss_y]], k=1)
            closest_ref_path_idx = closest_ref_path_idx[0][0] # weird bodgery
            
            # had a problem with obstacles not disappearing
            boss_obs_msg = MarkerArray()
            # marker = create_boss_obstacle(0,0,0,0, 0)
            # # marker.id = 3 # deletes all objects
            # boss_obs_msg.markers = [marker]
            boss_obstacle_publisher.publish(boss_obs_msg)

            # actually add in obstacles
            boss_obs_msg = MarkerArray()
            obstacles_list = []
            # obstacle only every other point
            for point_idx in range(closest_ref_path_idx,closest_ref_path_idx+7):
                point_idx = point_idx if point_idx < len(boss_ref_path_positions) else -1 
                x, y  = boss_ref_path_positions[point_idx]
                z, yaw = 0, 0
                marker = create_boss_obstacle(x,y,z,yaw, point_idx - closest_ref_path_idx)
                boss_obs_msg.markers.append(marker)
                obstacles_list.append([x, y, .1])
            marker = create_boss_obstacle(boss_x,boss_y,0,0, len(boss_obs_msg.markers))
            boss_obs_msg.markers.append(marker)
            obstacles_list.append([boss_x,boss_y, .1])

            boss_obstacle_publisher.publish(boss_obs_msg)
            # obstacles_kd_tree = KDTree(np.array(obstacles_list)[:,:2], leaf_size=2)
        else:
            rospy.sleep(0.1)


if __name__ == '__main__':
    rospy.init_node('path_planning_node')
    rospy.loginfo("Start path planning node")

    odom_msg_buffer = RealtimeBuffer()

    # odom_sub is for pose
    odom_topic = get_ros_param('~odom_topic', '/slam_pose')
    odom_sub = rospy.Subscriber(odom_topic, Odometry, odometry_callback, queue_size=10)
    path_pub = rospy.Publisher('/Routing/Path', PathMsg, queue_size=1)

    # side tasks
    side_task_client = rospy.ServiceProxy('/SwiftHaul/SideTask', Task)
    reward_client = rospy.ServiceProxy('/SwiftHaul/GetReward', Reward)

    static_obstacle_dict = {}
    def static_obs_callback(markers_msg):
        '''callback function for static obstacle topic'''
        # reset dict every call
        for marker in markers_msg.markers:
            idx, verts = get_obstacle_vertices(marker)
            static_obstacle_dict[idx] = verts
    static_obs_topic = get_ros_param('~static_obs_topic', '/Obstacles/Static')
    static_obs_sub = rospy.Subscriber(static_obs_topic, MarkerArray, static_obs_callback, queue_size=10)
            
    eps = 0.25
    config_path = __file__.replace("scripts/path_planner_task2_node.py", "task2.yaml")
    warehouse_positions = []
    with open(config_path, 'r') as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)
        for warehouse_letter in ['A', 'B', 'C', 'D', 'E']:
            curr_warehouse = config_dict[f'warehouse_{warehouse_letter}']
            warehouse_positions.append(curr_warehouse['location'])

    rospy.wait_for_service('/routing/plan')
    plan_client = rospy.ServiceProxy('/routing/plan', Plan)

    # initialize goals to far, far away
    x_goal = 5 # x coordinate of the goal ## temp values
    y_goal = 7 # y coordinate of the goal
    first_time = True

    path_msg = None
    t_last_pub = None

    # Update the static obstacles
    obstacles_list = []

    def update_obs_tree():
        '''create updated KD tree of obstacles'''
        global obstacles_list
        obstacles_list = [] # reset every time we call
        static_obstacle_dict_copy = deepcopy(static_obstacle_dict)
        for vertices in static_obstacle_dict_copy.values():
            x_mean = np.mean(vertices[:, 0])
            y_mean = np.mean(vertices[:, 1])
            r = np.sqrt((vertices[0, 0] - x_mean)**2 + (vertices[0, 1] - y_mean)**2)
            obstacles_list.append([x_mean, y_mean, r])
        if len(obstacles_list) == 0:
            obstacles_list = [[1000000, 1000000, 0.1]]
            return KDTree(np.array([[1000000, 1000000]]), leaf_size=2)
        return KDTree(np.array(obstacles_list)[:,:2], leaf_size=2)
    


    # make a kd tree for x, y coordinates of obstacles
    obstacles_kd_tree = update_obs_tree()
    waiting_iters = 0
    while np.asarray(obstacles_kd_tree.data)[0][0] == 1000000.0 and waiting_iters < 20:
        obstacles_kd_tree = update_obs_tree()
        waiting_iters += 1
        time.sleep(0.1)

    # start by assuming we want warehouse A
    target_warehouse = 0 

    # wait for ILQR to warm up before starting SwiftHaul
    time.sleep(10)

    swift_haul_client = rospy.ServiceProxy('/SwiftHaul/Start', Empty)
    swift_haul_client(EmptyRequest())
    boss_obstacle_thread = Thread(target=boss_obstacle_target)
    boss_obstacle_thread.start()
    
    while not rospy.is_shutdown():
        if control_state_buffer.new_data_available:
            odom_msg = control_state_buffer.readFromRT()

            x_start = odom_msg.pose.pose.position.x # x coordinate of the start
            y_start = odom_msg.pose.pose.position.y # y coordinate of the start
            if first_time or \
                (abs(x_start-x_goal) < eps and abs(y_start - y_goal) < eps) or \
                (rospy.Time.now() - t_last_pub).to_sec() > 0.5:
                first_time = False
                obstacles_kd_tree = update_obs_tree()
                
                if (abs(x_start-x_goal) < eps and abs(y_start - y_goal) < eps): ## truck at the warehouse
                    time.sleep(2)
                    reward_request = RewardRequest()
                    reward_request.task = target_warehouse
                    reward_response = reward_client(reward_request)
                    # the log prints out if this ^^ was a success    
                                    
                    # wait for a task to be available
                    curr_task = -1
                    curr_task_timeout = 0
                    while curr_task == -1:
                        # curr_task_timeout += 1
                        # if curr_task_timeout > 20: 
                        #     print("Took too long getting next task!!!!!")
                        #     break

                        # hopefully finished the old task, time for new task
                        task_response = side_task_client(TaskRequest())
                        curr_task = task_response.task
                        print(f'Got task {curr_task}')
                        print(f'Predicted reward: {task_response.reward}')
                        if curr_task == -1: time.sleep(1) # have to wait 5 seconds
                    target_warehouse = curr_task

                # Get new goal locations
                print(f'Getting warehouse id {target_warehouse}: {warehouse_positions[target_warehouse]}')
                x_goal = warehouse_positions[target_warehouse][0]
                y_goal = warehouse_positions[target_warehouse][1]
                plan_request = PlanRequest([x_start, y_start], [x_goal, y_goal])
                plan_response = plan_client(plan_request)
                x = []
                y = []
                width_L = [] ## bounds on road
                width_R = []
                speed_limit = []

                path_msg = plan_response.path
                final_path_msg = deepcopy(path_msg)
                start_pose, end_pose = path_msg.poses[0], path_msg.poses[-1]    
                path_ref_for_tangs = msg_to_ref_path(path_msg.poses)
                switched_point_idxs = set()
                for point_idx, point in enumerate(path_msg.poses):
                    x = point.pose.position.x # centerline x component
                    y = point.pose.position.y # centerline y component
                    width_L = point.pose.orientation.x
                    width_R = point.pose.orientation.y
                    speed_limit = point.pose.orientation.z ## should always be zero? guessing from printout
                    
                    dist_to_obs, closest_obs_idxs = obstacles_kd_tree.query([[x,y]], k=1)
                    closest_obs_idxs = closest_obs_idxs[0] # weird bodgery
                    for dist_to_obs, obs in zip(np.array(dist_to_obs).reshape(-1), np.array(obstacles_list)[closest_obs_idxs,:]):
                        x_obs, y_obs, r_obs = obs
                        # switch sides if inside object or close
                        if dist_to_obs < r_obs + 0.04:
                            switched_point_idxs.add(point_idx)
                            # -- calculate new x and y --
                            _, tang_gradient, _  = path_ref_for_tangs.get_closest_pts(np.array([x_obs, y_obs]))
                            tang_gradient = tang_gradient[0][0] + np.pi/2
                            deg_tang_gradient = tang_gradient * 180 / np.pi
                            new_x = x + (width_L - width_R) * np.cos(tang_gradient)
                            new_y = y + (width_L - width_R) * np.sin(tang_gradient)
                            path_msg.poses[point_idx].pose.position.x = new_x
                            path_msg.poses[point_idx].pose.position.y = new_y

                            # -- flip width l and r --
                            path_msg.poses[point_idx].pose.orientation.x = width_R
                            path_msg.poses[point_idx].pose.orientation.y = width_L

                final_path_msg.poses = [start_pose] # keep first index
                for point_idx in range(1, len(path_msg.poses)):
                    number_points_back = 6
                    nearby_set = set(range(point_idx-number_points_back, point_idx+number_points_back+1)) - {0}
                    overlap = nearby_set.intersection(switched_point_idxs)
                    if point_idx not in switched_point_idxs and len(overlap) > 0:
                        continue
                    # for the points just before or after a switch, set speed limit
                    near_change = 3
                    near_change_set = set(range(point_idx-number_points_back-near_change, point_idx+number_points_back+near_change+1)) - {0} - nearby_set
                    near_change_overlap = near_change_set.intersection(switched_point_idxs)
                    if point_idx in switched_point_idxs or len(near_change_overlap) > 0:
                        speed_limit = 1.2
                        path_msg.poses[point_idx].pose.orientation.z = speed_limit
                    final_path_msg.poses.append(path_msg.poses[point_idx])
                final_path_msg.poses[-1] = end_pose # keep last index

                final_path_msg.header = odom_msg.header
                path_pub.publish(final_path_msg)
                t_last_pub = rospy.Time.now()
                
                print(f'New reference path written from ({x_start}, {y_start}) to ({x_goal}, {y_goal})')
            
        rospy.sleep(0.1)