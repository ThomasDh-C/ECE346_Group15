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

from tf.transformations import euler_from_quaternion
from final_project.srv import Task, TaskRequest, TaskResponse, Reward, RewardRequest, RewardResponse


control_state_buffer = RealtimeBuffer()
boss_state_buffer = RealtimeBuffer()

# A subscriber callback for odom
def odometry_callback(odom_msg):
    '''
    Subscriber callback function of the robot pose
    '''
    control_state_buffer.writeFromNonRT(odom_msg)

def boss_odometry_callback(odom_msg):
    '''Subscriber to boss pose'''
    boss_state_buffer.writeFromNonRT(odom_msg)


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


if __name__ == '__main__':
    rospy.init_node('path_planning_node')
    rospy.loginfo("Start path planning node")
    
    odom_msg_buffer = RealtimeBuffer()

    # odom_sub is for pose
    odom_topic = get_ros_param('~odom_topic', '/slam_pose')
    odom_sub = rospy.Subscriber(odom_topic, Odometry, odometry_callback, queue_size=10)
    path_pub = rospy.Publisher('/Routing/Path', PathMsg, queue_size=1)


    # get boss pose
    boss_odom_topic = get_ros_param('~boss_odom_topic', '/Boss/Pose')
    boss_odom_subscriber = rospy.Subscriber(boss_odom_topic, Odometry, boss_odometry_callback, queue_size=10)
    static_obs_topic = get_ros_param('~static_obs_topic', '/Obstacles/Static')
    boss_obstacle_publisher = rospy.Publisher(static_obs_topic, MarkerArray, queue_size=1)

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
    static_obs_sub = rospy.Subscriber(static_obs_topic, MarkerArray, static_obs_callback, queue_size=10)

    # get warehouse positions from yaml file
    config_path = __file__.replace("scripts/path_planner_node.py", "task2.yaml")
    warehouse_positions = []
    with open(config_path, 'r') as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)
        for warehouse_letter in ['A', 'B', 'C', 'D', 'E']:
            curr_warehouse = config_dict[f'warehouse_{warehouse_letter}']
            warehouse_positions.append(curr_warehouse['location'])
            
    eps = 0.25

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
    
    def create_boss_obstacle(x,y,z, yaw):
        marker = Marker()
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

    # make a kd tree for x, y coordinates of obstacles
    obstacles_kd_tree = update_obs_tree()
    waiting_iters = 0
    while np.asarray(obstacles_kd_tree.data)[0][0] == 1000000.0 and waiting_iters < 20:
        obstacles_kd_tree = update_obs_tree()
        waiting_iters += 1
        time.sleep(0.1)

    # start by assuming we want warehouse A
    target_warehouse = 0 
    while not rospy.is_shutdown():
        if control_state_buffer.new_data_available:
            odom_msg = control_state_buffer.readFromRT()

            x_start = odom_msg.pose.pose.position.x # x coordinate of the start
            y_start = odom_msg.pose.pose.position.y # y coordinate of the start
            if first_time or \
                (abs(x_start-x_goal) < eps and abs(y_start - y_goal) < eps) or \
                (rospy.Time.now() - t_last_pub).to_sec() > 0.5:
                first_time = False

                if boss_state_buffer.new_data_available:
                    boss_msg = boss_state_buffer.readFromRT()
                    boss_obs_msg = MarkerArray()
                    next_points = []
                    
                    for p in next_points:
                        x,y,z,yaw = p
                        marker = create_boss_obstacle(x,y,z,yaw)
                        boss_obs_msg.markers.append(marker)
                    #TODO: if nothing works, grab more code from task1_obstacle_deteciton_node.py
                    boss_obstacle_publisher.publish(boss_obs_msg)
                    time.sleep(0.05)
                    
                

                obstacles_kd_tree = update_obs_tree()
                
                if (abs(x_start-x_goal) < eps and abs(y_start - y_goal) < eps): ## truck at the warehouse
                    reward_response = reward_client(RewardRequest())
                    print(f'Completed last task? {reward_response.done}')
                    print(f'Current reward: {reward_response.total_reward}')
                    
                    # wait for a task to be available
                    curr_task = -1
                    curr_task_timeout = 0
                    while curr_task == -1:
                        # 
                        curr_task_timeout += 1
                        if curr_task_timeout > 20: 
                            print("Took too long getting next task!!!!!")
                            break

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