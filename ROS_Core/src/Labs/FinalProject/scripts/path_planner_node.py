#!/usr/bin/env python
import rospy
from std_msgs.msg import String
from nav_msgs.msg import Odometry
from task2_world import RealtimeBuffer, RefPath
from utils import frs_to_obstacle, frs_to_msg, get_obstacle_vertices, get_ros_param
from visualization_msgs.msg import MarkerArray
import yaml
from racecar_routing.srv import Plan, PlanRequest
from nav_msgs.msg import Path as PathMsg
import time
import numpy as np
from sklearn.neighbors import KDTree

control_state_buffer = RealtimeBuffer()

# A subscriber callback for odom
def odometry_callback(odom_msg):
    '''
    Subscriber callback function of the robot pose
    '''
    # Add the current state to the buffer
    # Controller thread will read from the buffer
    # Then it will be processed and add to the planner buffer
    # inside the controller thread
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

if __name__ == '__main__':
    rospy.init_node('path_planning_node')
    rospy.loginfo("Start path planning node")
    
    odom_msg_buffer = RealtimeBuffer()
    

    # odom_sub is for pose
    odom_topic = get_ros_param('~odom_topic', '/slam_pose')
    odom_sub = rospy.Subscriber(odom_topic, Odometry, odometry_callback, queue_size=10)
    path_pub = rospy.Publisher('/Routing/Path', PathMsg, queue_size=1)

    static_obstacle_dict = {}
    def static_obs_callback(markers_msg):
        '''callback function for static obstacle topic'''
        # reset dict every call
        for marker in markers_msg.markers:
            idx, verts = get_obstacle_vertices(marker)
            static_obstacle_dict[idx] = verts
    static_obs_topic = get_ros_param('~static_obs_topic', '/Obstacles/Static')
    static_obs_sub = rospy.Subscriber(static_obs_topic, MarkerArray, static_obs_callback, queue_size=10)

    config_path = __file__.replace("scripts/path_planner_node.py", "task1.yaml")
    goals = []
    goal_order = []
    with open(config_path, 'r') as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)
        goals = config_dict['goals']
        goal_order = config_dict['goal_order']

    eps = 1.5
    total_num_waypoints = len(goal_order)

    rospy.wait_for_service('/routing/plan')
    plan_client = rospy.ServiceProxy('/routing/plan', Plan)

    # initialize goals to far, far away
    x_goal = 5 # x coordinate of the goal ## temp values
    y_goal = 7 # y coordinate of the goal
    first_time = True
    current_waypoint = 1

    path_msg = None
    t_last_pub = None

    # Update the static obstacles
    obstacles_list = []
    while len(obstacles_list) == 0:
        for vertices in static_obstacle_dict.values():
            x_mean = np.mean(vertices[:, 0])
            y_mean = np.mean(vertices[:, 1])
            r = np.sqrt((vertices[0, 0] - x_mean)**2 + (vertices[0, 1] - y_mean)**2)
            obstacles_list.append([x_mean, y_mean, r])
        rospy.sleep(0.1)
    # make a kd tree for x, y coordinates of obstacles
    obstacles_kd_tree = KDTree(np.array(obstacles_list)[:,:2], leaf_size=2) 
    
    while not rospy.is_shutdown():
        if control_state_buffer.new_data_available:
            odom_msg = control_state_buffer.readFromRT()

            x_start = odom_msg.pose.pose.position.x # x coordinate of the start
            y_start = odom_msg.pose.pose.position.y # y coordinate of the start
            if first_time or (rospy.Time.now() - t_last_pub).to_sec()> 6.0:
                first_time = False
                
                if (abs(x_start-x_goal) < eps and abs(y_start - y_goal) < eps):
                    current_waypoint += 1
                    if current_waypoint >= total_num_waypoints: break

                print(f'Getting the {current_waypoint}th waypoint: {goals[goal_order[current_waypoint]-1]}')
                # Get new goal locations
                x_goal = goals[goal_order[current_waypoint]-1][0]
                y_goal = goals[goal_order[current_waypoint]-1][1]

                plan_request = PlanRequest([x_start, y_start], [x_goal, y_goal])
                plan_response = plan_client(plan_request)
                x = []
                y = []
                width_L = [] ## bounds on road
                width_R = []
                speed_limit = []

                path_msg = plan_response.path
                path_ref_for_tangs = msg_to_ref_path(path_msg.poses)
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
                        if dist_to_obs < r_obs + 0.3:
                            # print('---')
                            # print(f'Current position is x: {x:.2f}, y: {y:.2f}')
                            # print(f'Avoiding obstacle id {closest_obs_idxs[0]} with x: {x_obs:.2f}, y: {y_obs:.2f}, r: {r_obs:.2f} at a distance of {dist_to_obs:.2f}')
                            # -- calculate new x and y --
                            _, tang_gradient, _  = path_ref_for_tangs.get_closest_pts(np.array([x_obs, y_obs]))
                            tang_gradient = tang_gradient[0][0] + np.pi/2
                            deg_tang_gradient = tang_gradient * 180 / np.pi
                            # print(f'Angle in degrees of path (-180 to 180 with x=1 = 0o): {deg_tang_gradient:.2f}')
                            new_x = x + (width_L - width_R) * np.cos(tang_gradient)
                            new_y = y + (width_L - width_R) * np.sin(tang_gradient)
                            path_msg.poses[point_idx].pose.position.x = new_x
                            path_msg.poses[point_idx].pose.position.y = new_y
                            # print(f'New position is x: {new_x:.2f}, y: {new_y:.2f}')

                            # -- flip width l and r --
                            # print(f'Width l ({width_L:.2f}) and r pre-flip ({width_R:.2f})')
                            path_msg.poses[point_idx].pose.orientation.x = width_R
                            path_msg.poses[point_idx].pose.orientation.y = width_L
                            # print(f'Width l ({path_msg.poses[point_idx].pose.orientation.x:.2f}) and r post-flip ({path_msg.poses[point_idx].pose.orientation.y:.2f})')

                path_msg.header = odom_msg.header
                path_pub.publish(path_msg)
                t_last_pub = rospy.Time.now()
                
                print(f'New reference path written from ({x_start}, {y_start}) to ({x_goal}, {y_goal})')
            
        rospy.sleep(0.1)

    print("Obstacle course complete!")