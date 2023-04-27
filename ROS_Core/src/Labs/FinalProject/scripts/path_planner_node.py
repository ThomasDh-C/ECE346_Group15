#!/usr/bin/env python
import rospy
from std_msgs.msg import String
# from path_planner import PathPlanner
from nav_msgs.msg import Odometry
from task2_world import RealtimeBuffer
from utils import frs_to_obstacle, frs_to_msg, get_obstacle_vertices, get_ros_param
import yaml
from racecar_routing.srv import Plan, PlanRequest
from nav_msgs.msg import Path as PathMsg

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

if __name__ == '__main__':
    rospy.init_node('path_planning_node')
    rospy.loginfo("Start path planning node")
    
    odom_msg_buffer = RealtimeBuffer()
    # odom_sub is for pose
    odom_topic = get_ros_param('~odom_topic', '/slam_pose')
    odom_sub = rospy.Subscriber(odom_topic, Odometry, odometry_callback, queue_size=10)
    path_pub = rospy.Publisher('/Routing/Path', PathMsg, queue_size=1)

    EPS = 0.2
    config_path = __file__.replace("scripts/path_planner_node.py", "task1.yaml")
    goals = []
    goal_order = []
    with open(config_path, 'r') as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)
        goals = config_dict['goals']
        goal_order = config_dict['goal_order']

    rospy.wait_for_service('/routing/plan')
    plan_client = rospy.ServiceProxy('/routing/plan', Plan)
    # print("Done waiting for service: /routing/plan!")

    # initialize goals to far, far away
    x_goal = 5 # x coordinate of the goal ## temp values
    y_goal = 7 # y coordinate of the goal
    first_time = True
    current_waypoint = 1

    path_msg = None
    t_last_pub = None
    while not rospy.is_shutdown():
        if control_state_buffer.new_data_available:
            # print("new control state buffer data available")
            odom_msg = control_state_buffer.readFromRT()

            x_start = odom_msg.pose.pose.position.x # x coordinate of the start
            y_start = odom_msg.pose.pose.position.y # y coordinate of the start
            # print(f'nav_thread: x_start: {x_start}, y_start: {y_start}')
            if first_time or \
                (abs(x_start-x_goal) < EPS and abs(y_start - y_goal) < EPS) or \
                (rospy.Time.now() - t_last_pub).to_sec()> 3.0:
                first_time = False
                # current_waypoint += 1
                print(f'Getting the {current_waypoint}th waypoint: {goals[goal_order[current_waypoint]-1]}')
                # Get new goal locations
                # TODO: handle last waypoint!
                x_goal = goals[goal_order[current_waypoint]-1][0]
                y_goal = goals[goal_order[current_waypoint]-1][1]

                plan_request = PlanRequest([x_start, y_start], [x_goal, y_goal])
                plan_response = plan_client(plan_request)
                path_msg = plan_response.path

                path_msg.header = odom_msg.header
                path_pub.publish(path_msg)
                t_last_pub = rospy.Time.now()
                # path_callback(path_msg) TODO: do we need this?
                
                print(f'New reference path written from ({x_start}, {y_start}) to ({x_goal}, {y_goal})')
            
        rospy.sleep(0.1)
