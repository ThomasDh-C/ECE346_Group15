import rospy
from racecar_routing.srv import Plan, PlanResponse, PlanRequest
from task2_world.util import RefPath, RealtimeBuffer
from nav_msgs.msg import Odometry

# This is a collection of janky boilerplate. Good luck.
# see todo.md for more info on what to do here
class SmartNavigator():
    def __init__(self):
        # Read ROS topic names to subscribe 
        rospy.wait_for_service('/routing/plan')
        plan_client = rospy.ServiceProxy('/routing/plan', Plan)
    
        # set up buffer 
        self.pose_buffer = RealtimeBuffer()
    
        self.pose_sub = rospy.Subscriber(self.odom_topic, Odometry, self.odometry_callback, queue_size=10)
    
    def plan(self):
        # Get the current pose from the buffer
        # [x, y, v, psi, timestamp]
        odom_msg = self.pose_buffer.readFromRT()
        x_start = odom_msg.pose.pose.position.x # x coordinate of the start
        y_start = odom_msg.pose.pose.position.y # y coordinate of the start

        x_goal = 0 # x coordinate of the goal
        y_goal = 0 # y coordinate of the goal

        plan_request = PlanRequest([x_start, y_start], [x_goal, y_goal])
        plan_response = plan_client(plan_request)

        # The following script will generate a reference path in [RefPath](scripts/task2_world/util.py#L65) class, which has been used in your Lab1's ILQR planner
        x = []
        y = []
        width_L = [] ## bounds on road
        width_R = []
        speed_limit = []

        for waypoint in plan_respond.path.poses:
            x.append(waypoint.pose.position.x)
            y.append(waypoint.pose.position.y)
            width_L.append(waypoint.pose.orientation.x)
            width_R.append(waypoint.pose.orientation.y)
            speed_limit.append(waypoint.pose.orientation.z)
                    
        centerline = np.array([x, y])

        # This is the reference path that we passed to the ILQR planner in Lab1
        ref_path = RefPath(centerline, width_L, width_R, speed_limit, loop=False)

    def odometry_callback(self, odom_msg):
        '''
        Subscriber callback function of the robot pose
        '''
        # Add the current state to the buffer
        # Controller thread will read from the buffer
        # Then it will be processed and add to the planner buffer 
        # inside the controller thread
        self.pose_buffer.writeFromNonRT(odom_msg)

        '''
        Extract the state from the full state [x, y, v, psi, timestamp]
        '''
        t_slam = odom_msg.header.stamp.to_sec()
        # get the state from the odometry message
        q = [odom_msg.pose.pose.orientation.x, odom_msg.pose.pose.orientation.y, 
                odom_msg.pose.pose.orientation.z, odom_msg.pose.pose.orientation.w]
        # get the heading angle from the quaternion
        psi = euler_from_quaternion(q)[-1]
        
        state_global =np.array([
                            odom_msg.pose.pose.position.x,
                            odom_msg.pose.pose.position.y,
                            odom_msg.twist.twist.linear.x,
                            psi,
                            t_slam
            ])
        return state_global


## publish static obstacles?
# self.static_obs_publisher = rospy.Publisher(self.static_obs_topic, MarkerArray, queue_size=1)

# pose_sub = message_filters.Subscriber(self.odom_topic, Odometry)
# # This subscribe to the 2D Nav Goal in RVIZ
# tag_sub = message_filters.Subscriber(self.static_obs_tag_topic, AprilTagDetectionArray)
# self.static_obs_detection = message_filters.ApproximateTimeSynchronizer([pose_sub, tag_sub], 10, 0.1)
# self.static_obs_detection.registerCallback(self.detect_obs)

if __name__ == '__main__':
    # rospy.init_node('static_obstacle_detection_node')
    # rospy.loginfo("Start static obstacle detection node")
    smart_nav = SmartNavigator()
    smart_nav.plan()
    rospy.spin()

