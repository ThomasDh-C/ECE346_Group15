
1. On truck: 
    1. voltage = 16v
    2. connect to ECE346 wifi and `ssh nvidia@192.168.1.215` --> nvidia
    3. `cd ~/StartUp`
    4. `./start_ros.sh 192.168.1.215`
2. On laptop (terminal 1) start localisation: 
    1. `cd ROS_Core` 
    2. `conda activate ros_base2`
    3. `catkin_make` if you have new packages
    4. `source network_ros_client.sh 192.168.1.215 <PC_IP>` to make them talk to each other
    5. launch visualization nodes:
        1. `source devel/setup.bash`
        2. `roslaunch racecar_interface visualization.launch`
    6. Wait 30 seconds for pop up windows
    7. In `Service Caller` click the refresh button and choose `/SLAM/start_slam` from the drop-down menu. Finally, click the call button to start localisation
1. RUNNING CONTROL PROGRAM OPTION 1: On laptop (terminal 2)
    1. `cd ROS_Core`
    2. `conda activate ros_base2`
    3. `catkin_make` if you have new packages
    4. `source network_ros_client.sh 192.168.1.215 <PC_IP>` to make them talk to each other
    5. launch visualization nodes:
        1. `source devel/setup.bash`
        2. `[ROSLAUNCH YOUR PROGRAM]` ... written as roslaunch < ROS Package > < Launch File >
1. RUNNING CONTROL PROGRAM OPTION 2: On truck (terminal 2)
    1. assume connected to ECE346 wifi from above and `ssh nvidia@192.168.1.215` --> nvidia
    2. `cd ~/Documents/ECE346/ROS_Core`
    3. `git submodule update --init --recursive`
    4. `source network_ros_host.sh 192.168.1.215`
    5. `source devel/setup.bash`
    6. `catkin_make` if you have new packages
    7. `[ROSLAUNCH YOUR PROGRAM]` ... written as roslaunch < ROS Package > < Launch File >