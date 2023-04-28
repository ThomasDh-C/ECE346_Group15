# Final project to-do

## Task 1
- Moving car around track
  - Figure out how to use lab 1/2ILQR
  - Figure out how to publish waypoints to /Routing/Path 
    - [--> publish to '/move_base_simple/goal']? ... does this make a beeline
      - NO
  - Figure out how to check if vehicle within epsilon of current waypoint

- Obstacle avoidance
  - Start using lab 2 static obstacle avoidance
  - Add Zixu's idea of moving the path segment around static obstacle to speed up ILQR
    - Using lanelet: get_section_width

## Task 1 - attempt 2
- control thread - takes in ilqr trajectory and steers car along it
- receding horizon thread - runs ilqr
- waypoint thread - publishes refpath to next waypoint (/Routing/Path)

roslaunch final_project task1_simulation.launch

- switch to jax - DONE
- update index correctly for next target
- clean up launch file so only essential args
- only add close obstacles (this is inside receding horizon thread)
- add subscriber for obstacles in planner_node and see if it works
- Try to see how to run faster speed of the car (ask zixu)



## Task 2