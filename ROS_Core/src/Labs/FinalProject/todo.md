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
- make better ref-path - DONE
- update index correctly for next target - DONE
  - but kinda slow sometimes, can optimize later
- clean up launch file so only essential args
- only add close obstacles (this is inside receding horizon thread)
- add subscriber for obstacles in planner_node and see if it works
- Try to see how to run faster speed of the car (ask zixu)

- Obstacles on other side of middle lane boundary are still accounted for
  - Need to check if the nearest obstacle is in the current lanelet?



## Task 2
Greedy strategy
- Treat boss as "dynamic" obstacle
- Wait in front of first warehouse, grab first task, always accept and drive fast