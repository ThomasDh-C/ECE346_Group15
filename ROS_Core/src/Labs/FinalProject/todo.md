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

## Task 2