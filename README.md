# vehicle-stop-check-ADAS
CPP code to check whether car is running or stopped.

This code is for checking whether a car is stopped or running using egocamera installed on car's dashboard.
It takes every 2 frames and find ORB keypoints and descriptors. If keypoints and descriptors doesnt change much, 
then it's concluded that the car is stopped.
