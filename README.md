### Pytorch object detection via ROS

A quick test for OpenMMLAB pytorch object detection via ROS framework

##### Source
- OpenMMLab MMDetection Pytorch toolbox: https://github.com/open-mmlab/mmdetection
- ROS driver for V4L USB camera: https://github.com/ros-drivers/usb_cam

##### Installation
- Install environment and MMDetection: https://mmdetection.readthedocs.io/en/latest/get_started.html
- conda activate openmmlab
- pip install rospkg

##### Build 
- catkin build

##### Run
- Using usb camera video:
  - roslaunch detection_core detection_2d.launch use_camera:=true

- Using static image:
  - roslaunch detection_core detection_2d.launch use_camera:=false