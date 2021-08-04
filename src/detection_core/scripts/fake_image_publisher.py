#!/usr/bin/env python

import sys
import os
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
import numpy as np

sys.path.remove("/opt/ros/melodic/lib/python2.7/dist-packages")
os.chdir(os.getenv("HOME") + "/ros/pytorch_object_detection/src/mmdetection/")
sys.path.append(os.getenv("HOME") + "/ros/pytorch_object_detection/src/mmdetection/")
# from mmdet.apis import init_detector, inference_detector
# import mmcv
# sys.path.append('/opt/ros/melodic/lib/python2.7/dist-packages')

class FakeImagePublisher:
    def __init__(self, 
                input_image_path="images/000000002532.jpg",
                output_image_topic="/usb_cam/image_raw"
                ):

        self.image_pub = rospy.Publisher(output_image_topic, Image, queue_size=10)
        self.image = cv2.imread(input_image_path)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.image_height, self.image_width, self.image_channels = self.image.shape

def main():
    rospy.loginfo("Running fake_image_publisher node")
    rospy.init_node('fake_image_publisher', anonymous=False)
    rate = rospy.Rate(10)
    fake_image_pub = FakeImagePublisher()
    while not rospy.is_shutdown():

        ### Construct output image (Note: cv_bridge is not used due to conflict with python3)
        msg = Image()
        msg.header.stamp = rospy.Time.now()
        msg.height = fake_image_pub.image_height
        msg.width = fake_image_pub.image_width
        msg.encoding = "rgb8"
        msg.is_bigendian = False
        msg.step = 3 * fake_image_pub.image_width
        msg.data = np.array(fake_image_pub.image).tobytes()
        fake_image_pub.image_pub.publish(msg)
        rate.sleep()

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
    
