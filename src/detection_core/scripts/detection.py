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
from mmdet.apis import init_detector, inference_detector
import mmcv
sys.path.append('/opt/ros/melodic/lib/python2.7/dist-packages')

class Detector:
    def __init__(self, 
                input_image_topic="/usb_cam/image_raw",
                config_file="configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py",
                checkpoint_file="checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth",
                score_thr=0.8,
                box_color = (0, 255, 0),
                text_color = (0, 255, 0),
                mask_color = None
                ):
        self.image = None
        self.image_height = None
        self.image_width = None
        self.image_pub = rospy.Publisher("image_detection", Image)
        self.class_pub = rospy.Publisher("detection_classes", String)
        self.score_pub = rospy.Publisher("detection_scores", String)
        self.image_sub = rospy.Subscriber(input_image_topic, Image, self.image_callback, queue_size=1)
        self.config_file =  config_file
        self.checkpoint_file = checkpoint_file
        self.model = init_detector(self.config_file, self.checkpoint_file, device='cuda:0')
        self.score_thr = score_thr
        self.classes = self.model.CLASSES
        self.box_color = box_color
        self.text_color = text_color
        self.mask_color = mask_color

    def image_callback(self, data):

        try:
            # Note: cv_bridge is not used due to conflict with python3
            self.image = np.frombuffer(data.data, dtype=np.uint8).reshape(data.height, data.width, -1)
            self.image_height = data.height
            self.image_width = data.width
        except:
            rospy.logerr("Unable to load image stream ... ...")

def main():
    rospy.loginfo("Running object detection node")
    # pub = rospy.Publisher('chatter', String, queue_size=10)
    rospy.init_node('detector', anonymous=False)
    rate = rospy.Rate(10)
    detector = Detector()
    while not rospy.is_shutdown():
        # hello_str = "hello world %s" % rospy.get_time()
        # rospy.loginfo(hello_str)
        # pub.publish(hello_str)

        ### Run inference
        result = inference_detector(detector.model, detector.image)

        if isinstance(result, tuple):
            bbox_result, segm_result = result
            if isinstance(segm_result, tuple):
                segm_result = segm_result[0]  # ms rcnn
        else:
            bbox_result, segm_result = result, None
        bboxes = np.vstack(bbox_result)
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)

        ##### draw segmentation masks
        segms = None
        if segm_result is not None and len(labels) > 0:  # non empty
            segms = mmcv.concat_list(segm_result)
            if isinstance(segms[0], torch.Tensor):
                segms = torch.stack(segms, dim=0).detach().cpu().numpy()
            else:
                segms = np.stack(segms, axis=0)

        assert bboxes.ndim == 2, f"bboxes ndim should be 2, but its ndim is {bboxes.ndim}."
        assert labels.ndim == 1, f"labels ndim should be 1, but its ndim is {labels.ndim}."
        assert bboxes.shape[0] == labels.shape[0], f"bboxes.shape[0] and labels.shape[0] should have the same length."
        assert bboxes.shape[1] == 4 or bboxes.shape[1] == 5, f"bboxes.shape[1] should be 4 or 5, but its {bboxes.shape[1]}."

        ##### Filter result based on score_thr
        if detector.score_thr > 0:
            assert bboxes.shape[1] == 5
            scores = bboxes[:, -1]
            inds = scores > detector.score_thr
            bboxes = bboxes[inds, :]
            labels = labels[inds]
            if segms is not None:
                segms = segms[inds, ...]

        ##### Get mask color
        mask_colors = []
        if labels.shape[0] > 0:
            if detector.mask_color is None:
                # random color
                np.random.seed(42)
                mask_colors = [
                    np.random.randint(0, 256, (1, 3), dtype=np.uint8)
                    for _ in range(max(labels) + 1)
                ]
            else:
                # specify  color
                mask_colors = [
                    np.array(mmcv.color_val(detector.mask_color)[::-1], dtype=np.uint8)
                ] * (
                    max(labels) + 1)

        # ### Project result on image (Mask projection TBC)
        for idx, (bbox, label, score) in enumerate(zip(bboxes, labels, scores)):
            topLeft = (int(round(bbox[0])), int(round(bbox[1])))
            bottomRight = (int(round(bbox[2])), int(round(bbox[3])))
            text_x = topLeft[0] + 5
            if topLeft[1] < 50:
                text_y = topLeft[1] + 20
            else: 
                text_y = topLeft[1] - 5
            text = str(detector.classes[label]) + ": " + str(round(score, 4))
            detector.image = cv2.rectangle(detector.image, topLeft, bottomRight, detector.box_color, 2)
            detector.image = cv2.putText(detector.image, text, (text_x, text_y), cv2.FONT_HERSHEY_PLAIN, 1.5, detector.text_color, 2)

        ### Construct output image (Note: cv_bridge is not used due to conflict with python3)
        msg = Image()
        msg.header.stamp = rospy.Time.now()
        msg.height = detector.image_height
        msg.width = detector.image_width
        msg.encoding = "rgb8"
        msg.is_bigendian = False
        msg.step = 3 * detector.image_width
        msg.data = np.array(detector.image).tobytes()
        detector.image_pub.publish(msg)
        rate.sleep()

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
    
##### Default model classes
# model.classes: 
#     ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 
#     'traffic_light', 'fire_hydrant', 'stop_sign', 'parking_meter', 'bench', 'bird', 'cat', 
#     'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 
#     'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports_ball', 
#     'kite', 'baseball_bat', 'baseball_glove', 'skateboard', 'surfboard', 'tennis_racket', 
#     'bottle', 'wine_glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 
#     'sandwich', 'orange', 'broccoli', 'carrot', 'hot_dog', 'pizza', 'donut', 'cake', 'chair', 
#     'couch', 'potted_plant', 'bed', 'dining_table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 
#     'keyboard', 'cell_phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 
#     'clock', 'vase', 'scissors', 'teddy_bear', 'hair_drier', 'toothbrush')