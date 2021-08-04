from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import mmcv

# Specify the path to model config and checkpoint file
config_file = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
checkpoint_file = 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# test a single image and show the results
img = 'images/000000002532.jpg'  # or img = mmcv.imread(img), which will only load it once
result = inference_detector(model, img)

# show the results
show_result_pyplot(model, img, result)

# test a video and show the results
#video = mmcv.VideoReader('video.mp4')
#for frame in video:
#    result = inference_detector(model, frame)
#    model.show_result(frame, result, wait_time=1)
