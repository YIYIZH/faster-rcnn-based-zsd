import mmdet
from mmdet.apis import init_detector, inference_detector, show_result
import mmcv

config_path = 'configs/'
model_path = 'models/'
config_file = config_path+'faster_rcnn_r101_fpn_1x.py'
checkpoint_file = model_path+'faster_rcnn_r101_fpn_1x_20181129-d1468807.pth'

# Loading the model from the config and saved checkpoint, inference to GPU
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# test a single image and show the results
img = 'test.jpg'  # or img = mmcv.imread(img), which will only load it once
result = inference_detector(model, img)
# visualize the results in a new window
#show_result(img, result, model.CLASSES, show=False)
# or save the visualization results to image files
show_result(img, result, model.CLASSES, show=False, out_file='result.jpg')