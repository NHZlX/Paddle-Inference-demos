import numpy as np
from paddle.fluid.core import AnalysisConfig
from paddle.fluid.core import create_paddle_predictor

class Model:
   def __init__(self, model_file, params_file, use_mkldnn=True, use_gpu = False, device_id = 0):
     config = AnalysisConfig(model_file, params_file)
     config.switch_use_feed_fetch_ops(False)
     config.switch_specify_input_names(True)
     config.enable_memory_optim()

     if use_gpu:
       print ("ENABLE_GPU")
       config.enable_use_gpu(100, device_id)

     if use_mkldnn: 
       config.enable_mkldnn()
     self.predictor = create_paddle_predictor(config)
     
   def run(self, img_list):

     input_names = self.predictor.get_input_names()
     for i, name in enumerate(input_names):
       input_tensor = self.predictor.get_input_tensor(input_names[i])
       input_tensor.reshape(img_list[i].shape)   
       input_tensor.copy_from_cpu(img_list[i].copy())

     self.predictor.zero_copy_run()

     results = []
     output_names = self.predictor.get_output_names()

     for i, name in enumerate(output_names):
       output_tensor = self.predictor.get_output_tensor(output_names[i])
       output_data = output_tensor.copy_to_cpu()
       results.append(output_data)

     return results



mn = Model('./pyramidbox_lite/model', './pyramidbox_lite/params')

import cv2
from preprocess import face_detect_preprocess

img = cv2.imread('../assets/test_mask_detection.jpg')
img_h, img_w, c = img.shape
img = face_detect_preprocess(img, 0.3)

print (img.shape)

result = mn.run([img])
print (result[0])

def get_faces(data, h, w):
  faces_loc = []
  for d in data:
    if d[1] >= 0.5:
      x_min = max(d[2] * w, 0)
      y_min = max(d[3] * h, 0)
      x_h = min((d[4] - d[2]) * w, w)
      y_w = min((d[5] - d[3]) * h, h)
      faces_loc.append([int(x_min), int(y_min), int(x_h), int(y_w)])
  return faces_loc


faces = get_faces(result[0], img_h, img_w)
print faces
