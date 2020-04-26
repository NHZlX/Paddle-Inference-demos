from paddle.fluid.core import AnalysisConfig

import sys
sys.path.append("..")
from pd_model import Model

import numpy as np

max_batch=1
# this is a simple resnet block for dynamci test.
config = AnalysisConfig('../../models/resnet_small')
config.switch_use_feed_fetch_ops(False)
config.switch_specify_input_names(True)
config.enable_memory_optim()
config.enable_use_gpu(100, 0)
config.enable_tensorrt_engine(workspace_size = 1<<30, 
          max_batch_size=max_batch, min_subgraph_size=3,
		  precision_mode=AnalysisConfig.Precision.Float32,
		  use_static=False, use_calib_mode=False)

config.set_trt_dynamic_shape_info({'image':[1, 3, 10, 10]}, 
		                              {'image':[1, 3, 224, 224]},
		                              {'image':[1, 3, 50, 50]})

model = Model(config)
data = np.ones((1, 3, 50, 50)).astype(np.float32)

result = model.run([data])
print result[0][:10]
