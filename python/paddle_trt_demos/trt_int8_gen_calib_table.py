from paddle.fluid.core import AnalysisConfig
import numpy as np

import sys
sys.path.append("..")
from pd_model import Model

max_batch = 1

config = AnalysisConfig('../../models/resnet50/model', '../../models/resnet50/params')
config.switch_use_feed_fetch_ops(False)
config.switch_specify_input_names(True)
config.enable_memory_optim()
config.enable_use_gpu(100, 0)
config.enable_tensorrt_engine(workspace_size = 1<<30, 
          max_batch_size=max_batch,
		  min_subgraph_size=3,
		  precision_mode=AnalysisConfig.Precision.Int8,
		  use_static=False,
		  use_calib_mode=False)

model = Model(config)
data = np.ones((1, 3, 224, 224)).astype(np.float32)

result = model.run([data])
print result[0][:10]
