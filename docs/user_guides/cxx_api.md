
# 使用C++API

为了简单方便地进行推理部署，飞桨提供了一套高度优化的C++ API推理接口。下面对各主要API使用方法进行详细介绍。    

## API详细介绍

### 使用AnalysisConfig管理推理配置
AnalysisConfig管理AnalysisPredictor的推理配置，提供了模型路径设置、推理引擎运行设备选择以及多种优化推理流程的选项。  
配置方法如下。  
#### 1. 设置模型和参数路径
从磁盘加载模型时，根据模型和参数文件存储方式不同，设置AnalysisConfig加载模型和参数的路径有两种形式：
* 非combined形式：模型文件夹`model_dir`下存在一个模型文件和多个参数文件时，传入模型文件夹路径，模型文件名默认为`__model__`。

`config->SetModel("./model_dir");`
* combined形式：模型文件夹`model_dir`下只有一个模型文件`model`和一个参数文件`params`时，传入模型文件和参数文件路径。
`config->SetModel("./model_dir/model", "./model_dir/params");`

#### 2. 配置推理 
##### 配置CPU推理 
设置使用CPU推理：  
`config->DisableGpu();		  // 禁用GPU`
CPU下性能调优配置：  

```
config->EnableMKL();          			// 开启MKL，可加速CPU推理
config->EnableMKLDNN();	  	  		// 开启MKLDNN，可加速CPU推理
config->SetCpuMathLibraryNumThreads(10); 	// 设置CPU数学库线程数，CPU核心数支持时调高一些可加速推理
```
##### 配置GPU推理  
设置使用GPU推理：  
```
config->EnableUseGpu(100, 0); // 初始化100M显存，使用GPU ID为0
config->GpuDeviceId();        // 返回正在使用的GPU ID
```
GPU下性能调优，可以在安装了 TensorRT 的环境下打开 TensorRT 子图加速引擎： 

```
// 开启TensorRT推理，可提升GPU推理性能，需要使用带TensorRT的推理库
config->EnableTensorRtEngine(1 << 20      	   /*workspace_size*/,   
                        	 batch_size        /*max_batch_size*/,     
                        	 3                 /*min_subgraph_size*/, 
                       		 AnalysisConfig::Precision::kFloat32 /*precision*/, 
                        	 false             /*use_static*/, 
                        	 false             /*use_calib_mode*/);
```
通过计算图分析，Paddle可以自动将计算图中部分子图融合，并调用NVIDIA的 TensorRT 来进行加速。
#### 3. 通用优化配置  

```
config->SwitchIrOptim(true);  // 开启计算图分析优化，包括OP融合等
config->EnableMemoryOptim();  // 开启内存/显存复用
```
**Note**: 使用ZeroCopyTensor必须设置： 
`config->SwitchUseFeedFetchOps(false);  // 关闭feed和fetch OP使用，使用ZeroCopy接口必须设置此项`


### 使用 ZeroCopyTensor 管理输入/输出
ZeroCopyTensor 是 AnalysisPredictor 的输入/输出数据结构。ZeroCopyTensor 的使用可以避免推理时候准备输入以及获取输出时多余的数据复制，提高推理性能。    
**Note:**  
使用ZeroCopyTensor，务必在创建config时设置`config->SwitchUseFeedFetchOps(false);`。  

```
// 1、通过创建的AnalysisPredictor获取输入和输出的tensor
auto input_names = predictor->GetInputNames();
auto input_t = predictor->GetInputTensor(input_names[0]);
auto output_names = predictor->GetOutputNames();
auto output_t = predictor->GetOutputTensor(output_names[0]);

///2、对tensor进行reshape
input_t->Reshape({batch_size, channels, height, width});

// 通过copy_from_cpu接口，将cpu数据输入；通过copy_to_cpu接口，将输出数据copy到cpu
input_t->copy_from_cpu<float>(input_data /*数据指针*/);
output_t->copy_to_cpu(out_data /*数据指针*/);

///3、设置LOD 
std::vector<std::vector<size_t>> lod_data = {{0}, {0}};
input_t->SetLoD(lod_data);

///4、获取Tensor数据指针
float *input_d = input_t->mutable_data<float>(PaddlePlace::kGPU);  // CPU下使用PaddlePlace::kCPU
int output_size;
float *output_d = output_t->data<float>(PaddlePlace::kGPU, &output_size);
```


### 使用AnalysisPredictor进行高性能推理
飞桨采用 AnalysisPredictor 进行推理。AnalysisPredictor 是一个高性能推理引擎，该引擎通过对计算图的分析，完成对计算图的一系列的优化（如OP的融合、内存/显存的优化、 MKLDNN，TensorRT 等底层加速库的支持等），能够大大提升推理性能。
为了展示完整的推理流程，下面是一个使用 AnalysisPredictor 进行推理的完整示例，其中涉及到的具体概念和配置会在后续部分展开详细介绍。

```c++
#include "paddle_inference_api.h"

namespace paddle {
void CreateConfig(AnalysisConfig* config, const std::string& model_dirname) {
  // 模型从磁盘进行加载，设置模型路径
  config->SetModel(model_dirname + "/model",                                                                                             
                   model_dirname + "/params");  
  // config->SetModel(model_dirname);
  // 如果模型从内存中加载，可以使用SetModelBuffer接口
  // config->SetModelBuffer(prog_buffer, prog_size, params_buffer, params_size); 
    
  /* for gpu */
  config->EnableUseGpu(100 /*设定GPU初始显存池为100MB*/,  0 /*设定使用GPU ID为0*/); //开启GPU推理
  
  /* for cpu 
  config->DisableGpu();						// 关闭GPU推理
  config->EnableMKLDNN();   				// 开启MKLDNN加速
  config->SetCpuMathLibraryNumThreads(10); 	// 设置CPU数学库线程数
  */
 
  // 使用ZeroCopyTensor，此处必须设置为false
  config->SwitchUseFeedFetchOps(false);
  // 若输入为多个，此处必须设置为true
  config->SwitchSpecifyInputNames(true);
  config->SwitchIrDebug(true); 		// 可视化调试选项，若开启，则会在每个图优化过程后生成dot文件
  // config->SwitchIrOptim(false); 	// 默认为true。如果设置为false，关闭所有优化
  // config->EnableMemoryOptim(); 	// 开启内存/显存复用
}

void RunAnalysis(int batch_size, std::string model_dirname) {
  // 1. 创建AnalysisConfig
  AnalysisConfig config;
  CreateConfig(&config, model_dirname);
  
  // 2. 根据config 创建predictor，并准备输入数据，此处以全0数据为例
  auto predictor = CreatePaddlePredictor(config);
  int channels = 3; // 根据模型信息设置输入shape
  int height = 224;
  int width = 224;
  float input[batch_size * channels * height * width] = {0};
  
  // 3. 创建输入
  // 使用了ZeroCopy接口，可以避免推理中多余的CPU copy，提升推理性能
  auto input_names = predictor->GetInputNames();			// 获取输入名称	
  auto input_t = predictor->GetInputTensor(input_names[0]); // 获取输入Tensor
  input_t->Reshape({batch_size, channels, height, width}); 	// 必须对输入数据进行Reshape
  input_t->copy_from_cpu(input);							// 从CPU拷贝输入，准备进行推理

  // 4. 运行推理引擎
  CHECK(predictor->ZeroCopyRun());
   
  // 5. 获取输出
  std::vector<float> out_data;
  auto output_names = predictor->GetOutputNames();				// 获取输出名称
  auto output_t = predictor->GetOutputTensor(output_names[0]);	// 获取输出Tensor
  std::vector<int> output_shape = output_t->shape();
  int out_num = std::accumulate(output_shape.begin(), output_shape.end(), 1, std::multiplies<int>()); // 获取输出数目以打印输出或进行后续处理

  out_data.resize(out_num);
  output_t->copy_to_cpu(out_data.data()); // 将输出拷贝至CPU
}
}  // namespace paddle

int main() { 
  // 模型下载地址 http://paddle-inference-dist.cdn.bcebos.com/tensorrt_test/mobilenet.tar.gz
  paddle::RunAnalysis(1, "./mobilenet");
  return 0;
}
```

## C++ API使用示例
本节提供一个使用飞桨 C++ 预测库和 mobilenet v1 模型进行图像分类预测的代码示例，展示预测库使用的完整流程。

1. 下载或编译飞桨预测库，参考[源码编译](https://aistudio.baidu.com/aistudio/projectdetail/248512)。
2. 下载[预测样例](https://paddle-inference-dist.bj.bcebos.com/tensorrt_test/paddle_inference_sample_v1.7.tar.gz)并解压，进入`sample/inference`目录下。   

	`inference` 文件夹目录结构如下：

	``` shell
    inference
    ├── CMakeLists.txt
    ├── mobilenet_test.cc
    ├── thread_mobilenet_test.cc
    ├── mobilenetv1
    │   ├── model
    │   └── params
    ├── run.sh
    └── run_impl.sh
	```

	- `mobilenet_test.cc` 为单线程预测的C++源文件，与上文中使用AnalysisPredictor进行高性能预测中的示例代码对应S
	- `thread_mobilenet_test.cc` 为多线程预测的C++源文件  
	- `mobilenetv1` 为模型文件夹
	- `run.sh` 为预测运行脚本文件

3. 配置编译与运行脚本
	
    编译运行预测样例之前，需要根据运行环境配置编译与运行脚本`run.sh`。`run.sh`的选项与路径配置的部分如下：
	
    ``` shell
    # 设置是否开启MKL、GPU、TensorRT，如果要使用TensorRT，必须打开GPU
    WITH_MKL=ON
    WITH_GPU=OFF
    USE_TENSORRT=OFF

    # 按照运行环境设置预测库路径、CUDA库路径、CUDNN库路径、模型路径
    LIB_DIR=YOUR_LIB_DIR
    CUDA_LIB_DIR=YOUR_CUDA_LIB_DIR
    CUDNN_LIB_DIR=YOUR_CUDNN_LIB_DIR
    MODEL_DIR=YOUR_MODEL_DIR
    
    # 编译运行预测样例，若使用多线程预测，将 mobilenet_test 更换为 thread_mobilenet_test 即可
    sh run_impl.sh ${LIB_DIR} mobilenet_test ${MODEL_DIR} ${WITH_MKL} ${WITH_GPU} ${CUDNN_LIB_DIR} ${CUDA_LIB_DIR} ${USE_TENSORRT}
    ```
    
    按照实际运行环境配置`run.sh`中的选项开关和所需lib路径。
    
4. 编译与运行样例   

	``` shell
	sh run.sh
	```
	
	
	
## C++ API优化

在前面预测接口的介绍中，我们了解到，通过使用AnalysisConfig可以对AnalysisPredictor进行配置模型运行的信息。   
在本节中，我们会对AnalysisConfig中的优化配置进行详细的介绍。


### 优化原理

预测主要存在两个方面的优化，一是预测期间内存/显存的占用，二是预测花费的时间。  
* 预测期间内存/显存的占用决定了一个模型在机器上的并行的数量，如果一个任务是包含了多个模型串行执行的过程，内存/显存的占用也会决定任务是否能够正常执行（尤其对GPU的显存来说）。内存/显存优化增大了模型的并行量，提升了服务吞吐量，同时也保证了任务的正常执行，因此显的极为重要。
* 预测的一个重要的指标是模型预测的时间，通过对 kernel 的优化，以及加速库的使用，能够充份利用机器资源，使得预测任务高性能运行。


### 内存/显存优化
在预测初始化阶段，飞桨预测引擎会对模型中的 OP 输出 Tensor 进行依赖分析，将两两互不依赖的 Tensor 在内存/显存空间上进行复用。

可以通过调用以下接口方式打开内存/显存优化。

```c++
AnalysisConfig config;
config.EnableMemoryOptim();
```


**内存/显存优化效果**

| 模型 | 关闭优化 | 打开优化 |
| -------- | -------- | -------- |
| MobileNetv1（batch_size=128 ）     | 3915MB    | 1820MB  |
| ResNet50(batch_size=128)     |   6794MB   | 2204MB    |



### 性能优化

在模型预测期间，飞将预测引擎会对模型中进行一系列的 OP 融合，比如 Conv 和 BN 的融合，Conv 和 Bias、Relu 的融合等。OP 融合不仅能够减少模型的计算量，同时可以减少 Kernel Launch 的次数，从而能提升模型的性能。

可以通过调用以下接口方式打开 OP 融合优化：

```c++
AnalysisConfig config;
config.SwitchIrOptim(true);  // 默认打开
```

除了通用的 OP 融合优化外，飞桨预测引擎有针对性的对 CPU 以及 GPU 进行了性能优化。

#### CPU 性能优化

#### 1.对矩阵库设置多线程

模型在CPU预测期间，大量的运算依托于矩阵库，如 OpenBlas，MKL。
通过设置矩阵库内部多线程，能够充分利用 CPU 的计算资源，加速运算性能。

可以通过调用以下接口方式设置矩阵库内部多线程。

```
AnalysisConfig config;
// 通常情况下，矩阵内部多线程（num） * 外部线程数量 <= CPU核心数
config->SetCpuMathLibraryNumThreads(num);  
```

##### 2.使用 MKLDNN 加速

[MKLDNN](https://github.com/intel/mkl-dnn)是Intel发布的开源的深度学习软件包。目前飞桨预测引擎中已经有大量的OP使用MKLDNN加速，包括：Conv，Batch Norm，Activation，Elementwise，Mul，Transpose，Pool2d，Softmax 等。

可以通过调用以下接口方式打开MKLDNN优化。

```c++
AnalysisConfig config;
config.EnableMKLDNN();  
```

开关打开后，飞桨预测引擎会使用 MKLDNN 加速的 Kernel 运行，从而加速 CPU 的运行性能。


#### GPU 性能优化

##### 使用 TensorRT 子图性能优化

TensorRT 是 NVIDIA 发布的一个高性能的深度学习预测库，飞桨预测引擎采用子图的形式对 TensorRT 进行了集成。在预测初始化阶段，通过对模型分析，将模型中可以使用 TensorRT 运行的 OP 进行标注，同时把这些标记过的且互相连接的 OP 融合成子图并转换成一个 TRT OP 。在预测期间，如果遇到 TRT OP ，则调用 TensorRT 库对该 OP 进行优化加速。


可以通过调用以下接口的方式打开 TensorRT 子图性能优化：

```c++
config->EnableTensorRtEngine(1 << 20      /* workspace_size*/,   
                        batch_size        /* max_batch_size*/,     
                        3                 /* min_subgraph_size*/, 
                        AnalysisConfig::Precision::kFloat32 /* precision*/, 
                        false             /* use_static*/, 
                        false             /* use_calib_mode*/);
```

该接口中的参数的详细介绍如下：

- workspace_size，类型：int，默认值为1 << 20。指定TensorRT使用的工作空间大小，TensorRT会在该大小限制下筛选合适的kernel执行预测运算。
- max_batch_size，类型：int，默认值为1。需要提前设置最大的batch大小，运行时batch大小不得超过此限定值。
- min_subgraph_size，类型：int，默认值为3。Paddle-TRT 是以子图的形式运行，为了避免性能损失，当子图内部节点个数大于min_subgraph_size的时候，才会使用Paddle-TRT运行。
- precision，类型：enum class Precision {kFloat32 = 0, kHalf, kInt8,};, 默认值为AnalysisConfig::Precision::kFloat32。指定使用TRT的精度，支持FP32（kFloat32），FP16（kHalf），Int8（kInt8）。若需要使用Paddle-TRT int8离线量化校准，需设定precision为 - AnalysisConfig::Precision::kInt8, 且设置use_calib_mode 为true。
- use_static，类型：bool, 默认值为 false 。如果指定为 true ，在初次运行程序的时候会将 TRT 的优化信息进行序列化到磁盘上，下次运行时直接加载优化的序列化信息而不需要重新生成。
- use_calib_mode，类型：bool, 默认值为false。若要运行 Paddle-TRT int8 离线量化校准，需要将此选项设置为 true 。


目前 TensorRT 子图对图像模型有很好的支持，支持的模型如下
- 分类：Mobilenetv1/2, ResNet, NasNet, VGG, ResNext, Inception, DPN，ShuffleNet
- 检测：SSD，YOLOv3，FasterRCNN，RetinaNet
- 分割：ICNET，UNET，DeepLabV3，MaskRCNN



