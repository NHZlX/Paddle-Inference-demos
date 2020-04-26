
<h2 align="center">Paddle Inference简介</h2>

Paddle Inference为飞桨核心框架推理引擎。Paddle Inference功能特性丰富，性能优异，针对不同平台不同的应用场景进行了深度的适配优化,做到高吞吐、低时延，保证了飞桨模型在服务器端即训即用，快速部署。    

- 主流软硬件环境兼容适配   
	支持服务器端X86 CPU、NVIDIA GPU芯片，兼容Linux/macOS/Windows系统。
- 支持飞桨所有模型    
	支持所有飞桨训练产出的模型，真正即训即用。
- 多语言环境丰富接口可灵活调用   
	支持C++, Python, C, Go和R语言API, 接口简单灵活，20行代码即可完成部署。可通过Python API，实现对性能要求不太高的场景快速支持；通过C++高性能接口，可与线上系统联编；通过基础的C API可扩展支持更多语言的生产环境。


<h2 align="center">Paddle Inference 核心功能</h2>

- 内存/显存复用提升服务吞吐量  
	在推理初始化阶段，对模型中的OP输出Tensor 进行依赖分析，将两两互不依赖的Tensor在内存/显存空间上进行复用，进而增大计算并行量，提升服务吞吐量。

- 细粒度OP横向纵向融合减少计算量
	在推理初始化阶段，按照已有的融合模式将模型中的多个OP融合成一个OP，减少了模型的计算量的同时，也减少了 Kernel Launch的次数，从而能提升推理性能。目前Paddle Inference支持的融合模式多达几十个。

- 内置高性能的CPU/GPU Kernel
	内置同Intel、Nvidia共同打造的高性能kernel，保证了模型推理高性能的执行。

- 子图集成[TensorRT](https://developer.nvidia.com/tensorrt)加快GPU推理速度
	Paddle Inference采用子图的形式集成TensorRT，针对GPU推理场景，TensorRT可对一些子图进行优化，包括OP的横向和纵向融合，过滤冗余的OP，并为OP自动选择最优的kernel，加快推理速度。

- 集成MKLDNN 加速CPU推理性能
   
- 支持加载PaddleSlim量化压缩后的模型   
	[PaddleSlim](https://github.com/PaddlePaddle/PaddleSlim)是飞桨深度学习模型压缩工具，Paddle Inference可联动PaddleSlim，支持加载量化、裁剪和蒸馏后的模型并部署，由此减小模型存储空间、减少计算占用内存、加快模型推理速度。其中在模型量化方面，[Paddle Inference在X86 CPU上做了深度优化](https://github.com/PaddlePaddle/PaddleSlim/tree/80c9fab3f419880dd19ca6ea30e0f46a2fedf6b3/demo/mkldnn_quant/quant_aware)，常见分类模型的单线程性能可提升近3倍，ERNIE模型的单线程性能可提升2.68倍。
	
<h2 align="center">Paddle Inference使用样例</h2>

我们在这里提供了Paddle Inference 在python以及c++下的测试样例以及部署案例，包含内容如下：


- Python   
  该目录中包含了Paddle Inference的cpu，gpu，以及gpu下trt子图的使用样例。
  
- C++   
  该目录中包含了Paddle Inference的cpu，gpu，以及gpu下trt子图的使用样例。

- projects  
  该目录中包含了Paddle Inference部署案例，包括口罩识别部署等（此目录会持续更新）。
  
- models   
  该目录中包含了Paddle Inference测试样例需要用到的预测模型。
  

### 相关链接

- [Python 预测 API介绍](https://www.paddlepaddle.org.cn/documentation/docs/zh/advanced_guide/inference_deployment/inference/python_infer_cn.html)

- [C++预测API介绍](https://www.paddlepaddle.org.cn/documentation/docs/zh/advanced_guide/inference_deployment/inference/native_infer.html)

- [Paddle-TRT子图介绍](https://www.paddlepaddle.org.cn/documentation/docs/zh/advanced_guide/performance_improving/inference_improving/paddle_tensorrt_infer.html)

- [编译安装Linux C++预测库](https://www.paddlepaddle.org.cn/documentation/docs/zh/advanced_guide/inference_deployment/inference/build_and_install_lib_cn.html)

- [安装与编译 Windows 预测库](https://www.paddlepaddle.org.cn/documentation/docs/zh/advanced_guide/inference_deployment/inference/windows_cpp_inference.html)
	
