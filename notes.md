# 1 语义分割的概念

计算机视觉中关于图像识别有四大类任务

+ 分类-Classification：解决 **是什么** 的问题，即给定一张图片或一段视频判断里面包含什么类别的目标
+ 定位-Location：解决 **在哪里** 的问题，即定位出这个目标的位置
+ 检测-Detection：同时解决 **在哪里** 和 **是什么** 两个问题，既要定位出目标位置，又要知道目标类别
+ 分割-Segmentation：分为实例分割 (Instance-level) 和场景分割 (Scene-level) ，解决 **每一个像素属于哪个目标物或场景** 的问题

语义分割与目标检测的不同点在于，目标检测只需找到图中目标，打上框然后分出类别。语义分割是以描边的形式，将整张图不留缝隙的分割成每个区域，每个区域是一个类别，没有类别的默认为背景，语义分割不仅需要标识图像中存在的对象，还需要为每一个像素赋予一个语义标签。可以帮助计算机实现对图像中对象的更精细和准确的理解，以及对视觉信息进行更好的利用，为各种领域提供更多智能化的应用。

# 2 OpenMMLab

OpenMMLab是一个基于PyTorch的开源的CV算法库，目前OpenMMLab已经陆续开源30多个视觉算法库，实现了300多种算法，并包含2000+预训练模型，涵盖 **2D/3D目标检测、语义分割、视频理解、姿态分析** 等多个方向，其特点如下：

+ **模块化组合设计** 将网络框架分解为不同组件，将数据集构建、模型搭建、训练过程设计等过程封装为模块，在统一而灵活的架构上，用户能够轻松组合调用不同的模块，构建自定义计算机视觉网络框架
+ **高性能** 基于底层库MMCV，OpenMMLab中几乎所有基本运算操作都在GPU上运行，训练速度快
+ **SOTA方法** 开源框架中集成计算机视觉各个领域最新的先进算法，并且不断更新，使用者能够轻松使用新方法并进行改进。OpenMMLab系列项目的核心组件是 **MMCV**，它是用于计算机视觉研究的基础Python库，支持OpenMMLab旗下其他开源库，是上述一系列上层框架的基础支持库，提供底层通用组件，灵活性强，可扩展性好。

# 3 MMSegmentation

MMSegmentation 是 OpenMMLab project 的一部分，是基于PyTorch实现的功能强大的语义分割工具箱，其主要由七部分组成

+ **apis** 为模型推理提供的高级 API
+ **structures** 分割数据结构 `SegDataSample`
+ **datasets** 支持语义分割的多种数据集
  + **transforms** 中包括了许多数据增强变换
+ **models** 囊括了一个分割器的不同组件
  + **segmentors** 定义了所有的分割模型类
  + **data_preprocessors** 对输入模型的数据进行预处理
  + **backbones** 包含了能将图像转变为 feature map 的不同的 backbone 网络
  + **necks** 包含多种 neck 组件，将 backbone 部分和 head 部分连接起来
  + **decode_heads** 包含多种 heads 组件，将 feature map作为输入，输出分割结果
  + **losses** 包含多种损失函数
+ **engine** 是用于拓展 MMEngine 功能的运行时组件的一部分
  + **optimizers** 提供了 optimizers 和 optimizers wrappers
  + **hooks** 为 runner 提供多种 hook
+ **evaluation** 为模型性能评估提供不同标准
+ **visualization** 对分割结果进行可视化
