
# 项目名称

## 简介

本项目基于 MMDetection，旨在进行奶山羊面部检测。我们提出了一种全新的网络架构——上下文辅助星形可变形网络（CAADNet）该架构能够有效应对奶山羊面部几何变换和毛色变化等问题，另外我们自定义了数据集，通过在 MMDetection 中的经典检测模型（如 Faster R-CNN、RetinaNet、SSD 等）进行训练和评估的基础上，显著提高了奶山羊面部检测的准确性和效率。

## 依赖

- **操作系统**：Linux 和 macOS （Windows 理论上支持）
- **Python 3.6+**
- **PyTorch 1.3+**
- **CUDA 9.2+** （如果基于 PyTorch 源码安装，也能够支持 CUDA 9.0）
- **GCC 5+**
- **[MMCV](https://mmcv.readthedocs.io/en/latest/#installation)**

## 安装流程

### 从零开始设置脚本

假设当前已经成功安装 CUDA 10.1，这里提供了一个完整的基于 conda 安装 MMDetection 的脚本。您可以参考下一节中的分步安装说明。


```shell
# 创建并激活一个新的 conda 环境
conda create -n sheep_face python=3.7 -y
conda activate sheep_face

# 安装 PyTorch 和 torchvision
conda install pytorch torchvision -c pytorch


# 安装 openmim
pip install openmim

# 安装 MMCV
mim install mmcv-full


# 克隆的代码库
git clone https://github.com/tiana-tang/Goat-Face-Detection-and-Recognition.git

# 进入 目录
cd Goat-Face-Detection-and-Recognition

# 安装依赖
pip install -r requirements/build.txt
pip install -v -e .

```
如果需要额外的依赖，参考 [MMDetection 安装说明](https://mmdetection.readthedocs.io/en/stable/get_started.html) 获取更多细节。

## 数据集准备

本项目使用了自定义的数据集 `nwafu_sheep_face`，并将数据放置在 `data/nwafu_sheep_face` 文件夹中。数据集包含了奶山羊面部的图片和标注信息。

## 配置文件

所有的配置文件都放置在 `configs/nwafu_sheep_face` 目录下，配置文件包括不同的模型（如 Faster R-CNN、RetinaNet、SSD 等）的训练设置。你可以根据需求选择不同的模型进行训练。每个配置文件继承自 `mmdetection` 中的基础配置文件，并根据需要进行修改。

## 训练模型

训练模型时，只需运行以下命令：

```shell
python tools/train.py configs/nwafu_sheep_face/faster_rcnn_r50_fpn_sheep_face.py
```

此命令将启动训练过程，并将日志输出到 `work_dirs` 目录下。

## 测试与推理

训练完毕后，你可以运行以下命令来测试模型并进行推理：

```shell
python tools/test.py configs/nwafu_sheep_face/faster_rcnn_r50_fpn_sheep_face.py work_dirs/faster_rcnn_r50_fpn_sheep_face/latest.pth --eval bbox
```

## 引用

请根据以下格式引用本项目：

```
@article{Addressing2025,
  title={Addressing Facial Geometric Transformations in Goat Face
Detection and Recognition},
  author={Gaoge Han,  Lianyue Zhang, Zihan Bai, Xue Zhang, Ruizi Han, Chao Tang, Lianyue Zhang and
Jinglei Tang},
  journal={The Visual Computer},
  year={2025},
  volume={XX},
  pages={XX-XX},
}
```

