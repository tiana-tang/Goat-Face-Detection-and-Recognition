
# Project Name

## Introduction

This project is based on MMDetection and focuses on goat face detection. We propose a new network architecture, **Context-Assisted Astrous Deformable Network (CAADNet)**, which effectively addresses the geometric transformations of goat faces and issues such as coat color variation. Additionally, we have customized a dataset and improved the accuracy and efficiency of goat face detection by training and evaluating on classic detection models in MMDetection, such as Faster R-CNN, RetinaNet, SSD, etc.

## Dependencies

- **Operating System**: Linux and macOS (Windows is theoretically supported)
- **Python 3.6+**
- **PyTorch 1.3+**
- **CUDA 9.2+** (Also supports CUDA 9.0 if installed from PyTorch source)
- **GCC 5+**
- **[MMCV](https://mmcv.readthedocs.io/en/latest/#installation)**

## Installation Process

### Starting from scratch

Assuming CUDA 10.1 has been successfully installed, here is a complete script for setting up MMDetection based on `conda`. You can refer to the next section for step-by-step installation instructions:

```shell
# Create and activate a new conda environment
conda create -n sheep_face python=3.7 -y
conda activate sheep_face

# Install PyTorch and torchvision
conda install pytorch torchvision -c pytorch

# Install openmim
pip install openmim

# Install MMCV
mim install mmcv-full

# Clone your code repository 
git clone https://github.com/tiana-tang/Goat-Face-Detection-and-Recognition.git

# Enter the  directory
cd code

# Install dependencies
pip install -r requirements/build.txt
pip install -v -e .
```

If additional dependencies are needed, refer to the [MMDetection installation guide](https://mmdetection.readthedocs.io/en/stable/get_started.html) for more details.

## Dataset Preparation

This project uses a custom dataset called `nwafu_sheep_face`, stored in the `data/nwafu_sheep_face` folder. The dataset includes images of goat faces and corresponding annotation information.

## Configuration Files

All configuration files are located in the `configs/nwafu_sheep_face` directory. These configuration files include training setups for various models (such as Faster R-CNN, RetinaNet, SSD, etc.). You can choose different models based on your requirements. Each configuration file inherits from the basic configuration files in MMDetection and is modified as needed.

## Model Training

To train the model, simply run the following command:

```shell
python tools/train.py configs/nwafu_sheep_face/faster_rcnn_r50_fpn_sheep_face.py
```

This command will start the training process and output logs to the `work_dirs` directory.

## Testing and Inference

After training, you can run the following command to test the model and perform inference:

```shell
python tools/test.py configs/nwafu_sheep_face/faster_rcnn_r50_fpn_sheep_face.py work_dirs/faster_rcnn_r50_fpn_sheep_face/latest.pth --eval bbox
```

## Citation

Please cite this project as follows:

```
@article{Addressing2025,
  title={Addressing Facial Geometric Transformations in Goat Face Detection and Recognition},
  author={Gaoge Han, Lianyue Zhang, Zihan Bai, Xue Zhang, Ruizi Han, Chao Tang, Lianyue Zhang, and Jinglei Tang},
  journal={The Visual Computer},
  year={2025},
  volume={XX},
  pages={XX-XX},
}
```

