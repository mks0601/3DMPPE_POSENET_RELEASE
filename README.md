[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/camera-distance-aware-top-down-approach-for-1/3d-multi-person-pose-estimation-absolute-on)](https://paperswithcode.com/sota/3d-multi-person-pose-estimation-absolute-on?p=camera-distance-aware-top-down-approach-for-1)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/camera-distance-aware-top-down-approach-for-1/3d-multi-person-pose-estimation-root-relative)](https://paperswithcode.com/sota/3d-multi-person-pose-estimation-root-relative?p=camera-distance-aware-top-down-approach-for-1)

# PoseNet of "Camera Distance-aware Top-down Approach for 3D Multi-person Pose Estimation from a Single RGB Image"

<p align="center">
<img src="https://cv.snu.ac.kr/research/3DMPPE/figs/qualitative_intro.PNG" width="800" height="300">
</p>

<p align="middle">
<img src="https://cv.snu.ac.kr/research/3DMPPE/figs/video1.gif" width="400" height="300"> <img src="https://cv.snu.ac.kr/research/3DMPPE/figs/video2.gif" width="400" height="300">
</p>


## Introduction

This repo is official **[PyTorch](https://pytorch.org)** implementation of **[Camera Distance-aware Top-down Approach for 3D Multi-person Pose Estimation from a Single RGB Image (ICCV 2019)](https://arxiv.org/abs/1907.11346)**. It contains **PoseNet** part.

**What this repo provides:**
* [PyTorch](https://pytorch.org) implementation of [Camera Distance-aware Top-down Approach for 3D Multi-person Pose Estimation from a Single RGB Image (ICCV 2019)](https://arxiv.org/abs/1907.11346).
* Flexible and simple code.
* Compatibility for most of the publicly available 2D and 3D, single and multi-person pose estimation datasets including **[Human3.6M]([http://vision.imar.ro/human3.6m/description.php](http://vision.imar.ro/human3.6m/description.php)), [MPII](http://human-pose.mpi-inf.mpg.de/), [MS COCO 2017](http://cocodataset.org/#home), [MuCo-3DHP](http://gvv.mpi-inf.mpg.de/projects/SingleShotMultiPerson/) and [MuPoTS-3D](http://gvv.mpi-inf.mpg.de/projects/SingleShotMultiPerson/)**.
* Human pose estimation visualization code.

## Dependencies
* [PyTorch](https://pytorch.org)
* [CUDA](https://developer.nvidia.com/cuda-downloads)
* [cuDNN](https://developer.nvidia.com/cudnn)
* [Anaconda](https://www.anaconda.com/download/)
* [COCO API](https://github.com/cocodataset/cocoapi)

This code is tested under Ubuntu 16.04, CUDA 9.0, cuDNN 7.1 environment with two NVIDIA 1080Ti GPUs.

Python 3.6.5 version with Anaconda 3 is used for development.

## Directory

### Root
The `${POSE_ROOT}` is described as below.
```
${POSE_ROOT}
|-- data
|-- common
|-- main
|-- vis
`-- output
```
* `data` contains data loading codes and soft links to images and annotations directories.
* `common` contains kernel codes for 3d multi-person pose estimation system.
* `main` contains high-level codes for training or testing the network.
* `vis` contains scripts for 3d visualization.
* `output` contains log, trained models, visualized outputs, and test result.

### Data
You need to follow directory structure of the `data` as below.
```
${POSE_ROOT}
|-- data
|-- |-- Human36M
|   `-- |-- bbox_root
|       |   |-- bbox_root_human36m_output.json
|       |-- images
|       `-- annotations
|-- |-- MPII
|   `-- |-- images
|       `-- annotations
|-- |-- MSCOCO
|   `-- |-- bbox_root
|       |   |-- bbox_root_coco_output.json
|       |-- images
|       |   |-- train/
|       |   |-- val/
|       `-- annotations
|-- |-- MuCo
|   `-- |-- data
|       |   |-- augmented_set
|       |   |-- unaugmented_set
|       |   `-- MuCo-3DHP.json
`-- |-- MuPoTS
|   `-- |-- bbox_root
|       |   |-- bbox_mupots_output.json
|       |-- data
|       |   |-- MultiPersonTestSet
|       |   `-- MuPoTS-3D.json
```
* Download Human3.6M parsed data [[images](https://cv.snu.ac.kr/dataset/3DMPPE/Human36M/images.zip)][[annotations](https://cv.snu.ac.kr/dataset/3DMPPE/Human36M/annotations.zip)]
* Download MPII parsed data [[images](http://human-pose.mpi-inf.mpg.de/)][[annotations](https://cv.snu.ac.kr/dataset/3DMPPE/MPII/annotations.zip)]
* Download MuCo parsed and composited data [[images_1](https://cv.snu.ac.kr/dataset/3DMPPE/MuCo/augmented.zip)][[images_2](https://cv.snu.ac.kr/dataset/3DMPPE/MuCo/unaugmented.zip)][[annotations](https://cv.snu.ac.kr/dataset/3DMPPE/MuCo/MuCo-3DHP.json)]
* Download MuPoTS parsed parsed data [[images](http://gvv.mpi-inf.mpg.de/projects/SingleShotMultiPerson/)][[annotations](https://cv.snu.ac.kr/dataset/3DMPPE/MuPoTS/MuPoTS-3D.json)]
* All annotation files follow [MS COCO format](http://cocodataset.org/#format-data).
* If you want to add your own dataset, you have to convert it to [MS COCO format](http://cocodataset.org/#format-data).
### Output
You need to follow the directory structure of the `output` folder as below.
```
${POSE_ROOT}
|-- output
|-- |-- log
|-- |-- model_dump
|-- |-- result
`-- |-- vis
```
* Creating `output` folder as soft link form is recommended instead of folder form because it would take large storage capacity.
* `log` folder contains training log file.
* `model_dump` folder contains saved checkpoints for each epoch.
* `result` folder contains final estimation files generated in the testing stage.
* `vis` folder contains visualized results.

### 3D visualization
* Run `$DB_NAME_img_name.py` to get image file names in `.txt` format.
* Place your test result files (`preds_2d_kpt_$DB_NAME.mat`, `preds_3d_kpt_$DB_NAME.mat`) in `single` or `multi` folder.
* Run `draw_3Dpose_$DB_NAME.m`

## Running 3DMPPE_POSENET
### Start
* In the `main/config.py`, you can change settings of the model including dataset to use, network backbone, and input size and so on.

### Train
In the `main` folder, run
```bash
python train.py --gpu 0-1
```
to train the network on the GPU 0,1. 

If you want to continue experiment, run 
```bash
python train.py --gpu 0-1 --continue
```
`--gpu 0,1` can be used instead of `--gpu 0-1`.

### Test
Place trained model at the `output/model_dump/`.

In the `main` folder, run 
```bash
python test.py --gpu 0-1 --test_epoch 20
```
to test the network on the GPU 0,1 with 20th epoch trained model. `--gpu 0,1` can be used instead of `--gpu 0-1`.

## Results
Here I report the performance of the PoseNet. Also, I provide pre-trained models of the PoseNetNet. Bounding box and root locations are obtained from DetectNet and RootNet. 


#### Human3.6M dataset using protocol 1
For the evaluation, you can run `test.py` or there are evaluation codes in `Human36M`.
<p align="center">
<img src="https://cv.snu.ac.kr/research/3DMPPE/figs/H36M_P1.png">
</p>

* Bounding box [[H36M_protocol1](https://cv.snu.ac.kr/research/3DMPPE/result/bbox/Human36M/Protocol1/bbox_human36m_output.json)]
* Bounding box + 3D Human root coordinatees in camera space [[H36M_protocol1](https://cv.snu.ac.kr/research/3DMPPE/result/bbox_root/Human36M/Protocol1/bbox_root_human36m_output.json)]
* PoseNet model trained on Human3.6M protocol 1 + MPII [[model](https://cv.snu.ac.kr/research/3DMPPE/model/PoseNet/human3.6m/p1/snapshot_24.pth.tar
)]
#### Human3.6M dataset using protocol 2
For the evaluation, you can run `test.py` or there are evaluation codes in `Human36M`.
<p align="center">
<img src="https://cv.snu.ac.kr/research/3DMPPE/figs/H36M_P2.png">
</p>

* Bounding box [[H36M_protocol2](https://cv.snu.ac.kr/research/3DMPPE/result/bbox/Human36M/Protocol2/bbox_human36m_output.json)]
* Bounding box + 3D Human root coordinatees in camera space [[H36M_protocol2](https://cv.snu.ac.kr/research/3DMPPE/result/bbox_root/Human36M/Protocol2/bbox_root_human36m_output.json)]
* PoseNet model trained on Human3.6M protocol 2+ MPII [[model](https://cv.snu.ac.kr/research/3DMPPE/model/PoseNet/human3.6m/p2/snapshot_24.pth.tar
)]

#### MuPoTS-3D dataset
For the evaluation, run `test.py`.  After that, move `data/MuPoTS/mpii_mupots_multiperson_eval.m` in `data/MuPoTS/data`. Also, move the test result files (`preds_2d_kpt_mupots.mat` and `preds_3d_kpt_mupots.mat`) in `data/MuPoTS/data`. Then run `mpii_mupots_multiperson_eval.m` with your evaluation mode arguments.
<p align="center">
<img src="https://cv.snu.ac.kr/research/3DMPPE/figs/MuPoTS.png">
</p>

* Bounding box [[MuPoTS-3D](https://cv.snu.ac.kr/research/3DMPPE/result/bbox/MuPoTS-3D/bbox_mupots_output.json)]
* Bounding box + 3D Human root coordinatees in camera space [[MuPoTS-3D](https://cv.snu.ac.kr/research/3DMPPE/result/bbox_root/MuPoTS-3D/bbox_root_mupots_output.json)]
* PoseNet model trained on MuCO-3DHP + MSCOCO [[model](https://cv.snu.ac.kr/research/3DMPPE/model/PoseNet/muco/snapshot_24.pth.tar
)]

#### MSCOCO dataset

We additionally provide estimated 3D human root coordinates in on the MSCOCO dataset. The coordinates are in 3D camera coordinate system, and focal lengths are set to 1500mm for both x and y axis. You can change focal length and corresponding distance using equation 2 or equation in supplementarial material of my [paper](https://arxiv.org/abs/1907.11346)
* Bounding box + 3D Human root coordinates in camera space [[MSCOCO](https://cv.snu.ac.kr/research/3DMPPE/result/bbox_root/MSCOCO/bbox_root_coco_output.json)]

## Reference
  ```
@InProceedings{Moon_2019_ICCV_3DMPPE,
  author = {Moon, Gyeongsik and Chang, Juyong and Lee, Kyoung Mu},
  title = {Camera Distance-aware Top-down Approach for 3D Multi-person Pose Estimation from a Single RGB Image},
  booktitle = {The IEEE Conference on International Conference on Computer Vision (ICCV)},
  year = {2019}
}
```

