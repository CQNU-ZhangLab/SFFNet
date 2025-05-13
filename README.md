<p align="center">
  <img src="figures/title.png" width=70%> <br>
</p>

<h2 align="center">
  SFFNet: Synergistic Feature Fusion Network With Dual-Domain Edge Enhancement for UAV Image Object Detection
</h2>

<p align="center">
    <a href="https://github.com/CQNU-ZhangLab/SFFNet/blob/main/LICENSE">
        <img alt="license" src="https://img.shields.io/badge/license-AGPL--3.0-blue">
    </a>
    <a href="https://github.com/CQNU-ZhangLab/SFFNet/pulls">
        <img alt="prs" src="https://img.shields.io/github/issues-pr/CQNU-ZhangLab/SFFNet?style=rounded&color=red&label=PRs">
    </a>
    <a href="https://github.com/CQNU-ZhangLab/SFFNet/issues">
        <img alt="issues" src="https://img.shields.io/github/issues/CQNU-ZhangLab/SFFNet?color=olive&style=rounded">
    </a>
    <a href="https://github.com/CQNU-ZhangLab/SFFNet">
        <img alt="stars" src="https://img.shields.io/github/stars/CQNU-ZhangLab/SFFNet">
    </a>
    <a href="mailto:2023210516058@stu.cqnu.edu.cn">
        <img alt="Contact Us" src="https://img.shields.io/badge/Contact-Email-yellow">
    </a>
</p>

<p align="center">
    SFFNet is a synergistic feature fusion network tailored for object detection in UAV images, addressing challenges like background noise and target scale imbalance. It combines multi-scale dynamic dual-domain coupling for effective edge extraction and a synergistic feature pyramid network to enhance geometric and semantic representations, achieving excellent performance across various scales. 
</p>

---

<div align="center">
  Wenfeng Zhang<sup>1</sup>,
  Jun Ni<sup>1</sup>,
  Yue Meng<sup>2</sup>,
  Xiaodong Pei<sup>2</sup>, 
  Wei Hu<sup>1</sup>,
  Qibing Qin<sup>3</sup>,
  Lei Huang<sup>4</sup>
</div>

  
<p align="center">
<i>
1. Chongqing Normal University &nbsp; 2. CETC Yizhihang (Chongqing) Technology Co., Ltd &nbsp; 3. Weifang University &nbsp; 4. Ocean University of China
</i>
</p>

<p align="center">
  📧 Corresponding author: <a href="mailto:wei.workstation@gmail.com">wei.workstation@gmail.com</a>
</p>



<p align="center">
<strong>If our work helps your project, please give us a ⭐!</strong>
</p>

<p align="center">
  <img src="figures/params.png" width=60%> <br>
  The relationship between the AP value and the number of parameters for different object detection algorithms on the VisDrone dataset. The points near the upper left corner of the figure indicate that the model can achieve higher accuracy while maintaining a lower number of parameters. Our algorithm (marked <strong>"Ours"</strong>) outperforms others by achieving a higher AP value with fewer parameters.
</p>

<details>
  <summary>
  <font size="+1">Abstract 📄</font>
  </summary>
Object detection in unmanned aerial vehicle (UAV) images remains a highly challenging task, primarily caused by the complexity of background noise and the imbalance of target scales. Traditional methods easily struggle to effectively separate objects from intricate backgrounds and fail to fully leverage the rich multi-scale information contained within images. To address these issues, we have developed a synergistic feature fusion network (SFFNet) with dual-domain edge enhancement specifically tailored for object detection in UAV images. Firstly, the multi-scale dynamic dual-domain coupling (MDDC) module is designed. This component introduces a dual-driven edge extraction architecture that operates in both the frequency and spatial domains, enabling effective decoupling of multi-scale object edges from background noise. Secondly, to further enhance the representation capability of the model's neck in terms of both geometric and semantic information, a synergistic feature pyramid network (SFPN) is proposed. SFPN leverages linear deformable convolutions to adaptively capture irregular object shapes and establishes long-range contextual associations around targets through the designed wide-area perception module (WPM). Moreover, to adapt to the various applications or resource-constrained scenarios, six detectors of different scales (N/S/M/B/L/X) are designed. Experiments on two challenging aerial datasets (VisDrone and UAVDT) demonstrate the outstanding performance of SFFNet-X, achieving 36.8 AP and 20.6 AP, respectively. The lightweight models (N/S) also maintain a balance between detection accuracy and parameter efficiency.
</details>


## Performance 🏆 🌟
- VisDrone
  
| Model | AP | AP<sub>50</sub> | AP<sub>75</sub> | AP<sub>s</sub> | AP<sub>m</sub> | AP<sub>l</sub> |#Params |FLOPs|
|:---------------|:---:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| SFFNet-N |    26.7   |  45.4   |     26.8     | 19.8 |  35.3 |   35.0   |   1.7M  | 7.2G   |
| SFFNet-S |    31.6   |  52.8   |     32.3     | 25.3 |  40.8 |   35.7   |   6.3M  | 24.0G  |
| SFFNet-M |    35.3   |  57.5   |     36.6     | 28.0 |  46.0 |   42.9   |   14.2M | 63.2G  |
| SFFNet-B |    35.9   |  58.2   |     37.3     | 28.6 |  46.6 |   46.2   |   19.7M | 105.6G |
| SFFNet-L |    36.1   |  58.5   |     37.8     | 28.9 |  46.6 |   48.7   |   24.6M | 131.0G |
| SFFNet-X |    36.8   |  59.3   |     38.5     | 29.6 |  47.4 |   45.4   |   38.5M | 203.7G |

- UAVDT
  
| Model                | AP | AP<sub>50</sub> | AP<sub>75</sub> | AP<sub>s</sub> | AP<sub>m</sub> | AP<sub>l</sub> |
|:----------------------|:---------------:|:----------------------------:|:----------------------------:|:----------------------------:|:----------------------------:|:----------------------------:|
| ClusDet  | 13.7            | 26.5                         | 12.5                         | 9.1                          | 25.1                         | 31.2                         |
| CenterNet | 13.2            | 26.7                         | 11.8                         | 7.8                          | 26.6                         | 13.9                         |
| DMNet    | 14.7            | 24.6                         | 16.3                         | 9.3                          | 26.2                         | **35.2**                     |
| GFL     | 16.9            | 29.5                         | 17.9                         | -                            | -                            | -                            |
| AMRNet   | 18.2            | 30.4                         | 19.8                         | 10.3                         | 31.3                         | 33.5                         |
| GLSAN    | 19.0            | 30.5                         | 21.7                         | -                            | -                            | -                            |
| CEASC    | 17.1            | 30.9                         | 17.8                         | -                            | -                            | -                            |
| PRDET   | 19.1            | 33.8                         | 19.8                         | -                            | -                            | -                            |
| SCLNet   | 20.0            | 33.1                         | **22.3**                     | -                            | -                            | -                            |
| YOLOv8-X  | 18.1            | 31.1                         | 18.8                         | 13.0                         | 28.4                         | 32.4                         |
| YOLOv10-X | 19.5            | 32.3                         | 21.0                         | 13.4                         | 28.9                         | 31.5                         |
| YOLO11-X | 19.0            | 33.0                         | 19.6                         | 13.0                         | 30.7                         | 32.2                         |
| **SFFNet-X**  | **20.6**        | **34.4**                     | 21.9                         | **14.3**                     | **31.5**                     | 33.3                         |


## Installation 🛠️
`conda` virtual environment is recommended. 
```
conda create -n SFFNet python=3.9
conda activate SFFNet
pip install -r requirements.txt
pip install -e .
```
## Data Preparation 📦
To train on the VisDrone and UAVDT datasets, you need to organize them in the YOLO format. Follow the steps below to prepare your dataset:
1. **Organize Images:**
    Structure your dataset directories as follows:
    ```shell
    dataset/
    ├── images/
    │   ├── train/
    │   │   ├── image1.jpg
    │   │   ├── image2.jpg
    │   │   └── ...
    │   ├── val/
    │   │   ├── image1.jpg
    │   │   ├── image2.jpg
    │   │   └── ...
    └── labels/
        ├── train/
        │   ├── image1.txt
        │   ├── image2.txt
        │   └── ...
        └── val/
            ├── image1.txt
            ├── image2.txt
            └── ...
    ```
    - **`images/train/`**: Contains all training images.
    - **`images/val/`**: Contains all validation images.
    - **`labels/train/`**: Contains all training labels.
    - **`labels/val/`**: Contains all validation labels.

2. **Update Configuration Files:**

    Modify your [VisDrone.yml](https://github.com/CQNU-ZhangLab/SFFNet/blob/main/ultralytics/cfg/datasets/VisDrone.yaml) or [UAVDT.yml](https://github.com/CQNU-ZhangLab/SFFNet/blob/main/ultralytics/cfg/datasets/UAVDT.yaml)

    ```yaml
    path: # dataset root dir
    train: # train images (relative to 'path')  
    val: # val images (relative to 'path')  
    test: # test images (optional)  
    
    # Your Dataset Classes
    names:
      0: pedestrian
      1: people
      2: bicycle
      3: car
      4: van
      5: truck
      6: tricycle
      7: awning-tricycle
      8: bus
      9: motor

    ```



## Quick validation ✅
```
python val.py
```
## Quick training 🏋️‍♂️
```
python train.py
```
## Acknowledgement 🙏

The code base is built with [YOLOv10](https://github.com/THU-MIG/yolov10) and [ultralytics](https://github.com/ultralytics/ultralytics).

Thanks for the great implementations! 

## Support 🌟

If our work helps your project, please give us a ⭐!
