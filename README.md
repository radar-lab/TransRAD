# TransRAD: Retentive Vision Transformer for Enhanced Radar Object Detection

Published in **IEEE Transactions on Radar Systems:** https://ieeexplore.ieee.org/abstract/document/10869508

**ArXiv:** https://arxiv.org/abs/2501.17977
## Abstract
Despite significant advancements in environment perception capabilities for autonomous driving and intelligent robotics, cameras and LiDARs remain notoriously unreliable in low-light conditions and adverse weather, which limits their effectiveness. Radar serves as a reliable and low-cost sensor that can effectively complement these limitations. However, radar-based object detection has been underexplored due to the inherent weaknesses of radar data, such as low resolution, high noise, and lack of visual information.
In this paper, we present TransRAD, a novel 3D radar object detection model designed to address these challenges by leveraging the Retentive Vision Transformer (RMT) to more effectively learn features from information-dense radar Range-Azimuth-Doppler (RAD) data. Our approach leverages the Retentive Manhattan Self-Attention (MaSA) mechanism provided by RMT to incorporate explicit spatial priors, thereby enabling more accurate alignment with the spatial saliency characteristics of radar targets in RAD data and achieving precise 3D radar detection across Range-Azimuth-Doppler dimensions. Furthermore, we propose Location-Aware NMS to effectively mitigate the common issue of duplicate bounding boxes in deep radar object detection.
The experimental results demonstrate that TransRAD outperforms state-of-the-art methods in both 2D and 3D radar detection tasks, achieving higher accuracy, faster inference speed, and reduced computational complexity.
<center>
  <img src="https://github.com/radar-lab/TransRAD/blob/main/Figures/Fig.2.png" width=70% />
</center>

## Results

<center>
  <img src="https://github.com/radar-lab/TransRAD/blob/main/Figures/Fig.5.png" width=70% />
</center>

<center>
  <img src="https://github.com/radar-lab/TransRAD/blob/main/Figures/Fig.6.png" width=70% />
</center>

<center>
  <img src="https://github.com/radar-lab/TransRAD/blob/main/Figures/performence.png" width=75% />
</center>

## Train and Test
```
To be completed
```

## Acknowledgment

We sincerely acknowledge and appreciate the contributions of the following repositories, which have provided valuable references for our work:

- [RMT by qhfan](https://github.com/qhfan/RMT)
- [RADDet by ZhangAoCanada](https://github.com/ZhangAoCanada/RADDet)
- [YOLOv8-PyTorch by bubbliiiing](https://github.com/bubbliiiing/yolov8-pytorch)

Our implementation has been inspired by these works, and we extend our gratitude to the authors for their open-source contributions.
