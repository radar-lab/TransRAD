
<h1 align="center">📡 TransRAD</h1>
<p align="center">
  <b>IEEE Transactions on Radar Systems Paper:</b><br>
  <i>TransRAD: Retentive Vision Transformer for Enhanced Radar Object Detection</i>
</p>

<p align="center">
  <img src="https://img.shields.io/github/stars/radar-lab/TransRAD?style=social" alt="GitHub Repo stars"/>
  <img src="https://img.shields.io/github/forks/radar-lab/TransRAD?style=social" alt="GitHub forks"/>
  <img src="https://img.shields.io/github/last-commit/radar-lab/TransRAD" alt="GitHub last commit"/>
  <a href="https://ieeexplore.ieee.org/abstract/document/10869508">
    <img src="https://img.shields.io/badge/IEEE-Published-blue.svg" alt="IEEE"/>
  </a>
  <a href="https://arxiv.org/abs/2501.17977">
    <img src="https://img.shields.io/badge/arXiv-2501.17977-b31b1b.svg" alt="arXiv"/>
  </a>
</p>

---

## 🧑‍🤝‍🧑 Contributors

<a href="https://github.com/radar-lab/TransRAD/graphs/contributors">
  <img alt="Contributors" src="https://contrib.rocks/image?repo=radar-lab/TransRAD" />
</a>



## 📧 Contact
- 🧑‍💻 **Author**: [Lei Cheng](https://github.com/leicheng5)  
- 🏫 **Lab**   : [Radar-Lab](https://github.com/radar-lab)

---

## 🎯 I. Abstract
Despite significant advancements in environment perception capabilities for autonomous driving and intelligent robotics, cameras and LiDARs remain notoriously unreliable in low-light conditions and adverse weather, which limits their effectiveness. Radar serves as a reliable and low-cost sensor that can effectively complement these limitations. However, radar-based object detection has been underexplored due to the inherent weaknesses of radar data, such as low resolution, high noise, and lack of visual information.
In this paper, we present TransRAD, a novel 3D radar object detection model designed to address these challenges by leveraging the Retentive Vision Transformer (RMT) to more effectively learn features from information-dense radar Range-Azimuth-Doppler (RAD) data. Our approach leverages the Retentive Manhattan Self-Attention (MaSA) mechanism provided by RMT to incorporate explicit spatial priors, thereby enabling more accurate alignment with the spatial saliency characteristics of radar targets in RAD data and achieving precise 3D radar detection across Range-Azimuth-Doppler dimensions. Furthermore, we propose Location-Aware NMS to effectively mitigate the common issue of duplicate bounding boxes in deep radar object detection.
The experimental results demonstrate that TransRAD outperforms state-of-the-art methods in both 2D and 3D radar detection tasks, achieving higher accuracy, faster inference speed, and reduced computational complexity.
<p align="center">
  <img src="https://github.com/radar-lab/TransRAD/blob/main/Figures/Fig.2.png" width="90%">
</p>



## 📊 II. Results

<center>
  <img src="https://github.com/radar-lab/TransRAD/blob/main/Figures/Fig.5.png" width=70% />
</center>

<center>
  <img src="https://github.com/radar-lab/TransRAD/blob/main/Figures/Fig.6.png" width=70% />
</center>

<center>
  <img src="https://github.com/radar-lab/TransRAD/blob/main/Figures/performence.png" width=75% />
</center>


## 🚀 III. Train and Test

---

### 📂 1. Dataset Preparation
You need to use the **RADDet radar dataset** for training.  

👉 Please refer to [RADDet by ZhangAoCanada](https://github.com/ZhangAoCanada/RADDet) for preparing the dataset.

---

### 🏋️ 2. Train
Run the following script to train the **TransRAD** model:  
```bash
python Train.py
````

💡 You may change the configuration per your needs.

---

### ✅ 3. Test

After training your model, run:

```bash
python Test.py
```

to get the testing results.

---

## 🙏 IV. Acknowledgment

We sincerely acknowledge and appreciate the contributions of the following repositories, which have provided valuable references for our work:

- [RMT by qhfan](https://github.com/qhfan/RMT)
- [RADDet by ZhangAoCanada](https://github.com/ZhangAoCanada/RADDet)
- [YOLOv8-PyTorch by bubbliiiing](https://github.com/bubbliiiing/yolov8-pytorch)

Our implementation has been inspired by these works, and we extend our gratitude to the authors for their open-source contributions.
