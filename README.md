# Deep Learning-based Abnormal Activity Recognition in Video

## 📄 Overview
This repository presents a deep learning-based framework for recognizing abnormal activities in video data. By leveraging **optical flow** and **temporal feature learning**, our model effectively detects irregular events such as violent or non-violent scenes.

### Key Features
- **Fusion Model**: Combines 3D Convolutional Neural Networks (3D CNN) and optical flow for robust spatial-temporal pattern learning.
- **Optical Flow Implementation**: Includes custom implementations of **Lucas-Kanade** and **Farneback** algorithms for motion estimation.
- **Dataset Utilization**: Validated using the mini-RWF2000 dataset, with an ablation study highlighting the significance of optical flow integration.


---

## 🛠️ Model Architecture
The model integrates **optical flow** and **3D CNNs** to learn spatial and temporal features from video frames:
1. **Optical Flow**: Implements the Lucas-Kanade and Farneback algorithms for motion estimation.
2. **3D CNN**: Processes both **RGB channels** and **optical flow data** for time-sequential learning.
3. **Fusion Mechanism**: Combines outputs from RGB and optical flow channels for classification using a **Multilayer Perceptron (MLP)**.

---

## 🏋️‍♂️ Training Details
- **Dataset**: Mini-RWF2000 dataset with pre-labeled "Fight" and "Non-Fight" scenarios.
- **Hardware**: Trained on NVIDIA A100 GPU.
- **Optimizer**: SGD with learning rate `0.003` and weight decay `1e-6`.
- **Loss Function**: Cross Entropy Loss.
- **Learning Rate Scheduler**: Cosine Annealing.

### Data Augmentation
- **Color Jittering**: Applies random changes to brightness, contrast, and saturation.
- **Flipping**: Horizontal flipping for variability.

---

## 📊 Results
### Optical Flow
- Qualitative evaluation demonstrates effective motion tracking using Lucas-Kanade and Farneback algorithms.

### Abnormal Activity Detection
- Training accuracy: ~75%
- Validation accuracy: ~75%
- Ablation study shows improved performance with optical flow integration.

### ROC Curve
![ROC Curve](path/to/roc_curve_image.png)

---

## 🧪 Ablation Study
- **Without Optical Flow**: Higher training accuracy but reduced generalization.
- **With Optical Flow**: Improved validation accuracy and robust detection.

---

## 📁 Dataset
- **Mini-RWF2000**: Contains 200 videos (160 training, 40 validation) pre-labeled with "Fight" and "Non-Fight".
- **SCVD Dataset**: Used for inference and validation in a real-world setting.

---

## 🔧 Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/your-repo-name.git
   cd your-repo-name
