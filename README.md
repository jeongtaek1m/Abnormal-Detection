Deep Learning-based Abnormal Activity Recognition in Video
This repository contains the implementation of a deep learning framework designed for recognizing abnormal activities in videos. The model leverages optical flow and temporal feature learning to detect irregular events effectively.

Overview
The project integrates 3D Convolutional Neural Networks (3D CNN) and optical flow techniques to identify spatial-temporal patterns critical for detecting abnormal activities. It was evaluated using the mini-RWF2000 dataset, which includes scenarios labeled as "Fight" and "Non-Fight."

Key Features
Lucas-Kanade Optical Flow Algorithm: Implemented without OpenCV, relying only on NumPy for efficient motion vector calculation.
Farneback Optical Flow: Used for dense optical flow computation with multi-scale displacement estimation.
3D Convolutional Networks: For extracting spatial and temporal features from video inputs.
Fusion Model: Combines RGB data and optical flow features for robust abnormal activity detection.
Ablation Studies: To analyze the significance of optical flow in improving detection accuracy.
Model Architecture
Optical Flow
Lucas-Kanade Algorithm:

Detects sparse motion vectors using the Harris Corner Detector.
Calculates object movements for feature tracking.
Farneback Algorithm:

Provides dense motion vector computation.
Utilizes a multi-scale approach for estimating large displacements.
Abnormal Activity Detection
Fusion Model: Combines 3D convolutions on RGB and optical flow inputs, followed by feature multiplication.
MLP Classifier: Classifies scenes as "Normal" or "Abnormal."
Training
Dataset
Mini-RWF2000:
160 training videos and 40 validation videos.
Videos are pre-labeled as "Fight" and "Non-Fight."
Data augmentation includes color jittering and flipping.
Hardware
Trained on A100 GPUs using Google Colab.
Optimization
Optimizer: SGD with learning rate = 0.003, weight decay = 1e-6.
Scheduler: Cosine annealing for dynamic learning rates.
Loss Function: Cross-Entropy Loss.
Results
Optical Flow: Qualitative evaluation shows accurate motion tracking for sparse and dense features.
Abnormal Activity Detection:
Achieved ~75% accuracy on both training and validation sets using the base model.
ROC curves and ablation studies demonstrate the efficacy of integrating optical flow.
Future Work
Attention Mechanisms: Exploring cross-attention between optical flow and RGB channels for better spatial relationship learning.
New Datasets: Validated on the Smart-City CCTV Violence Detection Dataset (SCVD) for additional robustness testing.
References
RWF2000 Dataset
Vaswani, A., et al. "Attention is All You Need." NeurIPS, 2017.
Horn, B. K. P., and Schunck, B. G. "Polynomial Expansion Motion Estimation." AI, 1981.
Kumar, V., et al. "Towards Smart City Security." ICIP, 2020.
