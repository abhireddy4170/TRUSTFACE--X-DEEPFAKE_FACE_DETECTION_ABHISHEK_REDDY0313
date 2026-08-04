# TRUSTFACE-X: Hybrid CNN-LSTM Intelligence for Deepfake Exposure

## Overview

TRUSTFACE-X is an AI-powered deepfake detection system designed to identify manipulated facial images and videos with high accuracy. The project leverages a hybrid Convolutional Neural Network (CNN) and Long Short-Term Memory (LSTM) architecture to capture both spatial and temporal features, enabling robust detection of AI-generated and tampered media.

As deepfake technology becomes more sophisticated, TRUSTFACE-X aims to provide an effective solution for digital media authentication and misinformation prevention.

---

## Features

- Detects real and deepfake facial images/videos
- Hybrid CNN-LSTM architecture for improved accuracy
- Face extraction and preprocessing using OpenCV
- Deep learning-based binary classification
- Performance evaluation using accuracy, precision, recall, and F1-score
- Easy-to-use and scalable architecture

---

## Tech Stack

- **Programming Language:** Python
- **Deep Learning Framework:** TensorFlow / Keras
- **Computer Vision:** OpenCV
- **Libraries:** NumPy, Pandas, Matplotlib, Scikit-learn
- **Development Environment:** Jupyter Notebook / VS Code
- **Version Control:** Git & GitHub

---

## Project Architecture

```
Input Image/Video
        │
        ▼
 Face Detection & Extraction
        │
        ▼
 Image Preprocessing
        │
        ▼
 CNN Feature Extraction
        │
        ▼
 LSTM Temporal Learning
        │
        ▼
 Binary Classification
 (Real / Deepfake)
        │
        ▼
 Prediction Result
```

---

## Project Structure

```
TRUSTFACE-X/
│
├── dataset/
│   ├── real/
│   └── fake/
│
├── models/
│
├── notebooks/
│
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   ├── predict.py
│   └── utils.py
│
├── results/
│
├── requirements.txt
│
├── README.md
│
└── LICENSE
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/abhireddy4170/TRUSTFACE--X-DEEPFAKE_FACE_DETECTION_ABHISHEK_REDDY0313
.git
```

Navigate to the project folder

```bash
cd TRUSTFACE-X
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

Train the model

```bash
python train.py
```

Run prediction

```bash
python predict.py
```

---

## Dataset

The project uses publicly available deepfake datasets containing both authentic and manipulated facial images/videos.

Example datasets:

- FaceForensics++
- Celeb-DF
- DeepFake Detection Challenge (DFDC)

---

## Model

The proposed model combines:

- **CNN** for extracting spatial features from facial images.
- **LSTM** for learning temporal relationships between video frames.
- **Binary Classifier** to classify media as Real or Deepfake.

---

## Performance

Evaluation Metrics:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

*(Replace with your actual results.)*

Example:

| Metric | Score |
|---------|-------|
| Accuracy | 96.4% |
| Precision | 95.8% |
| Recall | 96.1% |
| F1-Score | 95.9% |

---

## Applications

- Social Media Content Verification
- Digital Media Authentication
- Fake News Detection
- Cybersecurity
- Journalism
- Law Enforcement
- Identity Protection

---

## Future Enhancements

- Real-time webcam deepfake detection
- Mobile application integration
- Explainable AI (XAI) visualization
- Transformer-based architectures
- Multi-face detection support
- Cloud deployment

---

## Team

Developed as a B.Tech Major Project in Artificial Intelligence & Machine Learning.

---

## License

This project is developed for educational and research purposes.

---

## Acknowledgements

- TensorFlow
- OpenCV
- Keras
- Scikit-learn
- FaceForensics++
- DFDC Dataset
