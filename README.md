# 🧠 AI-Based Chest X-ray Classification System

### Developed by Diwanshu Gangwar

---

## 🚀 Live Demo

https://chest-x-ray-classifier-by-diwanshu-gangwar.streamlit.app/

---

## 🎬 Demo Video  

[![Watch Demo](https://img.youtube.com/vi/2m3MVTG9uNQ/0.jpg)](https://www.youtube.com/watch?v=2m3MVTG9uNQ)

---

## 📌 Project Overview

This project presents a deep learning-based medical imaging system that classifies chest X-ray images into three categories:

* Normal
* Pneumonia
* Tuberculosis (TB)

The system uses transfer learning with state-of-the-art convolutional neural networks and is deployed as a Streamlit web application for real-time predictions.

---

## 📊 Dataset Details

* Total Images: 9298
* Training Set: 6543 images
* Validation Set: 1391 images
* Test Set: 1364 images
* Classes: 3 (Normal, Pneumonia, TB)

---
* Initial model: EfficientNet
* Final model: DenseNet121
* Transfer learning used for feature extraction
* Fine-tuning applied to improve performance
---

## 📈 Model Performance

### ROC-AUC Scores

* Normal: 0.988
* Pneumonia: 0.990
* TB: 0.998

### Precision-Recall

* TB PR-AUC: 0.992

### Key Observations

* Strong performance across all classes
* Very high accuracy in detecting TB
* Some confusion between Pneumonia and Normal cases

---

## ⚙️ System Performance

* Total Parameters: 7,040,579
* Trainable Parameters: 2,161,155
* Model Size: 47 MB
* Latency: ~403 ms per image
* Throughput: ~0.68 images/sec

---

## 💻 Features

* Upload chest X-ray images
* Real-time prediction
* Deep learning-based classification
* Deployed using Streamlit

---

## 🛠️ Tech Stack

* Language: Python
* Framework: TensorFlow
* Models: EfficientNet, DenseNet121
* Deployment: Streamlit
* Version Control: Git & GitHub

---

## 🔄 How It Works

1. User uploads a chest X-ray image
2. Image is preprocessed
3. Passed through trained DenseNet model
4. Model predicts class probabilities
5. Result is displayed

---

## ⚠️ Limitations

* Trained on a limited dataset
* Not clinically validated
* Possible misclassification in visually similar cases

---

## 🔮 Future Work

* Increase dataset size
* Improve generalization
* Add explainable AI (Grad-CAM)
* Optimize inference speed

---
---

## Support

If you find this project useful, consider giving it a star ⭐ on GitHub.
