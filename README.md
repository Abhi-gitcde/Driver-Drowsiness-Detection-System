# 🛌 Driver Drowsiness Detection 🚘

A real-time driver drowsiness detection system using OpenCV, Haar cascades, and a deep learning model. It alerts drivers with a sound alarm if they appear drowsy, helping prevent accidents.

---

## 🔧 Features

- 🧠 Deep learning-based eye state detection
- 👁️ Real-time face & eye detection using OpenCV
- 🔊 Alarm triggered when eyes remain closed too long
- 🎥 Webcam-based live monitoring
- 📈 FPS counter for performance tracking

---

## 🖥️ Tech Stack

- Python
- OpenCV
- TensorFlow (Keras)
- Pygame
- Haar Cascades

---


---


**About eye_status_model.h5**

The file eye_status_model.h5 is a pre-trained deep learning model used to classify the state of the eyes (open or closed). It enhances the accuracy of drowsiness detection, especially when traditional methods are less reliable.
- Model Input: Cropped eye images (64x64 pixels)
- Output: Binary prediction — 0 for open eyes, 1 for closed eyes
- This file is not included in the repository due to size constraints.
- You can:
  
   -Use your own trained model (ensure the input shape and output match).
  
   -Skip it and the system will fall back to Haarcascade-based traditional detection.
  
For best results, it’s recommended to train this model on a dataset that includes both people with and without spectacles to improve generalization.


---


## 🚀 How to Run

1. **Clone this repo**

```bash
git clone https://github.com/Abhi-gitcde/driver-drowsiness-detection.git
cd driver-drowsiness-detection


