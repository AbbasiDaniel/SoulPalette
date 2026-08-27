# 🎨 SoulPalette

> **A web application project that detects your inner and hidden emotions, going beyond just a simple label.**

[![Live Demo](https://img.shields.io/badge/Live_Demo-Try_It_Now-brightgreen?style=for-the-badge&logo=render)](https://soulpalette.onrender.com)

🌐 **Live Website:** [soulpalette.onrender.com](https://soulpalette.onrender.com)

---
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat&logo=opencv&logoColor=white)](https://opencv.org/)
[![Flask](https://img.shields.io/badge/Flask-000000?style=flat&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Groq API](https://img.shields.io/badge/LLM-Llama%203.3-f34f29)](https://groq.com/)
[![Render](https://img.shields.io/badge/Deployed%20on-Render-46E3B7?style=flat&logo=render&logoColor=white)](https://render.com)

---

## Overview

SoulPalette illustrates your emotions as colors that contain parameters such as intensity, wavelength, and stability. Additionally, it features the demonstration of your emotions through charts and numbers. Finally, it provides users with a brief elaboration of their emotional state and personalized advice suggested to them based on the calculation.

---

## Architecture & Pipeline

**1. Video Capture & Face Detection:**  
Initially, the frontend records a solid 60-second video from the user once the button is pressed. The recorded video is compiled into 60 frames (one frame per second). Then, by using OpenCV functions, the video's pixels are extracted and stored within an array. However, the emotion detection system input is designed specifically for human face frames. Therefore, I used the pre-trained face detection functions of the OpenCV library (`cv2`) to extract the human face from each image.

**2. Deep Learning Classification:**  
Next comes the emotion detection system. I utilized the classic FER-2013 dataset, which contains thousands of human face images with their corresponding emotion labels. I used PyTorch to design a machine learning pattern recognizer. Moreover, in order to consider small spatial details (such as subtle facial movements and eye contours), I applied a Convolutional Neural Network (CNN) architecture. This model provided an emotion label for each frame of the user's video.

**3. Mathematical Signal Processing:**  
However, as mentioned, the scope of this project is multi-dimensional. Hence, I implemented four mathematical signal functions to analyze the emotion data stream over time:
* **Intensity:** Converts emotion tags into numeric values and calculates the overall average emotion score using `np.mean`.
* **Stability:** Measures how consistent the emotion remains over time by taking the inverse standard deviation (`1.0 - np.std`).
* **Wavelength:** Applies Fast Fourier Transform (`scipy.fft`) to extract the dominant oscillation frequency and derive the emotional wavelength.
* **Burstiness:** Uses `scipy.signal.find_peaks` to count sudden emotional intensity spikes that exceed a calculated threshold.

**4. LLM Integration:**  
At the very end, I built an extra feature that connects to Groq AI (using its API). It converts these calculated signal elements into a structured prompt, using Groq as an AI psychological assistant to deliver personalized advice.

---

## Tech Stack

* **Deep Learning & Computer Vision:** PyTorch, Torchvision, OpenCV (`cv2`)
* **Signal Processing & Data Analysis:** NumPy, SciPy
* **Backend & Web Framework:** Python 3.x, Flask
* **LLM & API Integration:** Groq API (Llama 3.3 via OpenAI SDK)
* **Environment & Deployment:** Render, Python-Dotenv

---

## Technical Challenges & Optimizations

One of the major problems I faced during implementation was server RAM limitations. In an online deployment scenario (such as on Render), the server does not provide sufficient RAM for heavy PyTorch operations, which led to several key changes in my system architecture:

* **Pre-trained Inference:** I trained the PyTorch model on my local system and saved the weights and biases into a separate state file, bypassing any server-side training. As a result, the cloud-based emotion detection is reduced strictly to a lightweight forward pass (inference mode).
* **Lightweight Dependencies:** To ensure Render could handle the application within memory limits, I utilized headless and optimized packages (such as `opencv-python-headless`), which reduced resource consumption without sacrificing core functionality.
* **Emotion Encoding:** To make emotion lists mathematically manageable for signal processing, I mapped the 7 primary emotions into discrete numerical values ranging from 1 to 7.
* **Frame Downsampling & Optimization:** Instead of processing full-rate video streams at 1,800 frames (30 FPS × 60 seconds), I filtered and downsampled the video to 60 keyframes (1 FPS), significantly reducing memory footprint and processing latency.

> **Note:** Some of these optimizations are specifically applied to the cloud demo version to suit constrained free-tier servers, whereas the local version is designed to utilize full system RAM for higher resolution input.

---

## Installation & Local Setup

Follow these steps to run **SoulPalette** locally on your machine:

### 1. Prerequisites
* Python 3.10 or higher
* Git

### 2. Clone the Repository
```bash
git clone https://github.com/your-username/SoulPalette.git
cd SoulPalette
```

### 3. Create & Activate Virtual Environment
```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Configure Environment Variables
Create a `.env` file in the root directory and add your secret keys:
```env
GROQ_API_KEY=your_groq_api_key_here
SECRET_KEY=your_flask_secret_key
```

### 6. Run the Application
This repository provides two different entry points depending on your hardware limits and testing goals:

* **To run the full-scale Local Version (Requires sufficient RAM):**
  ```bash
  python Local_Version.py
  ```

* **To run the optimized Demo Version (Used for cloud deployment):**
  ```bash
  python demo.py
  ```

Open your browser and navigate to `[http://127.0.0.1:5000](http://127.0.0.1:5000)`.
