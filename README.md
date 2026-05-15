# Emotion Recognition - 100428904
Y3P Project by Christian Iannotta

## Overview
This project implements a real-time facial emotion recognition system designed for use within video call environments. The system processes live webcam streams, extracts facial landmarks using MediaPipe, and classifies emotional states using a custom-built neural network implemented with NumPy.

The architecture consists of a React-based frontend implemented with a Daily.co video call session, a Python FastAPI backend for frame processing and inference, and a machine learning pipeline that operates on engineered facial geometry features rather than raw image data. Each video frame is captured from the webcam stream, decoded using OpenCV, and passed through a sequence of preprocessing steps including face validation, alignment, landmark normalisation, and feature extraction.

A lightweight multilayer perceptron (MLP) is used to classify each frame into one of the seven emotion categories listed below. Temporal smoothing is applied to stabilise predictions across consecutive frames, improving robustness in real-time conditions.

#### Emotion Classes
    0 Neutral
    1 Happy
    2 Sad
    3 Angry
    4 Surprise
    5 Fear
    6 Disgust

The system serves as a proof-of-concept for integrating traditional computer vision techniques with real-time communication platforms, demonstrating both the challenges and feasibility of emotion-aware video conferencing systems.

## Pipeline
To prepare for development, I broke the problem down into distinct modular stages, creating a pipeline as shown below:

                            -   receive frame
                            1)  detect face and create face mesh
                            2)  validation and quality checks
                                -   fallback to previous frame if needed
                            3)  find bounding box for face
                                -   crop raw frame to new ROI
                            4)  align and normalise
                            5)  feature extraction
                            6)  classification
                            -   output

## Model

- Input: 15 engineered facial geometry features
- Architecture: 2-layer MLP (64 → 32 → 7)
- Output: 7 emotion classes
- Implementation: NumPy-based forward propagation
- Temporal smoothing applied to outputs
- Accuracy: ~58–59% (imbalanced dataset limitations)


## Dependencies

### Backend
    - Python 3.11+
    - NumPy
    - OpenCV
    - MediaPipe
    - FastAPI
    - Uvicorn

### Frontend
    - React
    - Vite
    - @daily-co/daily-js


## Python Packages

Install backend dependencies using:

```bash
pip install -r requirements.txt
    
