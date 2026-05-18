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
To ensure modular architecture, I broke the problem down into distinct modular stages, creating a pipeline as shown below:

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

## Dataset

This project uses the RAF-DB (Real-world Affective Faces Database) for training and evaluation of the emotion classification model.

The dataset consists of facial images labelled across seven emotion categories:
Neutral, Happy, Sad, Angry, Surprise, Fear, and Disgust.

### Dataset Details
- Source: [https://www.kaggle.com/datasets/shuvoalok/raf-db-dataset?select=train_labels.csv]

### Data Usage
The dataset is not included in this repository due to size constraints.

To reproduce the system, download the dataset and place it in the following directory and format:

```bash
backend/data/raw/RAF-DB/DATASET/
    ├── train/
    ├── test/
```

### Preprocessing
Each image is processed using MediaPipe Face Mesh to extract 468 facial landmarks. These landmarks are then used to compute a fixed-length vector of geometric features, which serve as input to the classifier.

Some images may fail landmark detection due to:
- occlusion
- extreme head pose
- low image quality

Such samples are excluded during preprocessing, resulting in a reduced effective dataset size.

### Notes
Feature extraction is consistent between training and real-time inference to ensure identical input representations across both stages of the system.


## Model

- Input: 15 engineered facial geometry features
- Architecture: 2-layer MLP (64 → 32 → 7)
- Output: 7 emotion classes
- Implementation: NumPy-based forward propagation
- Temporal smoothing applied to outputs
- Accuracy: ~58–59% (imbalanced dataset limitations)


## Python Packages

Install backend dependencies using:

```bash
pip install -r requirements.txt
```

## How to run

### Backend

The backend is responsible for frame processing, facial landmark detection, feature extraction, and emotion classification.

---

#### Setup

Before running the application, all dependencies must be installed:

```bash
cd backend
pip install -r requirements.txt
```

---

#### Running the Backend Server

To start the FastAPI backend server:

```bash
uvicorn main:app --reload
```

This enables communication between the frontend and backend (although frontend frame capture is not fully implemented in the current version of the system).

---

#### Independent Backend Testing

The backend can also be run independently using a local webcam-based testing pipeline.

Before running tests, the dataset must be preprocessed and normalised. This is handled using:

```bash
python dataset_processing.py
```

This step prepares the feature representations used for model training and evaluation.

---

To test the trained model using a live webcam feed:

```bash
python webcam_debug.py
```

This script:
- loads/trains the MLP model
- captures live webcam frames
- extracts facial landmarks and features
- outputs real-time emotion predictions in an OpenCV window


### Frontend

The frontend is implemented using React and provides a video conferencing interface using the Daily.co SDK.

Its primary purpose is to simulate a real-time video call environment in which emotion recognition could be integrated.

---

#### Setup

Before running the application, dependencies must be installed:

```bash
cd frontend
npm install
```

---

#### Running the Frontend

To start the development server:

```bash
npm run dev
```

---

#### Functionality

The frontend currently provides:
- Real-time video call interface using Daily.co
- Multi-user video streaming support
- UI for displaying participants within a call session

---

#### Current Limitations

In the current implementation, the frontend does not transmit video frames to the backend for processing. As a result, emotion recognition is not yet integrated into the live video call interface.

Instead, backend processing is tested independently using a local webcam-based pipeline.


### System Note

The system was designed as a full client-server architecture for real-time emotion recognition in video calls. In this design, the frontend captures video frames from a live Daily.co call and transmits them to a FastAPI backend for processing and classification.

In the current implementation, full frame transmission between the frontend and backend has not been completed. As a result, the system operates in two separate parts:

- The backend is fully functional and processes webcam input independently for real-time emotion recognition.
- The frontend provides a working video call interface but is not yet connected to the emotion recognition pipeline.

Despite this, the core machine learning pipeline (landmark extraction, feature engineering, and MLP classification) has been fully implemented and validated within the backend environment.
    
