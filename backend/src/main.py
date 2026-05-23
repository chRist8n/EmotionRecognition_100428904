from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    #print("'/predict' endpoint triggered.")

    contents = await file.read()

    # decode image from frontend
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # TODO: pipeline here
    # landmarks → features → model

    emotion = "happy"  # placeholder
    confidence = 0.82

    return {
        "emotion": emotion,
        "confidence": float(confidence)
    }