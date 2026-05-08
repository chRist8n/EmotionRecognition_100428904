from fastapi import FastAPI, File, UploadFile
import numpy as np
import cv2

app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()

    # decode image from frontend
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # TODO: your pipeline here
    # landmarks → features → model

    emotion = "happy"  # placeholder
    confidence = 0.82

    return {
        "emotion": emotion,
        "confidence": float(confidence)
    }