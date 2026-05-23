from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import cv2
import numpy as np

app = FastAPI()

# store active connections per participant
connections = {}

@app.websocket("/ws/{participant_id}")
async def websocket_endpoint(websocket: WebSocket, participant_id: str):
    await websocket.accept()
    connections[participant_id] = websocket

    try:
        while True:
            # receive raw image bytes
            data = await websocket.receive_bytes()

            np_arr = np.frombuffer(data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            # ---- PIPELINE HERE ----
            emotion = "happy"
            confidence = 0.82
            # -----------------------

            await websocket.send_json({
                "participant_id": participant_id,
                "emotion": emotion,
                "confidence": float(confidence)
            })

    except WebSocketDisconnect:
        del connections[participant_id]