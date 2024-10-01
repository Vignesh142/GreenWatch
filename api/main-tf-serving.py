import re
from fastapi import FastAPI, File, UploadFile
from matplotlib.pylab import beta
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import requests
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

endpoint = "http://localhost:8502/v1/models/potatoes_model:predict"

CLASS_NAMES =   ['Early Blight', 'Late Blight', 'Healthy']

@app.get("/ping")
async def ping():
    return "Hello, I am alive!"

def read_file_as_image(file) -> np.ndarray:
    image = np.array(Image.open(BytesIO(file)))
    image = np.resize(image, (256, 256, 3))
    return image

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    img = read_file_as_image(await file.read())
    image_batch = np.expand_dims(img, 0)
    json_data = {
        "instances": image_batch.tolist()
    }
    response = requests.post(endpoint, json=json_data)
    
    prediction = response.json()["predictions"][0]    
    confidence = np.max(prediction)
    
    return {
        "class": CLASS_NAMES[np.argmax(prediction)],
        "confidence": confidence
    }

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)