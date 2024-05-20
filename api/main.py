from pydoc import render_doc
from fastapi import FastAPI, File, UploadFile
from matplotlib.pylab import beta
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = tf.saved_model.load("../models/1/")
beta_model = tf.saved_model.load("../models/2/")

CLASS_NAMES =   ['Early Blight', 'Late Blight', 'Healthy']


@app.get("/ping")
async def ping():
    return "Hello, I am alive!"

def read_file_as_image(file) -> np.ndarray:
    image = np.array(Image.open(BytesIO(file)))
    image = np.resize(image, (256, 256, 3))
    return image

@app.get("/predict", response_class=HTMLResponse)
async def get_predict_page():
    with open("static/home.html") as file:
        return HTMLResponse(content=file.read(), status_code=200)

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    img = read_file_as_image(await file.read())
    img = tf.convert_to_tensor(img)
    image_batch = tf.expand_dims(img, 0)
    image_batch = tf.cast(image_batch, tf.float32)
    predictions = MODEL(image_batch)
    print(predictions)
    print(CLASS_NAMES[np.argmax(predictions)])
    return {"class": CLASS_NAMES[np.argmax(predictions)]}

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=3000)
    # uvicor main:app --reload