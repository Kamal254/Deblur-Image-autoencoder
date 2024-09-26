from fastapi import FastAPI, File,  UploadFile
import requests
from fastapi.middleware.cors import CORSMiddleware
from flask import Flask, redirect, url_for, request, render_template

from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from fastapi.middleware.cors import CORSMiddleware
# import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import InputLayer
from fastapi.responses import StreamingResponse

import mlflow
from mlflow import MlflowClient
import os


app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory='templates')

# Load Champion Model from the mlflow Model Registry
os.environ['MLFLOW_TRACKING_USERNAME'] = 'Kamal254'
os.environ['MLFLOW_TRACKING_PASSWORD'] = '3cbdb442b5873e54a9130d1e83862bb2e7993f55'
mlflow.set_tracking_uri("https://dagshub.com/Kamal254/Deblur-Image-autoencoder.mlflow")

client = MlflowClient()
model_name = "autoencoder"
champion_model_version = client.get_model_version_by_alias(
                                    name=model_name,
                                    alias="Champion").version
if(champion_model_version):
    model_uri = f"models:/{model_name}/{champion_model_version}"
    Model = mlflow.keras.load_model(model_uri)



@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})



def read_file_as_image(data) -> np.ndarray:
    img = Image.open(BytesIO(data))
    img = img.resize((128, 128))
    img = np.array(img) / 255.0  # Normalize the image
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    ip_image = read_file_as_image(await file.read())
    predictions = Model.predict(tf.convert_to_tensor(ip_image))
    
    # Since predictions already has the shape (1, 128, 128, 3), we just need to remove the batch dimension
    output_array = predictions[0]  # Remove the batch dimension

    # Convert the predictions back to the image
    output_image = (output_array * 255).astype(np.uint8)  # Scale back to 0-255
    output_image_pil = Image.fromarray(output_image)
    
    buf = BytesIO()
    output_image_pil.save(buf, format='PNG')
    buf.seek(0)
    
    return StreamingResponse(buf, media_type="image/png")

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)