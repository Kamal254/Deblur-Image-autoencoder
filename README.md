## Restore Blur Image Using Autoencoder
![](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) ![](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white) ![](https://img.shields.io/badge/flask-%23000.svg?style=for-the-badge&logo=flask&logoColor=white) ![](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)

This repository contains a Flask application that deblurs images using an autoencoder model. The model is based on TensorFlow Lite with 8-bit integer quantization, derived from the original trained model.
Click [here](https://restore-blurred-images.onrender.com/) to deblur your Image 😀

### Introduction

This Flask application provides a simple interface for users to upload a blurred image and receive a deblurred version. The backend leverages an autoencoder model, optimized using TensorFlow Lite for efficient and fast inference.

![](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcS578X5jC8sjBBcsNwWXtrD-JXqDt5r45H49g&s)

### Model Architecture

Autoencoders are neural networks that you can use to compress and reconstruct data. The encoder compresses input, and the decoder attempts to recreate the information from this compressed version.

![](https://miro.medium.com/v2/resize:fit:750/format:webp/0*LtrxkZrn87VTYML6.png)

#### Features

- Image Upload: Upload blurred images directly through the web interface.
- Deblurring: The application deblurs the uploaded image using a TensorFlow Lite 8-bit integer quantized autoencoder model.
- Real-time Inference: The quantized model ensures quick deblurring with minimal resource usage.

### Kaggle Dataset

Model trained on 20k Synthetic Faces Dataset Available on kaggle [here](https://www.kaggle.com/datasets/selfishgene/synthetic-faces-high-quality-sfhq-part-1)

### Installation

###### Clone the Repo
```bash
git clone https://github.com/your-username/image-deblurring-flask.git
cd image-deblurring-flask
```

###### Install Requirements
```bash
pip install -r requirements.txt
```

###### Run the Application
```bash
python app.py
```
