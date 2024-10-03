## Restore Blur Image Using Autoencoder
![](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) ![](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white) ![](https://img.shields.io/badge/flask-%23000.svg?style=for-the-badge&logo=flask&logoColor=white) ![](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi) ![Kubernetes](https://img.shields.io/badge/kubernetes-%23326ce5.svg?style=for-the-badge&logo=kubernetes&logoColor=white) ![mlflow](https://img.shields.io/badge/mlflow-%23d9ead3.svg?style=for-the-badge&logo=numpy&logoColor=blue)


This repository contains a Flask application that deblurs images using an residual connection-based autoencoder model with accuracy **PSNR : 28.3** . The model is based on TensorFlow. 
The Deep Model used in this project is inspired by the paper : [Link to the paper](https://arxiv.org/pdf/1812.11262)

Click [here](https://restore-blurred-images.onrender.com/) to deblur your Image 😀

### Introduction

This Flask application provides a simple interface for users to upload a blurred image and receive a deblurred version. The backend leverages an autoencoder model, optimized using TensorFlow Lite for efficient and fast inference.

![](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcS578X5jC8sjBBcsNwWXtrD-JXqDt5r45H49g&s)

### Project Architecture
This project is a comprehensive Machine Learning Pipeline designed to restore blurred images using an autoencoder model. The architecture is built for scalability, flexibility, and efficient deployment, leveraging various tools for orchestration, tracking, and continuous integration. Each component is scalable independent to any other component.


- **Data Ingestion and Preparation:** Downloads and preprocesses image datasets for model training, including resizing and normalization.
- **Model Building:** Implements an autoencoder with residual connections for image deblurring.
- **Training and Experiment Tracking:** Uses Kubeflow for pipeline orchestration and MLflow for tracking experiments and model versions.
- **Model Evaluation:** Evaluates model performance using PSNR and SSIM metrics for selecting the best model.
- **Deployment and CI/CD:** Deploys the model with Flask, integrated with Jenkins for continuous integration and deployment.
- **Orchestration and Workflow Management:** Manages the pipeline using Kubeflow, ensuring automated and scalable workflows.

### Technologies Used
 - **Kubeflow**: For pipeline orchestration and workflow automation.
 - **MLflow**: For experiment tracking, model versioning, and metric logging.
 - **Jenkins**: For CI/CD pipeline automation and seamless deployment.
 - **Flask**: For building the web application to serve the deblurring model.
 - **Autoencoder**: For deblurring image restoration with residual connections.

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
git clone https://github.com/Kamal254/Deblur-Image-autoencoder.git
cd Deblur-Image-autoencoder
```

###### Install Requirements
```bash
pip install -r requirements.txt
```

###### Run the Application
Flask Application will run of the 8000 port.
```bash
python app.py
```
