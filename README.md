# Plant Disease Classification Using Tensorflow/Keras 
This project is a plant disease classification project using Convolutional Neural Networks (CNN). The dataset used in this project is the [PlantVillage Dataset]https://www.kaggle.com/emmarex/plantdisease). 
## Installation
1. Clone the repository
```bash
git clone https://github.com/Vignesh142/Plant_Disease_Prediction
```
2. Install the required libraries
```bash
pip install -r requirements.txt
```
3. Run the Website
```bash
uvicorn main:app --reload --port 3000
```
4. Open the Website
```bash
http://localhost:3000
```

## Usage
1. Open the Website
2. Upload an image of a plant leaf
4. The model will classify the plant disease and display the result

## Directory Structure
```
E2E_Potato_Disease_Classification
│   src
│   │   components
│   │   │   data_ingestion.py
│   │   │   data_transformation.py
│   │   │   model_trainer.py
│   │   pipeline
│   │   │   data_pipeline.py
│   │   │   model_pipeline.py
|   api
|   |   main.py
|   |   static
|   |   |   home.html
|   |   |   images
|   |   |   |   image.jpg
|   models
│   │   version-1
│   │   │   model.pb
│   │   version-2
│   │   │   model.pb
│   utils
│   setup.py
│   requirements.txt
```

## Dataset
The dataset used in this project is the [PlantVillage Dataset](https://www.kaggle.com/emmarex/plantdisease). Only Potato Plant leafs are classfied. The dataset is divided into two parts: training and testing.

## Model
The model used in this project is a Convolutional Neural Network (CNN). The model is trained on the training dataset and tested on the testing dataset. The model is evaluated on the testing dataset using accuracy and loss metrics.

## Results
The model achieved an accuracy of 98.5% on the testing dataset. The model is able to classify plant diseases and healthy plants with high accuracy.

## Conclusion
The model is able to classify plant diseases and healthy plants with high accuracy. The model can be used to detect plant diseases early and prevent crop loss.
## References
1. [PlantVillage Dataset](https://www.kaggle.com/emmarex/plantdisease)

