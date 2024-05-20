import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt

from src.constants import IMAGE_SIZE, BATCH_SIZE, EPOCHS
from src.exception import CustomException
from src.utils import save_object
from src.logger import logging
from traitlets import default


@dataclass
class ModelTrainerConfig:
    trained_model_path: str = 'models'


class ModelTrainer:

    def __init__(self, n_classes: int):
        self.config = ModelTrainerConfig()
        self.n_classes = n_classes

    def initiate_model(self):
        '''
        Initialize the model layers and compile it
        Returns the model
        '''
        try:
            resize_and_rescale = tf.keras.Sequential([
                layers.Resizing(IMAGE_SIZE, IMAGE_SIZE),
                layers.Rescaling(1. / 255)
            ])

            data_augmentation = tf.keras.Sequential([
                layers.RandomFlip("horizontal_and_vertical"),
                layers.RandomRotation(0.2),
                layers.RandomZoom(0.2),
            ])

            input_shape = (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3)

            self.model = tf.keras.Sequential([
                resize_and_rescale,
                data_augmentation,

                layers.Conv2D(64, (3,3), activation= 'relu', input_shape=input_shape),
                layers.MaxPooling2D((2,2)),
                layers.Conv2D(64, (3,3), activation='relu'),
                layers.MaxPooling2D((2,2)),
                layers.Conv2D(32, (3,3), activation='relu'),
                layers.MaxPooling2D((2,2)),
                layers.Conv2D(32, (3,3), activation='relu'),
                layers.MaxPooling2D((2,2)),

                layers.Flatten(),
                layers.Dense(64, activation='relu'),
                layers.Dense(self.n_classes, activation='softmax')
            ])
            self.model.build(input_shape=input_shape)

            self.model.compile(
                optimizer="adam",
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics=["accuracy"]
            )
            print(self.model.summary())
            logging.info("Model initiated successfully")
        except Exception as e:
            logging.error(f"Error in initiating model: {str(e)}")
            raise CustomException(f"Error in initiating model: {str(e)}")

    def train_model(self, train_ds, val_ds):
        '''
        Train the model on the given dataset
        Returns the trained model
        '''
        try:
            self.history = self.model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=EPOCHS,
                batch_size = BATCH_SIZE,
                verbose=1
            )
            logging.info("Model trained successfully")
            self.save_model()

        except Exception as e:
            logging.error(f"Error in training model: {str(e)}")
            raise CustomException(f"Error in training model: {str(e)}")
    
    def save_model(self, type='keras'):
        '''
        Save the trained model in different types
        params: type: str: 'h5' or 'pb' or 'keras'
        '''
        try:
            if type == 'h5':
                # Use the native Keras format for saving models
                self.model.save(f'{self.config.trained_model_path}/my_model.h5')
            elif type == 'pb':
                # Save the model in SavedModel format
                self.model.export(f'{self.config.trained_model_path}/my_model')
            elif type == 'keras':
                # Save the model in Keras format
                self.model.save(f'{self.config.trained_model_path}/my_model.keras')
            else:
                logging.error("Invalid model type")
                raise CustomException("Invalid model type")
            logging.info("Model saved successfully")

        except Exception as e:
            logging.error(f"Error in saving model: {str(e)}")
            raise CustomException(f"Error in saving model: {str(e)}")
    def plot_accuracy(self):
        '''
        Plot the accuracy of the model
        '''
        try:
            acc = self.history.history['accuracy']
            val_acc = self.history.history['val_accuracy']

            loss = self.history.history['loss']
            val_loss = self.history.history['val_loss']

            plt.figure(figsize=(8,8))
            plt.subplot(1,2,1)
            plt.plot(acc, label="Training Accuracy")
            plt.plot(val_acc, label="Validation Accuracy")
            plt.legend(loc="lower right")
            plt.title("Training and Validation Accuracy")

            plt.subplot(1,2,2)
            plt.plot(range(EPOCHS), loss, label="Training Loss")
            plt.plot(range(EPOCHS), val_loss, label="Validation Loss")
            plt.legend(loc="upper right")
            plt.title("Training and Validation Loss")
            plt.show()
        except Exception as e:
            logging.error(f"Error in plotting accuracy: {str(e)}")
            raise CustomException(f"Error in plotting accuracy: {str(e)}")
        
    def predict_img(self, img, class_names):
        img_arr = tf.expand_dims(img, 0) # create a batch 
        
        predictions = self.model.predict(img_arr)
        
        predicted_class = class_names[np.argmax(predictions[0])]
        confidence = round(100 * (np.max(predictions[0])), 2)
        
        plt.imshow(img.numpy().astype("uint8"))
        plt.title(f"Predicted: {predicted_class}\n Confidence: {confidence}%")
    
if __name__=="__main__":
    model = ModelTrainer(38)
    model.initiate_model()
    # model.train_model()
    # model.save_model()
    # model.save_model(type='keras')
    model.save_model(type='pb')
    # model.plot_accuracy()