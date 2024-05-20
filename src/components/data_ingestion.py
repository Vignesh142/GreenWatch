from ctypes import Union
from dataclasses import dataclass
import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from src.constants import IMAGE_SIZE, BATCH_SIZE
import tensorflow as tf


@dataclass
class DataIngestionConfig:
    data_path: str = os.path.join('PlantVillage')


class DataIngestion:
    def __init__(self):
        self.config = DataIngestionConfig()

    def load_data(self, data_path: str) -> tf.data.Dataset:
        '''
        Load data from the given path and create a dataset
        Returns the tf.data.Dataset object
        '''
        try:
            data_path = self.config.data_path
            self.dataset = tf.keras.preprocessing.image_dataset_from_directory(
                data_path,
                shuffle=True,
                image_size=(IMAGE_SIZE, IMAGE_SIZE),
                batch_size=BATCH_SIZE
            )
        except Exception as e:
            logging.error(f"Error in loading data: {str(e)}")
            raise CustomException(f"Error in loading data: {str(e)}")
        return self.dataset

    def get_dataset_partition(self, train_split, val_split, test_split, shuffle=True, shuffle_size=10000, seed=42) -> Union[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        '''
        Split the dataset into train, validation and test sets
        Returns the train, validation and test datasets'''
        try:
            self.dataset_size = len(self.dataset)
            if shuffle:
                self.dataset = self.dataset.shuffle(shuffle_size, seed=seed)

            train_size = int(train_split * self.dataset_size)
            val_size = int(val_split * self.dataset_size)
            test_size = int(test_split * self.dataset_size)

            self.train_ds = self.dataset.take(train_size)
            self.val_ds = self.test_dataset.skip(val_size).take(val_size)
            self.test_ds = self.dataset.skip(train_size + val_size)

            return self.train_ds, self.val_ds, self.test_ds
        except Exception as e:
            logging.error(f"Error in splitting data: {str(e)}")
            raise CustomException(f"Error in splitting data: {str(e)}")

    def get_classes(self):
        '''
        Get the class names from the dataset
        Returns the class names
        '''
        try:
            self.class_names = self.dataset.class_names
            return self.class_names
        except Exception as e:
            logging.error(f"Error in getting class names: {str(e)}")
            raise CustomException(f"Error in getting class names: {str(e)}")
if __name__ == "__main__":
    obj = DataIngestion()
    data_ingestion = obj.load_data('PlantVillage')
    classes = obj.get_classes()
    print(classes)
    train_ds, val_ds, test_ds= obj.get_dataset_partition(0.7, 0.2, 0.1)
    print(data_ingestion.train_ds)
    print(data_ingestion.val_ds)
    print(data_ingestion.test_ds)