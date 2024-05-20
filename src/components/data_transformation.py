import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
import tensorflow as tf

from src.constants import IMAGE_SIZE, BATCH_SIZE

@dataclass
class DataTransformationConfig:
    data_path: str = os.path.join('PlantVillage')
    target: str = 'label'

class DataTransformation:

    def __init__(self):
        self.config = DataTransformationConfig()

    def transform_data(self, train_ds, val_ds, test_ds):
        train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
        test_ds = test_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
        val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
        return train_ds, val_ds, test_ds
        