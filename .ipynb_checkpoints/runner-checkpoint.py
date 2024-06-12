import os
import numpy as np
import tensorflow as tf
from typing import Union
from preprocessing import DataPreprocessing

class Runner:
    def __init__(self) -> None:
        self.model = tf.keras.layers.TFSMLayer('sentiment_model', call_endpoint="serving_default")
        self.SENTIMENTS = [
            'Very Positive',
            'Positive',
            'Somewhat Positive',
            'Neutral',
            'Somewhat Negative',
            'Negative',
            'Very Negative'
        ]
        
    def run(self, user_input: str) -> Union[str, np.ndarray]:
        preprocessor = DataPreprocessing(user_input)
        preprocess_data = preprocessor.preprocess()
        prediction = self.model(preprocess_data)
        return self.SENTIMENTS[np.argmax(prediction)], prediction