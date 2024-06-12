import os
import numpy as np
import tensorflow as tf
from typing import Union
from transformers import TFAutoModel
from model import BertSentimentAnalysis
from preprocessing import DataPreprocessing

class Runner:
    def __init__(self) -> None:
        self.bert_model = TFAutoModel.from_pretrained("bert-base-uncased")
        self.model = BertSentimentAnalysis(self.bert_model, num_classes=7)
        self.model.load_weights('./weights/sentiment_model')
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