from cog import BasePredictor, Input, Path
import os
from typing import List
import cv2
import numpy as np
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dropout
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model

# import profanity_check
from PIL import Image
from cog import BasePredictor, Path, Input

from process import *

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.model = load_model("./demo_lstm.h5")

    # The arguments and types the model takes as input
    def predict(self,
            video_path: Path = Input(description="Input video file (MP4 or MOV)")
        ) -> Path:
        """Run a single prediction on the model"""
        destination_folder = "root-data/demo_sign"


        shutil.move(video_path, destination_folder)

        dataset, labels = process_videos(destination_folder)


        output = self.model(dataset)
        return postprocess(output)
