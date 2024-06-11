import keras
import numpy as np
import pandas as pd
from PIL import Image
from keras.utils.image_utils import img_to_array

model= keras.models.load_model("D:\\Repos\\REU Summer 2024\\archive\\EfficientNetB0-525-(224 X 224)- 98.97.h5", custom_objects={'F1_score':'F1_score'})

firstcsvline = True
dataframe = pd.read_csv('birds.csv', header = None, names = ['class id', 'image_path', 'labels', 'data set', 'scientific name'])
images = []
for file_path in dataframe['image_path']:
    if firstcsvline:
        firstcsvline = False
        continue
    image = Image.open(file_path)
    image_array = img_to_array(image)
    images.append(image_array)

input_data = np.array(images)