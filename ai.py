import keras
import numpy as np
import pandas as pd
from PIL import Image
from keras.utils.image_utils import img_to_array
from keras.applications.efficientnet import preprocess_input

model = keras.models.load_model("D:\\Repos\\REU_Summer_2024\\EfficientNetB0-525-(224 X 224)- 98.97.h5", custom_objects={'F1_score':'F1_score'})

firstcsvline = True
dataframe = pd.read_csv('D:\\Repos\\REU_Summer_2024\\birds.csv', header = None, names = ['class id', 'image_path', 'labels', 'data set', 'scientific name'])
images = []
for file_path in dataframe['image_path']:
    if firstcsvline:
        firstcsvline = False
        continue
    if file_path.startswith('test/'):
        image = Image.open(file_path)
        image = image.resize((224, 224))
        image_array = img_to_array(image)
        image_array = preprocess_input(image_array)
        images.append(image_array)

input_data = np.array(images)
increment = 0
pred = model.predict(input_data)
pred_classes = np.argmax(pred, axis = 1)
class_labels = dataframe['class id'].unique()

#ignore this
'''for file_path in dataframe['image_path']:
    if firstcsvline:
        firstcsvline = False
        continue
    if file_path.startswith('test/'):
        increment += 1
        x1 = dataframe.loc[file_path, 'file_path']
        print(x1)
        x2 = dataframe.loc[increment, 'class id']
        print(x1)
        x3 = class_labels[pred_classes[increment]]'''


for i, pred_class in enumerate(pred_classes):
    x1 = dataframe[dataframe['image_path'].str.startswith('test/')].iloc[i]['image_path']
    x2 = dataframe[dataframe['image_path'].str.startswith('test/')].iloc[i]['class id']
    x3 = class_labels[pred_class]
    print(f"Image: {x1}, Predicted: {x2}, Actual: {x3}")