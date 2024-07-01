import keras
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from keras.preprocessing.image import img_to_array
from keras.applications.resnet50 import preprocess_input
np.set_printoptions(threshold=np.inf)
pd.set_option('display.max_rows', 1000)

# model path on laptop C:\\Users\\acosc\\REU_Summer_2024\\EfficientNetB0-525-(224 X 224)- 98.97.h5
# model path on desktop D:\\Repos\\REU_Summer_2024\\EfficientNetB0-525-(224 X 224)- 98.97.h5
# model = keras.models.load_model("D:\\Repos\\REU_Summer_2024\\EfficientNetB0-525-(224 X 224)- 98.97.h5", custom_objects={'F1_score':'F1_score'})
model = keras.models.load_model("D:\\Repos\\REU_Summer_2024\\trained_model_1.h5", custom_objects={'F1_score':'F1_score'})

# csv path on laptop C:\\Users\\acosc\\REU_Summer_2024\\birds.csv
# csv path on desktop D:\\Repos\\REU_Summer_2024\\birds.csv
dataframe = pd.read_csv('D:\\Repos\\REU_Summer_2024\\birds.csv', names = ['class id', 'image_path', 'labels', 'data set', 'scientific name'])
testdf = dataframe[dataframe['image_path'].str.startswith('test/')].iloc[500*5-1:]
images = []
for file_path in testdf['image_path']:
    image = Image.open(file_path)
    image = image.resize((50, 50))
    image_array = img_to_array(image)
    image_array = preprocess_input(image_array)
    images.append(image_array)

input_data = np.array(images)
pred = model.predict(input_data)
'''print(pred[1])
plt.figure(figsize=(8,4))
plt.plot(pred[2], marker=',', linestyle='-', color='b')
plt.yscale('log')
plt.xlabel('Species ID')
plt.ylabel('Probabilities')
plt.grid(True)
plt.tight_layout()
plt.show()'''

pred_classes = np.argmax(pred, axis = 1)
class_labels = dataframe.loc[1:, 'class id'].unique()
print(pred_classes)
print(class_labels)
'''
for i, pred_class in enumerate(pred_classes):
    x1 = testdf.iloc[i]['image_path']
    x2 = testdf.iloc[i]['class id']
    x3 = class_labels[pred_class]
    print(f"Image: {x1}, Predicted: {x2}, Actual: {x3}")'''