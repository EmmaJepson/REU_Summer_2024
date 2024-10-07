import keras
import tensorflow
print (tensorflow.__version__)
print (keras.__version__)
from keras.preprocessing.image import ImageDataGenerator
from keras import layers
#from tensorflow.python.keras.engine.sequential import Sequential
#from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.python.keras.engine import data_adapter

def _is_distributed_dataset(ds):
    return isinstance(ds, data_adapter.input_lib.DistributedDatasetSpec)

data_adapter._is_distributed_dataset = _is_distributed_dataset

batch_size = 40
img_height = 244
img_width = 244
epochs = 10

# path on laptop C:\\Users\\acosc\\REU_Summer_2024
# path on desktop D:\\Repos\\REU_Summer_2024
# path on HPC /home/egj8n4
dataset_folder = "/home/egj8n4"

# data loading and preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
)

train_generator = train_datagen.flow_from_directory(
    dataset_folder + '/train',  # Path to training data
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
)

validation_generator = train_datagen.flow_from_directory(
    dataset_folder + '/valid',  # Path to validation data
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
)

base_model = tensorflow.keras.applications.ResNet50V2(include_top = False,)

base_model.trainable = False

inputs = tensorflow.keras.layers.Input(shape = (300, 300, 3), name = "input_layer")

x = base_model(inputs)

x = tensorflow.keras.layers.GlobalAveragePooling2D(name = "global_avg_pool_layer")(x)

outputs = tensorflow.keras.layers.Dense(525, activation = "softmax", name = "output_layer")(x)

inception_model = tensorflow.keras.Model(inputs, outputs)

# compile
inception_model.compile(optimizer=tensorflow.optimizers.Adam(learning_rate = 0.01), loss='categorical_crossentropy', metrics=['accuracy'])

# train
history = inception_model.fit(
    train_generator,
    epochs = epochs,
    validation_data = validation_generator,
    steps_per_epoch = len(train_generator),
    validation_steps = int(0.25 * len(validation_generator))
)

inception_model.save('resnet50v2_model.h5')