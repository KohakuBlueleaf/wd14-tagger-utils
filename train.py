from wd14 import *

import tensorflow as tf

from keras import optimizers, losses
from keras import Sequential
from keras.models import load_model
from keras.layers import *
from keras.activations import sigmoid, softmax
from keras.utils import image_dataset_from_directory
from keras.callbacks import LearningRateScheduler

from config import *


'''
Load Dataset
'''
train_dataset = image_dataset_from_directory(
    DATASET_FOLDER,
    image_size=(448, 448),
    batch_size=1,
    seed=SEED,
    subset='training',
    validation_split=0.5
)
augmentation = Sequential([
    RandomFlip("horizontal", seed=SEED),
    RandomRotation(0.5, seed=SEED),
    RandomTranslation(0.25, 0.25, seed=SEED),
    RandomContrast(0.3, seed=SEED),
    RandomZoom(0.3, seed=SEED)
])
train_dataset = train_dataset.map(lambda x, y : (augmentation(x), y), num_parallel_calls=tf.data.AUTOTUNE)

val_dataset = image_dataset_from_directory(
    DATASET_FOLDER,
    image_size=(448, 448),
    batch_size=32,
    seed=SEED,
    subset='validation',
    validation_split=0.5
)


'''
Load model and hijack it
'''
download_tagger(WD14_TAGGER_REPO, MODEL_DIR)
model = load_model(MODEL_DIR)

classes = 33
multi = False
dropout = Dropout(0.35, name='classifier_drop', seed=SEED)

def classifier(x):
    x = dropout(x)
    x = Dense(classes)(x)
    if multi:
        x = sigmoid(x)
    else:
        x = softmax(x)
    return x

new_model = replace_classifier(
    model, 
    classifier_func = classifier,
    freeze_original = True
)
new_model.summary(line_length=155)


'''
Training!
'''
base_lr = 1e-3
def scheduler(epoch, _):
    return base_lr * 0.9**epoch

lr_sch = LearningRateScheduler(scheduler)
new_model.compile(
    optimizer = optimizers.Adam(1e-3),
    loss = losses.SparseCategoricalCrossentropy(),
    metrics = ['accuracy'],
)

history = new_model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=50,
    callbacks=[lr_sch],
    verbose=1
)

new_model.save('')