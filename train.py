from wd14 import download_tagger, replace_classifier

import tensorflow as tf

from keras import optimizers, losses
from keras import Sequential
from keras.models import load_model
from keras.layers import (
    Dense,
    Dropout,
    RandomFlip,
    RandomRotation,
    RandomTranslation,
    RandomContrast,
    RandomZoom,
)
from keras.activations import sigmoid, softmax
from keras.utils import image_dataset_from_directory
from keras.callbacks import LearningRateScheduler, ModelCheckpoint

from config import (
    SEED,
    WD14_TAGGER_REPO,
    MODEL_DIR,
    DATASET_FOLDER,
    CLASSES,
    SAVE_PATH,
    TRAIN_BATCH,
    VALID_BATCH,
    VALID_SPLIT,
    EPOCH,
    BASE_LR,
    LR_DECAY,
    SAVE_FREQ,
)


'''
Load Dataset
'''
train_dataset = image_dataset_from_directory(
    DATASET_FOLDER,
    image_size = (448, 448),
    batch_size = TRAIN_BATCH,
    seed = SEED,
    subset = 'training',
    validation_split = VALID_SPLIT
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
    image_size = (448, 448),
    batch_size = VALID_BATCH,
    seed = SEED,
    subset = 'validation',
    validation_split = VALID_SPLIT
)


'''
Load model and hijack it
'''
download_tagger(WD14_TAGGER_REPO, MODEL_DIR)
model = load_model(MODEL_DIR)

classes = CLASSES
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
cp_callback = ModelCheckpoint(
    filepath = SAVE_PATH, 
    save_freq = SAVE_FREQ,
)
base_lr = BASE_LR
def scheduler(epoch, _):
    return base_lr * LR_DECAY**epoch

lr_sch = LearningRateScheduler(scheduler)
new_model.compile(
    optimizer = optimizers.Adam(base_lr),
    loss = losses.SparseCategoricalCrossentropy(),
    metrics = ['accuracy'],
)

history = new_model.fit(
    train_dataset,
    validation_data = val_dataset,
    epochs = EPOCH,
    callbacks = [lr_sch, cp_callback],
    verbose = 1
)

new_model.save(SAVE_PATH)