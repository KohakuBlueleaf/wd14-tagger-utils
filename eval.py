from wd14 import download_tagger, replace_classifier

import tensorflow as tf

from keras.models import load_model
from keras.utils import image_dataset_from_directory

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

from config import (
    SEED,
    SAVE_PATH,
    VALID_BATCH,
    VALID_SPLIT,
    EVAL_DATASET_FOLDER
)


'''
Load Dataset
'''
val_dataset = image_dataset_from_directory(
    EVAL_DATASET_FOLDER,
    image_size = (448, 448),
    batch_size = VALID_BATCH,
    seed = SEED,
)


'''
Load model and hijack it
'''
# download_tagger(WD14_TAGGER_REPO, MODEL_DIR)
model = load_model(SAVE_PATH)
model.evaluate(val_dataset)