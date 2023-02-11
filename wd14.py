from env import *
from config import *

import tensorflow as tf

from keras.models import Model
from keras.layers import *

from huggingface_hub import hf_hub_download


def download_tagger(repo, model_dir):
    print("downloading wd14 tagger model from hf_hub")
    for file in FILES:
        hf_hub_download(repo, file, cache_dir=model_dir, force_filename=file)
    for file in SUB_DIR_FILES:
        hf_hub_download(
            repo, file, 
            subfolder=SUB_DIR, 
            cache_dir=os.path.join(model_dir, SUB_DIR),
            force_filename=file
        )


def replace_classifier(
    model: Model, 
    classifier_func,
    freeze_original=False
):
    if freeze_original:
        model.trainable = False
    return Model(inputs=model.input, outputs=[
        classifier_func(model.layers[-2].input)
    ])