import os
import random

import tensorflow as tf
from keras import mixed_precision
import numpy as np

from config import *


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["PATH"] += "./cuda_lib"

mixed_precision.set_global_policy(PRECISION)


def set_seeds(seed=SEED):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)


def set_global_determinism(seed=SEED):
    set_seeds(seed=seed)
    
    # os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

# Call the above function with seed value
set_global_determinism(seed=SEED)


print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))