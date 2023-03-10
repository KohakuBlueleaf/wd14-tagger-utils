'''
Tensorflow/Keras env config
'''
PRECISION = "mixed_float16"
SEED = 1


'''
WD1.4 tagger config
'''
WD14_TAGGER_REPO = 'SmilingWolf/wd-v1-4-swinv2-tagger-v2'
FILES = ["keras_metadata.pb", "saved_model.pb", "selected_tags.csv"]
SUB_DIR = "variables"
SUB_DIR_FILES = ["variables.data-00000-of-00001", "variables.index"]
CSV_FILE = FILES[-1]
MODEL_DIR = f'./models/{WD14_TAGGER_REPO.split("/")[-1]}'


'''
TRAIN config
'''
DATASET_FOLDER = 'DATASET'
EVAL_DATASET_FOLDER = '.DATASET'
CLASSES = 203
SAVE_PATH = './model_out'

TRAIN_BATCH = 4
VALID_BATCH = 32
VALID_SPLIT = 0.25

EPOCH = 20
BASE_LR = 1e-3
LR_DECAY = 0.8
SAVE_FREQ = 10