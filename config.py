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
DATASET_FOLDER = './dataset/...'
CLASSES = 33
MODEL_PATH = 'PATH TO TRAINED MODEL'