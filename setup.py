import os
import logging
from train import TrainingController
from utils import download_blobs, download_model, upload_blob_to_bucket, upload_model

# APP AND HYPERPARAMETERS CONFIGURATION

# Main operation, you can set train or gen
CMD = os.getenv('CMD')

#for voice 512 for music 2048
N_FFT = int(os.getenv('N_FFT'))

# Number of sample for generating file (file lenght
SAMPLE_NUMBER = int(os.getenv('SAMPLE_NUMBER'))

# Path to directory, where model will be saved
MODEL_OUTPUT= os.getenv('MODEL_OUTPUT')
if not os.path.exists(MODEL_OUTPUT):
    os.makedirs(MODEL_OUTPUT)

# Log level (if set to debug - debug log level on, esle info level log is set)
LOG_LEVEL = os.getenv('LOG_LEVEL')

# Logging configuration
level_logs = logging.INFO
if(LOG_LEVEL == 'debug'):
    level_logs = logging.DEBUG
logging.basicConfig(format='%(asctime)s; %(levelname)s: %(message)s', level=level_logs)


# Download model from cloud storage repository
MODEL_BUCKET = os.getenv('MODEL_BUCKET')
MODEL_NAME = None
if MODEL_BUCKET != 'none':
    MODEL_NAME = os.getenv('MODEL_NAME')
    download_model(MODEL_BUCKET,MODEL_NAME, MODEL_OUTPUT)
else:
    MODEL_BUCKET = None

# Initialize training controller
training_controller = TrainingController(MODEL_BUCKET, MODEL_NAME)

# Run specyfic functionality based on global command
if CMD == 'train':
    DISCR_EPOCH_MUL = int(os.getenv('DISCR_EPOCH_MUL'))
    TRAINING_SET_PATH= os.getenv('TRAINING_SET_PATH')
    if not os.path.exists(TRAINING_SET_PATH):
        os.makedirs(TRAINING_SET_PATH)
    EPOCH = int(os.getenv('EPOCH'))
    BATCH_SIZE = int(os.getenv('BATCH_SIZE'))
    SAVE_INTERVAL = int(os.getenv('SAVE_INTERVAL'))
    # Download samples from GCP Storage
    TRAINING_SET_BUCKET = os.getenv('TRAINING_SET_BUCKET')
    if TRAINING_SET_BUCKET != 'none':
        download_blobs(TRAINING_SET_BUCKET,TRAINING_SET_PATH)

    training_controller.train(TRAINING_SET_PATH, N_FFT, MODEL_OUTPUT, EPOCH, BATCH_SIZE, SAVE_INTERVAL, SAMPLE_NUMBER, DISCR_EPOCH_MUL)

else:
    OUTPUT_FILE = os.getenv('OUTPUT_FILE')
    if not os.path.exists(OUTPUT_FILE):
        os.makedirs(OUTPUT_FILE)
    training_controller.generate(MODEL_OUTPUT,OUTPUT_FILE,N_FFT,SAMPLE_NUMBER)