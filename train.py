from utils import wav2spect, spect2wav, normalize_spectrums, upload_model, ModelsSufix, plot_spectrum
from models.cnn_voice_model import CNNSoundTransferModel
import numpy as np
import os
from logging import log, info, debug, basicConfig, DEBUG, INFO
import tensorflow as tf

class TrainingController:
    def __init__(self, model_bucket= None, model_blob_path= None):
        self.model_bucket = model_bucket
        self. model_blob_path = model_blob_path

    def save_callback(self, model_path_gen, model_content_path_disc, model_style_path_disc):
        info('Uploading model to bucket %s'%(self.model_bucket))
        upload_model(self.model_bucket, self.model_blob_path, model_content_path_disc, model_style_path_disc, model_path_gen)

    def load_and_fft_dataset(self, training_set_path, sample_number, N_FFT):
        train_data = []
        sr_lists = []
        for filename in os.listdir(training_set_path):
            if "wav" not in filename: 
                continue
            info('Creating FFT for file: %s',filename)
            spectrum, sr = wav2spect(training_set_path+'/'+filename, N_FFT)
            plot_spectrum(spectrum,filename+'.png')
            train_data.append(np.asarray(spectrum))
            debug("SR", sr)
            sr_lists.append(sr)       
        width = len(train_data[0])
        train_data = normalize_spectrums(train_data,width,sample_number)
        train_data_norm= np.asarray(train_data)
        return width, train_data_norm, sr_lists

    def train(self, training_content_set_path, training_style_set_path, N_FFT, model_path, epochs=1000, batch=4, save_interval=100, sample_number = 1025):
        
        #Open files and fft them
        info('Training started')
        c_width, c_training_data, c_sr = self.load_and_fft_dataset(training_content_set_path,sample_number, N_FFT)
        s_width, s_training_data, s_sr = self.load_and_fft_dataset(training_style_set_path,sample_number, N_FFT)

        # Set parameters for model
        info('C_width %d and S_width %d'%(c_width, s_width))
        width = c_width
        channels = 1

        # Check if batch is not bigger then training set.
        if batch > len(c_training_data):
            batch = len(c_training_data)
            info('Batch is bigger then dataset sample, changing batch size to %d'%(batch))

        gan = CNNSoundTransferModel(width,sample_number,channels,model_path)

        #train model
        info("Start training")
        save_bucket_callback = None
        if self.model_bucket != None:
            info('Exporting models mode to GCP bucket is turned on')
            save_bucket_callback = self.save_callback
        gan.train(c_training_data, s_training_data, epochs, batch, save_interval, save_bucket_callback)

    def generate(self, model_path, content_path, otputfile, N_FFT, sample_number = 1025, sr = 22050):
        # TODO: Add no model found exception here
        gan = tf.keras.models.load_model(model_path + ModelsSufix.GEN)
        c_w, content_inputs, sr = self.load_and_fft_dataset(content_path,sample_number, N_FFT)
        spectrums = gan.predict(content_inputs) 
        spec_num = 0
        for single_data in spectrums:
            sp_data= np.squeeze(single_data)
            plot_spectrum(sp_data,'gen_spec_all_%d.png'%(spec_num))
            spect2wav(sp_data, sr, otputfile+'/gen_music_%d.wav'%(spec_num), N_FFT)
