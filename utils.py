import os
import librosa
import numpy as np
from logging import log, info, debug, basicConfig, DEBUG, INFO, error
from google.cloud import storage
from google.api_core.exceptions import NotFound
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

def plot_spectrum(gen_spectrum, image_name):
    plt.figure(figsize=(5, 5))
    # we then use the 2nd column.
    plt.subplot(1, 1, 1)
    plt.title("CNN Voice Transfer Result")
    plt.imsave(image_name, gen_spectrum[:400, :])

def wav2spect(filename, N_FFT):
    x, sr = librosa.load(filename)
    S = librosa.stft(x, N_FFT)
    p = np.angle(S)
    S = np.log1p(np.abs(S))
    return S, sr

def spect2wav(spectrum, sr, outfile, N_FFT):
    a = np.exp(spectrum) - 1
    p = 2 * np.pi * np.random.random_sample(spectrum.shape) - np.pi
    for i in range(50):
        S = a * np.exp(1j * p)
        x = librosa.istft(S)
        p = np.angle(librosa.stft(x, N_FFT))
    librosa.output.write_wav(outfile, x, sr)

def normalize_spectrums(spectrum_array,width, height):
    result_list = []
    for spect in spectrum_array:
        debug('Start to normalizing first spectrum of size: %s', spect.shape)
        reshaped_spect = np.resize(spect,(width,height))
        result_list.append(reshaped_spect)
        debug('Changed to array of size: %s', reshaped_spect.shape)
    return result_list

def download_blobs(source_bucket_name, local_directory):
    storage_client = storage.Client()
    blob_list = storage_client.list_blobs(source_bucket_name)
    i = 0
    for blob_item in blob_list: 
        with open(local_directory+'/sample'+str(i)+'.wav','wb') as file_obj:
            storage_client.download_blob_to_file(blob_item, file_obj)
        i = i+1

def download_model(model_bucket_name, model_name, model_local_path):
    try:
        info('Downloading models from GCP storage repository')
        storage_client = storage.Client()
        bucket = storage_client.bucket(model_bucket_name)
        secure_download_single_blob(bucket, model_name + ModelsSufix.DICSR_CONT,model_local_path + ModelsSufix.DICSR_CONT)
        secure_download_single_blob(bucket, model_name + ModelsSufix.DICSR_STYLE,model_local_path + ModelsSufix.DICSR_STYLE)
        secure_download_single_blob(bucket, model_name + ModelsSufix.GEN,model_local_path + ModelsSufix.GEN)
    except NotFound:
        error('No model repository found.')

def secure_download_single_blob(bucket, blob_path, local_path):
    blob = bucket.blob(blob_path)
    if blob.exists():
            blob.download_to_filename(local_path)
    else:
        error('Blob %s not exists'%(blob_path))

def upload_model(model_bucket, model_blob_path, model_content_path_disc, model_style_path_disc, model_path_gen):
    storage_client = storage.Client()
    bucket = storage_client.bucket(model_bucket)
    upload_single_model(bucket,model_blob_path + ModelsSufix.DICSR_CONT, model_content_path_disc)
    upload_single_model(bucket,model_blob_path + ModelsSufix.DICSR_STYLE, model_style_path_disc)
    upload_single_model(bucket,model_blob_path + ModelsSufix.GEN, model_path_gen)

def upload_single_model(bucket, bucket_path, local_path):
    blob = bucket.blob(bucket_path)
    blob.upload_from_filename(local_path)

def upload_blob_to_bucket(filepath, outpu_bucket_name ):
    storage_client = storage.Client()
    bucket = storage_client.bucket(outpu_bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)

def prepare_local_path(path):
    if not os.path.exists(path):
        info('Path %s not existing - creating directory'%(path))
        os.makedirs(path)

class ModelsSufix:
    GEN = '/gen.h5'
    DICSR_CONT = '/desc_con.h5'
    DICSR_STYLE = '/desc_style.h5'