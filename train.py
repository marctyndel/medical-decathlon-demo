import datetime
import os
import tensorflow as tf
import keras as K
import numpy as np

from data import load_data
from model import unet
from plot_inference_examples import plot_results
from utils import *

import foundations


def train_and_predict(params):
    data_path = params['data_path']
    data_filename = params['data_filename']
    batch_size = params['batch_size']
    n_epoch = params['epochs']
    channels_first = params['channels_first']

    if channels_first:
        K.backend.set_image_data_format("channels_first")

    SESS = tf.Session(config=CONFIG)
    K.backend.set_session(SESS)

    """
    Step 1: Load the data
    """
    hdf5_filename = os.path.join(data_path, data_filename)
    print("-" * 30)
    print("Loading the data from HDF5 file ...")
    print("-" * 30)

    imgs_train, msks_train, imgs_validation, msks_validation, \
    imgs_testing, msks_testing = load_data(hdf5_filename, params['batch_size'],
                                           [params['crop_dim'], params['crop_dim']],
                                           params['channels_first'], params['seed'])
#    print(imgs_train.shape)
#    print(msks_train.shape)
    imgs_train = imgs_train[:5000]
    msks_train = msks_train[:5000]
    imgs_testing = imgs_testing[:2000]
    msks_testing = msks_testing[:2000]
    imgs_validation = imgs_validation[:500]
    msks_validation = msks_validation[:500]
#    print(imgs_train.shape)
#    print(msks_train.shape)
#    exit()
    

    print("-" * 30)
    print("Creating and compiling model ...")
    print("-" * 30)

    """
    Step 2: Define the model
    """
    unet_model = unet(channels_first=params['channels_first'],
                      fms=params['featuremaps'],
                      output_path=params['output_path'],
                      inference_filename=params['inference_filename'],
                      batch_size=params['batch_size'],
                      blocktime=params['blocktime'],
                      num_threads=params['num_threads'],
                      learning_rate=params['learningrate'],
                      num_inter_threads=params['num_inter_threads'],
                      use_upsampling=params['use_upsampling'],
                      use_dropout=params['use_dropout'],
                      print_model=params['print_model'],
                      dropout=params['dropout'])

    model = unet_model.create_model(imgs_train.shape, msks_train.shape)

    model_filename, model_callbacks = unet_model.get_callbacks()

    """
    Step 3: Train the model on the data
    """
    print("-" * 30)
    print("Fitting model with training data ...")
    print("-" * 30)

    model.fit(imgs_train[:3000], msks_train[:3000],
              batch_size=batch_size,
              epochs=n_epoch,
              validation_data=(imgs_validation, msks_validation),
              verbose=1, shuffle="batch",
              callbacks=model_callbacks)
    """
    Step 4: Evaluate the best model
    """
    print("-" * 30)
    print("Loading the best trained model ...")
    print("-" * 30)

    unet_model.evaluate_model(model_filename, imgs_testing, msks_testing)

    """
    Step 5: Plot inference example
    """
    png_directory = "inference_examples"
    if not os.path.exists(png_directory):
        os.makedirs(png_directory)

    indicies_testing = [40, 61, 102, 210, 371,
                        400, 1093, 1990, 1991, 1995, 1996]
#, 3540, 4485,
#                        5566, 5675, 6433]

    for idx in indicies_testing:
        plot_results(model, imgs_testing, msks_testing,
                     idx, png_directory='inference_examples')
    

__name__ = "__main__"

if __name__ == "__main__":
    
#    params = foundations.load_parameters()
    params = generate_params()
    foundations.log_params(params)

    CONFIG = tf.ConfigProto(intra_op_parallelism_threads=params['num_threads'],
                        inter_op_parallelism_threads=params['num_inter_threads'])
    CONFIG.gpu_options.allow_growth = True

    # download_data()
    train_and_predict(params)

    print("Stopped script on {}".format(datetime.datetime.now()))

