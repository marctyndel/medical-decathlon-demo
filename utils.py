import subprocess
import os
import urllib.request
import os
import numpy as np


def download_data(directory="../144x144"):
    print("(loading preprocessed data)")
    if not os.path.exists(directory):
        os.makedirs(directory)
    url = "http://deeplearni-public.s3.amazonaws.com/medical_demo/decathlon_brats.h5"
    urllib.request.urlretrieve(url, "{}/decathlon_brats.h5".format(directory))
    print("(loaded preprocessed data)")



def generate_params():
    params = {
#        "batch_size": int(2**np.random.randint(7, 9)),
	"batch_size": 64,
        "epochs": 1,#int(np.random.randint(1,5, size=1)),  
        "learningrate": float(np.random.choice([0.0001, 0.001])),
        'dropout': float(np.random.uniform(0.1, 0.6)), 
        "weight_dice_loss": float(np.random.uniform(0.5, 0.95)),
        "featuremaps": int(np.random.choice([32,64])),
        "crop_dim": -1,
        "blocktime": 1,
        "print_model": True,
        "seed": 816,
        "use_augmentation": True,
        "use_dropout": True,
        "use_upsampling": False,
        "channels_first": False,
        'num_threads':  80,
        'num_inter_threads':  1,
#        "data_path": os.path.join("../144x144/"),
        "data_path": os.path.join("../data/"),
        "data_filename": "decathlon_brats.h5",
        "inference_filename": "unet_model_for_decathlon.hdf5",
        "output_path": "./output/",
    }
    return params



if __name__ == '__main__':
    download_data("data/")

