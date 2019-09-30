import foundations
import numpy as np
import os

# Constant for the number of models to be submitted 
NUM_JOBS = 1

# Get params returns randomly generated architecture specifications 
# and hyperparameters in the form of a dictionary
def generate_params():
    params = {
        "batch_size": int(2**np.random.randint(3, 8)),
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
        "data_path": os.path.join("../144x144/"),
        "data_filename": "decathlon_brats.h5",
        "inference_filename": "unet_model_for_decathlon.hdf5",
        "output_path": "./output/",
    }
    return params
    
# A loop that calls the deploy method from the  
# Foundations SDK which takes in a parameters dictionary
# and the entrypoint script for our code (train.py)
for _ in range(NUM_JOBS):
#    foundations.deploy(
#        env="scheduler",
#        job_directory=".",
#        entrypoint="train.py",
#        project_name="medical_decathlon_0906",
#        params=generate_params(),
#    )
    foundations.submit(
        scheduler_config="scheduler", 
        command=["train.py"], 
        params=generate_params(),
    )

