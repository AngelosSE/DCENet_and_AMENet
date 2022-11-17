"""
It appears that results may vary although the random seeds are set. I am unable 
to figure out why...
"""
#--> https://stackoverflow.com/questions/32419510/how-to-get-reproducible-results-in-keras
# Seed value
# Apparently you may use different seed values at each stage
seed_value= 0

# 1. Set the `PYTHONHASHSEED` environment variable at a fixed value
import os
os.environ['PYTHONHASHSEED']=str(seed_value)
os.environ['TF_DETERMINISTIC_OPS']='1'

# 2. Set the `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)

# 3. Set the `numpy` pseudo-random generator at a fixed value
import numpy as np
np.random.seed(seed_value)

# 4. Set the `tensorflow` pseudo-random generator at a fixed value
import tensorflow as tf
tf.random.set_seed(seed_value)
# for later versions: 
tf.compat.v1.set_random_seed(seed_value)

# 5. Configure a new global `tensorflow` session ## Angelos: One should not use a session according to https://www.tensorflow.org/api_docs/python/tf/config/experimental/enable_op_determinism, but I find that there seems to be less randomness with it.
#from keras import backend as K
#session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
#sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
#K.set_session(sess)
# for later versions:
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
tf.compat.v1.keras.backend.set_session(sess)
#<--
import ame_trainer # AMEnet
import Trans_trainer # DCEnet
import angelos
import extract_ind_traj
import os
import shutil
import pathlib
import time

do_train = False
if do_train is True:
    extract_ind_traj.main()
    timestr = time.strftime("%Y%m%d-%H%M%S")
    for model,name in [
                        (ame_trainer,'amenet'),
                        (Trans_trainer,'dcenet')
                        ]:
        # Clear data cached by model
        shutil.rmtree(pathlib.Path(__file__).parent / f'../processed_data/train',ignore_errors=True) 
        os.mkdir(pathlib.Path(__file__).parent / f'../processed_data/train')
        model.main(do_train=True
                    ,model_name=f'{name}_{timestr}_{0}'
                    ,learning_rate=3e-4
                    ,pretrained_model=None)
        for i in range(1,3): # run fine-tuning twice
            model.main(do_train=True
                                ,model_name=f'{name}_{timestr}_{i}'
                                ,learning_rate=1e-4
                                ,pretrained_model=f'{name}_{timestr}_{i-1}'
                                )
else:
    models = [
                (ame_trainer,'amenet_20221109-090647_2'),
                (Trans_trainer,'dcenet_20221109-090647_2')]
    #models = [
    #            (ame_trainer,'amenet_20221109-132912_2'),
    #            (Trans_trainer,'dcenet_20221109-132912_2')]
   #models = [
   #            (ame_trainer,'amenet_20221110-220124_2'),
   #            (Trans_trainer,'dcenet_20221110-220124_2')
   #            ]

    for model,name in models:
        # Clear data cached by model
        shutil.rmtree(pathlib.Path(__file__).parent / f'../processed_data/train',ignore_errors=True) 
        os.mkdir(pathlib.Path(__file__).parent / f'../processed_data/train')
        model.main(do_train=False
            ,model_name=name)

angelos.main('AMENet')
angelos.main('DCENet')