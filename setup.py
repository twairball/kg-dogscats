from __future__ import division,print_function

import os, json, sys
from glob import glob
from shutil import copyfile
import numpy as np

#import modules
from utils import *
from vgg16 import Vgg16

import errno
def mkdir_p(path):
    """ 'mkdir -p' in Python """
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

##
## Setup training / validation datasets

def setup_datasets():
    #Create references to important directories we will use over and over
    current_dir = os.getcwd()
    LESSON_HOME_DIR = current_dir
    DATA_HOME_DIR = current_dir+'/data/kaggle_dogscats'

    #Allow relative imports to directories above lesson1/
    sys.path.insert(1, os.path.join(sys.path[0], '..'))

    #Create directories
    for folder in ['valid', 'results', 
        'sample/train', 'sample/test', 'sample/valid', 'sample/results',
        'test/unknown']

        folder_path = DATA_HOME_DIR + '/' + folder
        mkdir_p(folder_path)


    # randomly _move_ 2000 training images to validation
    g = glob(DATA_HOME_DIR + '/train/*.jpg')
    shuf = np.random.permutation(g)
    for i in range(2000): os.rename(shuf[i], DATA_HOME_DIR+'/valid/' + shuf[i])

    # randomly _copy_ 200 training images to sample
    g = glob(DATA_HOME_DIR + '/train/*.jpg')
    shuf = np.random.permutation(g)
    for i in range(200): copyfile(shuf[i], DATA_HOME_DIR+'/sample/train/' + shuf[i])

    # randomly _copy_ 50 validation images to sample
    g = glob(DATA_HOME_DIR + '/valid/*.jpg')
    shuf = np.random.permutation(g)
    for i in range(50): copyfile(shuf[i], DATA_HOME_DIR+'/sample/valid/' + shuf[i])

    #Divide cat/dog images into separate directories

    # %cd $DATA_HOME_DIR/sample/train
    # %mkdir cats
    # %mkdir dogs
    # %mv cat.*.jpg cats/
    # %mv dog.*.jpg dogs/

    # %cd $DATA_HOME_DIR/sample/valid
    # %mkdir cats
    # %mkdir dogs
    # %mv cat.*.jpg cats/
    # %mv dog.*.jpg dogs/

    # %cd $DATA_HOME_DIR/valid
    # %mkdir cats
    # %mkdir dogs
    # %mv cat.*.jpg cats/
    # %mv dog.*.jpg dogs/

    # %cd $DATA_HOME_DIR/train
    # %mkdir cats
    # %mkdir dogs
    # %mv cat.*.jpg cats/
    # %mv dog.*.jpg dogs/

    # Create single 'unknown' class for test set
    # %cd $DATA_HOME_DIR/test
    # %mv *.jpg unknown/
