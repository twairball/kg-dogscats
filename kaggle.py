from __future__ import division,print_function

import pandas as pd
import numpy as np
import os, json, sys
import os.path
from glob import glob
from utils import *


"""
    Util methods for submitting to kaggle
"""
def classnames():
    return ['cats', 'dogs']

def submit(preds, test_batches, filepath):

    def do_clip(arr, mx): 
        return np.clip(arr, (1-mx)/9, mx)
    
    def img_names(filenames):
        return np.array([int(f[8:f.find('.')]) for f in filenames])

    #Grab the dog prediction column
    isdog = preds[:,1]
    print("Raw Predictions: " + str(isdog[:5]))
    print("Mid Predictions: " + str(isdog[(isdog < .6) & (isdog > .4)]))
    print("Edge Predictions: " + str(isdog[(isdog == 1) | (isdog == 0)]))

    # clip
    isdog = do_clip(isdog, 0.97)

    #Extract imageIds from the filenames in our test/unknown directory 
    ids = img_names(test_batches.filenames)
    subm = np.stack([ids,isdog], axis=1)

    # write to csv
    np.savetxt(filepath, subm, fmt='%d,%.5f', header='id,label', comments='')


def push_to_kaggle(filepath):
    command = "kg submit -c dogs-vs-cats-redux-kernels-edition " + filepath
    os.system(command)

if __name__ == "__main__":
    print("======= making submission ========")
    preds = load_array('data/results/preds.h5/')
    test_batch = get_batches('data/test/')
    submit(preds, test_batch, 'subm.gz')

    print("======= pushing to kaggle ========")
    push_to_kaggle('subm.gz')