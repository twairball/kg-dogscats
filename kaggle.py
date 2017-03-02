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
def submit(preds, test_batches, filepath):
    filenames = test_batches.filenames

    def img_names(filenames):    
        # np.array([int(f[8:f.find('.')]) for f in filenames])
        df = pd.DataFrame(filenames, columns=['filenames'])
        df['id'] = df['filenames'].str.extract('/(\w*).jpg').astype(int)
        return df[['id']]

    def do_clip(arr, mx): 
        return np.clip(arr, (1-mx)/9, mx)

    def preprocess(preds):
        isdog = preds[:,1]
        subm = do_clip(isdog, 0.97)
        return pd.DataFrame(subm, columns=['label'])
        
    ids = img_names(filenames)
    submission = preprocess(preds)
    submission = pd.concat([ids, submission], axis=1)

    print(submission.head())
    print("saving to csv: " + filepath)
    submission.to_csv(filepath, index=False, compression='gzip')
    return submission


def push_to_kaggle(filepath):
    command = "kg submit -c dogs-vs-cats-redux-kernels-edition " + filepath
    os.system(command)


if __name__ == "__main__":
    print("======= making submission ========")
    preds = load_array('data/results/preds.h5/')
    test_batch = get_batches('data/test/')
    submit(preds, test_batch, 'submits/base_subm.gz')

    print("======= pushing to kaggle ========")
    push_to_kaggle('submits/base_subm.gz')