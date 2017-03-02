from __future__ import division,print_function

import os, json, sys
from glob import glob
from shutil import copyfile
import numpy as np

#import modules
from utils import *
from vgg16 import Vgg16


## 
## Train
##

# pathing
path = "data/kaggle_dogscats"
path = "data/kaggle_dogscats/sample"  # use smaller sample 
results_path = path + '/results'
test_path = path + '/test'

# training settings
batch_size = 64
nb_epoch = 3

# model
vgg = Vgg16()
batches = vgg.get_batches(path+'/train', batch_size=batch_size)
val_batches = vgg.get_batches(path+'/valid', batch_size=batch_size*2)
vgg.finetune(batches)
vgg.fit(batches, val_batches, nb_epoch=nb_epoch)

# save model weights
vgg.model.save_weights(results_path + '/model_weights.h5')


##
## Predict
##
batches, preds = vgg.test(test_path, batch_size = batch_size * 2)
filenames = batches.filenames

#Save our test results arrays so we can use them again later
save_array(results_path + 'test_preds.dat', preds)
save_array(results_path + 'filenames.dat', filenames)

# predict 0 or 1 for probability

#Grab the dog prediction column
isdog = preds[:,1]
print("Raw Predictions: " + str(isdog[:5]))
print("Mid Predictions: " + str(isdog[(isdog < .6) & (isdog > .4)]))
print("Edge Predictions: " + str(isdog[(isdog == 1) | (isdog == 0)]))


#So to play it safe, we use a sneaky trick to round down our edge predictions
#Swap all ones with .95 and all zeros with .05
isdog = isdog.clip(min=0.05, max=0.95)


#Extract imageIds from the filenames in our test/unknown directory 
filenames = batches.filenames
ids = np.array([int(f[8:f.find('.')]) for f in filenames])

subm = np.stack([ids,isdog], axis=1)

# write to csv
submission_file_name = 'kaggle_dogscats_vgg_submission.csv'
np.savetxt(submission_file_name, subm, fmt='%d,%.5f', header='id,label', comments='')
