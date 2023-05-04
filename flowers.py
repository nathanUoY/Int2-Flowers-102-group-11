#Make all neccessary imports
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib as plt

#load the dataset with tfds and split into train, validate and test
flowers, flowers_info = tfds.load('oxford_flowers102', as_supervised= True, with_info= True)
flowers_train, flowers_valid, flowers_test = flowers['train'], flowers['validation'], flowers['test']
