#Make all neccessary imports
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt

#load the dataset with tfds and split into train, validate and test
flowers, flowers_info = tfds.load('oxford_flowers102', as_supervised= True, with_info= True)
flowers_train,flowers_valid, flowers_test = flowers['train'], flowers['validation'], flowers['test']

#express the dataset in its splits
training_length = flowers_info.splits['train'].num_examples
validation_length = flowers_info.splits['validation'].num_examples
test_length = flowers_info.splits['test'].num_examples
print('There are ', training_length, ' images in traning split.')
print('There are ', validation_length, ' images in validation split.')
print('There are ', test_length, ' images in test split.')

#print image shape and label for 3 images in training split
for image, label in flowers_train.take(3):
    print('\nShape: ', image.shape)
    print('Label: ', label)

#Plot an image from the training split
##CURRENTLY NOT WORKING AS 'PLACEHOLDERS' ARE MISSING.
for image, label in flowers_train.take(1):
    image = image.numpy()
    label = label.numpy()
plt.figure()
plt.imshow(image)
plt.title(label)
plt.colorbar()
plt.grid(False)
plt.show()
