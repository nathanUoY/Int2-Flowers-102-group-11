#Make all neccessary imports (SOME AREN'T NEEDED, WILL REMOVE LATER)
import tensorflow as tf
import tensorflow_datasets as tfds
from keras.models import Sequential
from tensorflow.keras import layers
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt

#load the dataset with tfds and split into train, validate and test
flowers, flowers_info = tfds.load('oxford_flowers102', as_supervised = True, with_info = True)
flowers_train_raw, flowers_valid_raw, flowers_test_raw = flowers['train'], flowers['validation'], flowers['test']

#express the dataset in its splits
training_length = len(flowers_train_raw)
validation_length = len(flowers_valid_raw)
test_length = len(flowers_test_raw)

print('There are ', training_length, ' images in training split.')
print('There are ', validation_length, ' images in validation split.')
print('There are ', test_length, ' images in test split.')

#print image shape and label for 3 images in training split
for image, label in flowers_train_raw.take(3):
  print('\nShape: ', image.shape)
  print('Label: ', label)

#plot an image from the training split
for image, label in flowers_train_raw.take(1):
  image = image.numpy().squeeze()
  plt.imshow(image)
  plt.colorbar()
  plt.title(label.numpy())
  plt.show()
  

#find average image height and width for resizing
totalHeight = 0
totalWidth = 0
for image, label in flowers_train_raw:
  totalHeight += image.shape[0]
  totalWidth += image.shape[1]

avHeight = totalHeight / training_length 
avWidth = totalWidth / training_length
avDimen = round((avWidth + avHeight) / 2)

print(avDimen)

#testing out image resizing and rescaling
def imageAug(image, label):
  image = tf.cast(image, tf.float32)
  #resizes image
  image = tf.image.resize(image, (avDimen, avDimen))
  #normalise the pixel values
  image = image / 255.0
  return image, label

flowers_train = flowers_train_raw.map(imageAug)
flowers_validation = flowers_valid_raw.map(imageAug)
flowers_test = flowers_test_raw.map(imageAug)

#plots same image from before but agumented
for image, label in flowers_train.take(1):
  print("Augmented image shape: ", image.shape)
  image = image.numpy().squeeze()
  plt.imshow(image)
  plt.colorbar()
  plt.title(label.numpy())
  plt.show()

#build the model
#haven't given this much thought yet just testing to see if it works
model = Sequential()
#first layer
model.add(Conv2D(32, (5, 5), padding = 'Same', activation = 'relu', input_shape = (581, 581, 3)))
model.add(MaxPooling2D(pool_size = (2, 2)))
#second layer
model.add(Conv2D(32, (3, 3), padding = 'Same', activation = 'relu', input_shape = (581, 581, 3)))
model.add(MaxPooling2D(pool_size = (2, 2)))
#third layer
model.add(Conv2D(32, (3,3), padding = 'Same', activation = 'relu', input_shape = (581, 581, 3)))
model.add(MaxPooling2D(pool_size = (2, 2)))
#flatten layer
model.add(Flatten())
model.add(Dense(units = 128, activation = 'relu'))
#output layer
model.add(Dense(units = 102, activation = 'relu'))

print(model.summary())
