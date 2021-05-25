import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from numpy.core.defchararray import endswith
from numpy.core.shape_base import hstack
import pandas as pd
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import layers
import  random


### Loading images names in the foldar
entries = os.listdir('C:/Users/Engah/Desktop/Pattern/segmentation_WBC-master/Dataset 1')

### Images names listing
colored_images = []     
grey_images = []

### For randomly suffling the data For better accuracy

randomlist = random.sample(range(0 , 300) , 300)


### Creating Lists of Images names in the foldar 
for i in range(len(entries)):
    if entries[i].endswith('bmp'):
        colored_images.append(entries[i])
    else:
        grey_images.append(entries[i])

image_path = 'C:/Users/Engah/Desktop/Pattern/segmentation_WBC-master/Dataset 1/'+ colored_images[1]


#### Importing the Labels of the images from the csv file

a=pd.read_csv('C:/Users/Engah/Desktop/Pattern/segmentation_WBC-master/Class Labels of Dataset 1.csv')
b = a['class label'].tolist()
aa = np.array(b)
labels =aa.reshape(300,1)

#################

testing_size = len(grey_images)*.1  #### 30 testing image

Training_Data = np.zeros((270,120,120))
Testing_Data = np.zeros((30,120,120))

### Classes

classess = np.array([1,2,3,4,5])
Traing_data_labels = []     ### True Y values for training examples
Testing_data_labels = []    ### True y For testing examples

##### Data as pairs "image,label"

Full_data = []
paired_train_data = []
paired_test_data = []

####  Counters To correctly adding the data
counnter = 0
counnter1 = 0

####
#
#   The for loop is used to create the different lists of the images in the dataset foldar.
#   prepairing the images was the main problem i faced and i did alot of processing on the images.
#   First i started to pad the images and resize them to match the 120x120 Dimension 
#
####

for i in range(len(grey_images)):

    if i > (len(grey_images)*.9 - 1):
        image_path = 'C:/Users/Engah/Desktop/Pattern/segmentation_WBC-master/Dataset 1/'+ grey_images[randomlist[counnter]]
        image = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
        delta_w = 120 - image.shape[1]
        delta_h = 120 - image.shape[0]
        top, bottom = delta_h//2, delta_h-(delta_h//2)
        left, right = delta_w//2, delta_w-(delta_w//2)
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT,value=0)
        print(image.shape)     
        Testing_data_labels.append(labels[randomlist[counnter]][0] -1)
        Testing_Data[counnter] = image
        Full_data.append([image,labels[counnter]])
        paired_test_data.append([image,labels[counnter]])
        counnter+=1
    else:   
        image_path = 'C:/Users/Engah/Desktop/Pattern/segmentation_WBC-master/Dataset 1/'+ grey_images[randomlist[i]]
        image = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE) 
        delta_w = 120 - image.shape[1]
        delta_h = 120 - image.shape[0]
        top, bottom = delta_h//2, delta_h-(delta_h//2)
        left, right = delta_w//2, delta_w-(delta_w//2)
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT,value=0)
        print(image.shape)
        Training_Data[i] = image
        Traing_data_labels.append(labels[randomlist[i]][0]-1 )       
        Full_data.append([image,labels[i]])
        paired_train_data.append([image,labels[i]])
        counnter1 += 1

X = Training_Data
Y= []
X = np.array(X)

for i in range(len(Traing_data_labels)):
    Y.append(int(Traing_data_labels[i]))

X = np.array(X).reshape(-1,120,120,1)
Y = np.array(Y)

####################################################

num_classes = 5

########## Creating The Network ##############

model = Sequential(
[
    layers.experimental.preprocessing.Rescaling(1./255, input_shape=(120, 120, 1)),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    # layers.MaxPooling2D(),            ##### adds one more hidden layer
    layers.Flatten(),
    layers.Dense(num_classes, activation=tf.nn.softmax)
]
)

model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history=model.fit(X, Y, batch_size = 1, epochs=20)


########## Prepairing Testing Data ##############

X_test = Testing_Data
Y_test= []
X_test = np.array(X_test)

for i in range(len(Testing_data_labels)):
    Y_test.append(int(Testing_data_labels[i]))

X_test = np.array(X_test).reshape(-1,120,120,1)
Y_test = np.array(Y_test)

##### Model Evaluation (Accuracy,Loss) ######

model.evaluate(X_test , Y_test)

########################## Plotting and Printing ########################

########### Accuracy ##############

plt.plot(history.history['accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

########## Loss Measure ################

plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

######### Printing Shapes #############

# print(X.shape)
# print(Y.shape)
# print(Y_test)
# print(X_test.shape)
# print(Y_test.shape)
