
import os
import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from PIL import Image
from skimage.color import rgb2grey
from skimage.feature import hog
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, auc, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

df = pd.read_csv("data1.csv")
dir = "C:\Users\youss\Desktop\segmentation_WBC-master\Dataset 1"
b = df['label'].tolist()
aa = np.array(b)
labels =aa.reshape(300,1)


y=[]
for i in range(len(labels)):
    y.extend(labels[i])
    
def get_images(path):
    imgs = []
   
    valid_images = [".bmp"]
    for f in os.listdir(path):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_images:
            continue
        imgs.append(cv2.imread((os.path.join(path,f))))
    return imgs

def create_features(img):
    # flatten three channel color image
    color_features = img.flatten()
    # convert image to greyscale
    grey_image = rgb2grey(img)
    # get HOG features from greyscale image
    flat_features = np.hstack(color_features)
    return flat_features

def create_feature_matrix(images):
    features_list = []
    
    for i in range(len(images)):
        # load image
        
        # get features for image
        image_features = create_features(images[i])
        features_list.append(image_features)
        
    # convert list of arrays into a matrix
    feature_matrix = np.array(features_list)
    return feature_matrix
x=get_images(dir)
# run create_feature_matrix on our dataframe of images
feature_matrix = create_feature_matrix(x)

# get shape of feature matrix
print('Feature matrix shape is: ', feature_matrix.shape)

# define standard scaler
ss = StandardScaler()
# run this on our feature matrix
bees_stand = ss.fit_transform(feature_matrix)

pca = PCA(n_components=500)
# use fit_transform to run PCA on our standardized matrix
bees_pca = ss.fit_transform(bees_stand)
# look at new shape
print('PCA matrix shape is: ', bees_pca.shape)
