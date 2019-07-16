import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from keras.models import load_model
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle

def load_data(test=False):
    FTRAIN = 'data/training.csv'
    FTEST = 'data/test.csv'
    fname = FTEST if test else FTRAIN
    df = read_csv(os.path.expanduser(fname))  # load dataframes

    df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))

    df = df.dropna() 
    X = np.vstack(df['Image'].values) / 255.  
    X = X.astype(np.float32)
    X = X.reshape(-1, 96, 96, 1)

    if not test:  
        y = df[df.columns[:-1]].values
        y = (y - 48) / 48 
        X, y = shuffle(X, y, random_state=42)  
        y = y.astype(np.float32)
    else:
        y = None

    return X, y
