#MODULE IMPORTS
import numpy as np
import os
import glob
import cv2
import warnings
from matplotlib import pyplot

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.utils import np_utils
from sklearn.metrics import classification_report

np.random.seed(2017)
warnings.filterwarnings("ignore")


#CLASSES AND FUNCTIONS
class ProcessImages(object):

    def __init__(self):
        self.resize_dim = []

    def set_resize_dim(self, height, width):
        self.resize_dim = (height, width)

        print("Image will be resized to {dimension}".format(dimension=self.resize_dim))

    def load_img(self, filename):
        img = cv2.imread(filename)

        return img

    def resize_img(self, image):
        resized_img = cv2.resize(image, self.resize_dim, cv2.INTER_LINEAR)

        return resized_img

    def split2_img(self, image, show_images=False):
        img_height = image.shape[0]
        img_width = image.shape[1]

        if img_height >= img_width:
            threshold = int(img_height / 2)
            image_1 = image[0:threshold, :]
            image_2 = image[threshold:img_height, :]
        else:
            threshold = int(img_width / 2)
            image_1 = image[:, 0:threshold]
            image_2 = image[:, threshold:img_width]

        if show_images:
            plot_prep = np.concatenate((image_1, image_2), axis=1)
            pyplot.imshow(plot_prep)

        return image_1, image_2

    def load_batch_images(self):

        X_train = []
        train_id = []
        y_train = []

        self.MAINFOLDER = os.getcwd()
        os.chdir(self.MAINFOLDER)

        FOLDERS = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']

        for fld in FOLDERS:
            index = FOLDERS.index(fld)
            print('Load folder {} (Index: {})'.format(fld, index))
            path = os.path.join(self.MAINFOLDER, fld, '*.jpg')
            files = glob.glob(path)
            for fl in files:
                flbase = os.path.basename(fl)
                img = self.load_img(fl)
                img = self.resize_img(img)
                X_train.append(img)
                train_id.append(flbase)
                y_train.append(index)

        print('Convert to numpy...')
        train_data = np.array(X_train, dtype=np.uint8)
        train_target = np.array(y_train, dtype=np.uint8)

        print('Reshape...')
        train_data = train_data.transpose((0, 3, 1, 2))

        print('Convert to float...')
        train_data = train_data.astype('float32')
        train_data /= 255
        train_target = np_utils.to_categorical(train_target, 8)

        return train_data, train_target, train_id


class Model(object):

    def __init__(self, features, target):
        self.features = features
        self.target = target
        self.features_shape = features.shape[1:]
        self.model = []
        self.num_classes = target.shape[1]

    @staticmethod
    def convert_to_class(y_proba):
        y_class = []

        for yy in y_proba:
            y_class += [np.where(yy==max(yy))[0][0]]

        return y_class

    def train_test_split(self, p=0.7):
        rows = np.arange(0, self.target.shape[0])
        np.random.shuffle(rows)

        threshold = int(p * len(rows))
        train_rows = rows[0:threshold]
        test_rows = rows[threshold:len(rows)]

        train_features = self.features[train_rows]
        test_features = self.features[test_rows]

        train_target = self.target[train_rows]
        test_target = self.target[test_rows]

        return train_features, test_features, train_target, test_target

    def initialize_model(self):

        #CONVOLUTIONAL PART
        # INFO:
        # https://adeshpande3.github.io/adeshpande3.github.io/A-Beginner's-Guide-To-Understanding-Convolutional-Neural-Networks/
        # https://adeshpande3.github.io/A-Beginner's-Guide-To-Understanding-Convolutional-Neural-Networks-Part-2/

        # INITIALIZE MODEL
        model = Sequential()

        #Zeropadding adds a row of zeros to top/bottom and column of zeros to left/right of features
        model.add(ZeroPadding2D((1,1), input_shape=self.features_shape, dim_ordering='th'))

        # 2D CONVOLUTION: http://www.songho.ca/dsp/convolution/convolution.html#convolution_2d

        # 1-st convolution: This convolution returns an output of (8x8) using a (3x3) filter
        model.add(Convolution2D(8, 3, 3, activation='relu', dim_ordering='th', init='lecun_uniform', subsample=(1, 1)))

        #Add zero padding to convoluted layer
        model.add(ZeroPadding2D((1, 1), dim_ordering='th'))

        # 2-nd convlution: Apply convolution again, with dropout and pool results (see info link)
        model.add(Convolution2D(8, 3, 3, activation='relu', dim_ordering='th', init='lecun_uniform', subsample=(1, 1)))
        model.add(Dropout(0.2))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='th'))

        # 3-rd convolution: Same as above, different dimension
        model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
        model.add(Convolution2D(16, 3, 3, activation='relu', dim_ordering='th', init='lecun_uniform', subsample=(1, 1)))

        # 4-th convulation: Same as above
        model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
        model.add(Convolution2D(16, 3, 3, activation='relu', dim_ordering='th', init='lecun_uniform', subsample=(1, 1)))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='th'))
        model.add(Dropout(0.2))

        #NEURAL NETWORK PART

        # THE 'INPUT' LAYER - output 32 nodes
        model.add(Flatten())
        model.add(Dense(32, activation='relu', init='lecun_uniform'))
        model.add(Dropout(0.4))

        # THE 'HIDDEN' LAYER - output 32 nodes
        model.add(Dense(32, activation='relu', init='lecun_uniform'))
        model.add(Dropout(0.2))

        #THE 'OUTPUT' LAYER - output 8 nodes (number of classes)
        model.add(Dense(self.num_classes, activation='softmax'))

        sgd = SGD(lr=1e-2, decay=1e-4, momentum=0.9, nesterov=False)
        model.compile(optimizer=sgd, loss='categorical_crossentropy')

        self.model = model


    def fit_model(self):

        if not self.model:
            print('Please initialize model first...')

        x_train, x_test, y_train, y_test = self.train_test_split()

        self.model.fit(x_train, y_train,
                       batch_size=32, nb_epoch=10, verbose=1,
                       validation_data=(x_test, y_test))

        yp_train = self.model.predict(x_train, batch_size=32, verbose=2)
        yp_test = self.model.predict(x_test, batch_size=32, verbose=2)

        yp_tr = self.convert_to_class(yp_train)
        yp_tt = self.convert_to_class(yp_test)
        y_tr = self.convert_to_class(y_train)
        y_tt = self.convert_to_class(y_test)

        return yp_tr, y_tr, yp_tt, y_tt




    def evaluate_model(self):
        pass



#DOCUMENTATION

"""
Dropout consists in randomly setting a fraction rate
of input units to 0 at each update during training time,
which helps prevent overfitting.

"""

#CODE
process_image = ProcessImages()
process_image.set_resize_dim(32, 32)
x_train, y_train, image_ids = process_image.load_batch_images()

first_model = Model(x_train, y_train)
first_model.initialize_model()

yp_tr, y_tr, yp_tt, y_tt = first_model.fit_model()

print(classification_report(y_tr, yp_tr))
"""
             precision    recall  f1-score   support
          0       0.66      0.99      0.79      1196
          1       0.92      0.16      0.27       146
          2       1.00      0.04      0.07        79
          3       0.00      0.00      0.00        49
          4       0.90      0.67      0.77       326
          5       0.96      0.33      0.49       209
          6       0.95      0.77      0.85       137
          7       0.86      0.69      0.77       501
avg / total       0.78      0.74      0.70      2643

"""
print(classification_report(y_tt, yp_tt))

"""
             precision    recall  f1-score   support
          0       0.65      0.98      0.78       523
          1       1.00      0.06      0.11        54
          2       1.00      0.08      0.15        38
          3       0.00      0.00      0.00        18
          4       0.90      0.59      0.71       139
          5       0.79      0.24      0.37        90
          6       0.83      0.77      0.80        39
          7       0.85      0.67      0.75       233
avg / total       0.76      0.71      0.67      1134

"""










