from __future__ import print_function
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.wrappers.scikit_learn import KerasRegressor
from keras.layers.wrappers import TimeDistributed
from keras.layers import ELU
from keras.constraints import maxnorm
from keras.layers.core import RepeatVector, Activation, Dropout, Dense, Lambda, Flatten, TimeDistributedDense
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.recurrent import LSTM, GRU, SimpleRNN
from keras.callbacks import LearningRateScheduler
from keras.layers.advanced_activations import PReLU
from keras.layers.normalization import BatchNormalization
from sklearn.preprocessing import MinMaxScaler
from keras.optimizers import SGD, RMSprop, Adadelta
from keras.utils.visualize_util import plot
from keras.utils import np_utils
from keras import backend as K
from skimage import transform, color, exposure
from skimage.transform import rotate
from skimage.viewer import ImageViewer
from imutils import paths
from common import anorm2, draw_str
from time import clock
import skimage as skimage
import numpy as np
import argparse
import sklearn
import pandas
import cv2
import video
import math

seed = 7
np.random.seed(seed)  # for reproducibility

def step_decay(epoch):
    initial_lrate = 0.1
    drop = 0.5
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return float(lrate)

def baseline_model(optimizer='adam', init_mode='uniform'):
    model = Sequential()
    # model.add(Convolution2D(32, 2, 2, border_mode='valid', input_shape=(2, 32, 32), activation='relu'))
    # model.add(Convolution2D(32, 1, 1, activation='relu'))
    # model.add(MaxPooling2D(pool_size=(1, 1), dim_ordering="tf"))
    # model.add(Dropout(0.15))
    # # model.add(TimeDistributed(1536, input_dim=6144, init=init_mode, activation="relu"))
    # # model.add(LSTM(512, return_sequences=False))
    # # model.add(LSTM(256, return_sequences=False))
    # model.add(Flatten())
    # model.add(Dense(1))
    model.add(Lambda(
      lambda x: x/127.5 - 1.,
      input_shape=(2, 480, 640),
      output_shape=(2, 480, 640))
    )

    # Several convolutional layers, each followed by ELU activation
    # 8x8 convolution (kernel) with 4x4 stride over 16 output filters
    # model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
    # model.add(ELU())
    # # 5x5 convolution (kernel) with 2x2 stride over 32 output filters
    # model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
    # model.add(ELU())
    # # 5x5 convolution (kernel) with 2x2 stride over 64 output filters
    # model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
    # # Flatten the input to the next layer
    # model.add(Flatten())
    # # Apply dropout to reduce overfitting
    # model.add(Dropout(.2))
    # model.add(ELU())
    # # Fully connected layer
    # model.add(Dense(512))
    # # More dropout
    # model.add(Dropout(.2))
    # model.add(ELU())
    # # Fully connected layer with one output dimension (representing the speed).
    # model.add(Dense(1))

    model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu', input_shape=(2, 480, 640)))
    model.add(BatchNormalization(mode=0, axis=1))
    model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
    model.add(BatchNormalization(mode=0, axis=1))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
    model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu'))
    model.add(BatchNormalization(mode=0, axis=1))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(LSTM(512, return_sequences=True))
    model.add(TimeDistributed(Dropout(0.5)))
    model.add(TimeDistributed(Dense(128, activation='relu')))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")
    plot(model, to_file='model_lstm_cnn.png')
    print("[INFO] compiling uniform batch normalization model...")
    return model

class App:
    def __init__(self, video_src):
        self.cap = cv2.VideoCapture(video_src)
        self.frames = []
        self.data = []
        self.speeds = []
        self.X_train = []
        self.Y_train = []
        self.X_test = []
        self.Y_test = []
        self.model = None
        self.dataset = dict(pandas.read_json("drive.json"))
        self.frame_idx = 0

    def draw_flow(self, img, flow, step=20):
        h, w = img.shape[:2]
        y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
        fx, fy = flow[y,x].T
        lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
        lines = np.int32(lines + 0.5)
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.polylines(vis, lines, 0, (0, 255, 0))
        for (x1, y1), (x2, y2) in lines:
            cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
        return vis

    def get_flow(self):
        print("[INFO] constructing flow matrices and speed truth...")
        while (self.cap.isOpened()):
            ret, frame = self.cap.read()
            # Rexposes, blurs, and makes frame grayscale in order to smoothen it out.
            if ret and self.frame_idx < 10:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                rescale_gray = skimage.exposure.rescale_intensity(gray,out_range=(0,255))
                blur_gray = cv2.GaussianBlur(rescale_gray, (21, 21), 0)
                self.frames.append(blur_gray)
                if (self.frame_idx > 0):
                    prev_frame, curr_frame = self.frames[self.frame_idx-1], self.frames[self.frame_idx]
                    flow_matrix = cv2.calcOpticalFlowFarneback(prev_frame, curr_frame, 0.5, 3, 15, 2, 5, 1.2, 0)
                    self.data.append(flow_matrix)
                    speed = self.dataset[1][self.frame_idx]
                    self.speeds.append(speed)
                    cv2.imshow('flow', self.draw_flow(gray, flow_matrix))
                self.frame_idx += 1
            else:
                self.cap.release()
                cv2.destroyAllWindows()

            ch = 0xFF & cv2.waitKey(1)
            if ch == 27:
                break

    def construct_data(self):
        self.data = np.array(self.data)
        self.speeds = np.array(self.speeds)
        scale = MinMaxScaler(feature_range=(0, 1))
        ascolumns = self.data.reshape(-1, 2)
        dataset = scale.fit_transform(ascolumns)
        dataset = dataset.reshape(self.data.shape)
        (self.X_train, self.X_test, self.Y_train, self.Y_test) = train_test_split(dataset, self.speeds, test_size=0.15, random_state=42)
        # self.X_train = scale.fit_transform(self.X_train)
        # # self.X_test = scale.transform(self.X_test)
        self.X_train = self.X_train.reshape(self.X_train.shape[0], 2, 480, 640).astype('float32')
        self.X_test = self.X_test.reshape(self.X_test.shape[0], 2, 480, 640).astype('float32')

    def train_model(self):
        lrate = LearningRateScheduler(step_decay)
        callbacks_list = [lrate]
        print(self.X_train.shape)
        self.model = baseline_model()

        #self.model = KerasRegressor(build_fn=baseline_model, nb_epoch=150, batch_size=60, verbose=1)
        self.model.fit(self.X_train, self.Y_train, nb_epoch=50, batch_size=64, validation_split=0.15, shuffle=True, callbacks=callbacks_list, verbose=1)
        # self.model.save('lstm_model.h5')

    def evaluate_model(self):
        print("[INFO] evaluating on testing set...")
        (loss, accuracy) = self.model.evaluate(self.X_test, self.Y_test, batch_size=64, verbose=1)
        print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss, accuracy * 100))
        output = self.model.predict(self.X_test)
        print(output)
        for i in range(0, len(output)):
            print("actual: %f, model: %f" % (self.Y_test[i], output[i]))
        score = mean_squared_error(self.Y_test, output)
        print("SCORE: %f" % score)

def main():
    import sys
    video_src = "drive.mp4"
    app = App(video_src)
    app.get_flow()
    app.construct_data()
    app.train_model()
    app.evaluate_model()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
