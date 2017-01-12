#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import print_function

from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers.wrappers import TimeDistributed
from keras.callbacks import LearningRateScheduler
from keras.layers.core import Dropout, Dense, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.recurrent import LSTM
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from skimage import transform, color, exposure
import skimage as skimage
import numpy as np
import pandas
import math
import cv2

        
# Helper Methods
# ----------------------------------------------------------------------------------------
def draw_flow(img, flow, step=20):
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
    
def step_decay(epoch):
	initial_lrate = 0.1
	drop = 0.5
	epochs_drop = 10.0
	lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
	return float(lrate)
        
        
# Main Program
# ----------------------------------------------------------------------------------------
cap = cv2.VideoCapture('drive.mp4')
dataset = dict(pandas.read_json('drive.json'))
frame_idx = 0
frames = []
data = []
speeds = []
seed = 7
np.random.seed(seed)
K.set_image_dim_ordering('tf')

print('[INFO] Constructing flow matrices and speed truth...')
while (cap.isOpened()):
    ret, frame = cap.read()
    # Rexposes, blurs, and makes frame grayscale in order to smoothen it out.
    if ret and frame_idx < 100:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rescale_gray = skimage.exposure.rescale_intensity(gray,out_range=(0,255))
        blur_gray = cv2.GaussianBlur(rescale_gray, (21, 21), 0)
        frames.append(blur_gray)
        
        if (frame_idx > 0):
            if (frame_idx % 1000 == 0):
                print('[INFO] Processed %d frames' % (frame_idx))
                
            # Get frame speed from JSON
            speed = dataset[1][frame_idx]
            speeds.append(speed)
        
            
            # Calculate optical flow between previous frame and current frame
            prev_frame, curr_frame = frames[frame_idx-1], frames[frame_idx]
            flow_matrix = cv2.calcOpticalFlowFarneback(prev_frame, curr_frame, 0.5, 3, 15, 2, 5, 1.2, 0)
            s = flow_matrix.shape
            data.append(flow_matrix)
            
            # Visualize Flow Matrix
            cv2.imshow('flow', draw_flow(gray, flow_matrix))
        frame_idx += 1
    else:
        cap.release()
        cv2.destroyAllWindows()

    ch = 0xFF & cv2.waitKey(1)
    if ch == 27:
        break
    
print('[INFO] Constructing training data.')
data = np.array(data)
speeds = np.array(speeds)
#ascolumns = data.reshape(-1, 2)
#dataset = scale.fit_transform(ascolumns)
#dataset = dataset.reshape(data.shape)

# Create training and testing data
(X_train, X_test, Y_train, Y_test) = train_test_split(data, speeds, test_size=0.15, random_state=seed)
X_train = X_train.reshape(X_train.shape).astype('float32')
X_test = X_test.reshape(X_test.shape).astype('float32')

# Create prediction model
model = Sequential()
model.add(Convolution2D(30, 5, 5, border_mode='valid', input_shape=(480, 640, 2), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Convolution2D(60, 5, 5, border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(512, init='uniform', activation="relu"))
model.add(BatchNormalization())
model.add(Dense(128, init='uniform', activation="relu"))
model.add(BatchNormalization())
model.add(Dense(1))
model.compile(optimizer="adam", loss="mse")

print("[INFO] Training on data set...")
lrate = LearningRateScheduler(step_decay)
callbacks_list = [lrate]
model.fit(X_train, Y_train, nb_epoch=50, batch_size=64, validation_split=0.15, shuffle=True, callbacks=callbacks_list, verbose=1)
model.save('drive_predictor_model.h5')

print("[INFO] Evaluating on testing set...")
(loss, accuracy) = model.evaluate(X_test, Y_test, batch_size=64, verbose=1)
print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss, accuracy * 100))
output = model.predict(X_test)
print(output)
for i in range(0, len(output)):
    print("actual: %f, model: %f" % (Y_test[i], output[i]))
score = mean_squared_error(Y_test, output)
print("SCORE: %f" % score)

print('[INFO] Done training.')