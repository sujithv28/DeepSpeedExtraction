from __future__ import print_function
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential, load_model
from keras.wrappers.scikit_learn import KerasRegressor
from keras.layers.core import Activation, Dropout, Dense
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
from keras.utils import np_utils
from skimage import transform, color, exposure
from skimage.transform import rotate
from skimage.viewer import ImageViewer
from imutils import paths
from common import anorm2, draw_str
from time import clock
import skimage as skimage
import numpy as np
import argparse
import pandas
import cv2
import video
import json

seed = 7
np.random.seed(seed)  # for reproducibility

class App:
    def __init__(self, video_src):
        self.cap = cv2.VideoCapture(video_src)
        self.frames = []
        self.data = []
        self.speeds = []
        self.model = load_model('model.h5')
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
<<<<<<< HEAD
            if ret and self.frame_idx<100:
=======
            if ret:
>>>>>>> a5278f4e23bf4df075e3285734e782202abb0db6
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                rescale_gray = skimage.exposure.rescale_intensity(gray,out_range=(0,255))
                blur_gray = cv2.GaussianBlur(rescale_gray, (21, 21), 0)
                self.frames.append(blur_gray)
                if (self.frame_idx > 0):
                    prev_frame, curr_frame = self.frames[self.frame_idx-1], self.frames[self.frame_idx]
                    flow_matrix = cv2.calcOpticalFlowFarneback(prev_frame, curr_frame, 0.5, 3, 15, 2, 5, 1.2, 0)
                    features = cv2.resize(flow_matrix, (64, 48)).flatten()
                    self.data.append(features)
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

    def evaluate(self):
        self.data = np.array(self.data)
        self.speeds = np.array(self.speeds)
        scaler = StandardScaler()
        self.data = scaler.fit_transform(self.data)
        print("[INFO] evaluating on data set...")
        output = self.model.predict(self.data)
        mse = mean_squared_error(self.speeds, output)
        print("[INFO] mean_squared_error: %f" % mse)
        for i in range(0, len(output)):
            print("[INFO] actual: %f, estimate: %f" % (self.speeds[i], output[i]))
        row = np.array(output)
<<<<<<< HEAD
        with open("guess2.json", 'wb') as outfile:
=======
        with open("guess.json", 'wb') as outfile:
>>>>>>> a5278f4e23bf4df075e3285734e782202abb0db6
            json.dump(row.tolist(), outfile)

def main():
    import sys
    video_src = "drive.mp4"
    app = App(video_src)
    app.get_flow()
    app.evaluate()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
