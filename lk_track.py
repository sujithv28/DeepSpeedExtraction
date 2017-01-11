#!/usr/bin/env python

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2
import video
import pandas
import json
import math
from common import anorm2, draw_str
from time import clock
import argparse
import imutils
import skimage as skimage
from skimage import transform, color, exposure
from skimage.transform import rotate
from skimage.viewer import ImageViewer
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

dataset = dict(pandas.read_json("drive.json"))

lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
feature_params = dict( maxCorners = 15,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

class App:
    def __init__(self, video_src):
        self.track_len = 10
        self.detect_interval = 5
        self.tracks = []
        self.cam = video.create_capture(video_src)
        self.frame_idx = 0
        self.frames = []
        img_rows, img_cols = 640, 480

    def run(self):
        while True:
            ret, frame = self.cam.read()
            # Rexposes, blurs, and makes frame grayscale in order to smoothen it out.
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_gray = skimage.exposure.rescale_intensity(frame_gray,out_range=(0,255))
            frame_gray = cv2.GaussianBlur(frame_gray, (21, 21), 0)
            cv2.imwrite('bg_frames/%d.jpg' % self.frame_idx, frame_gray)
            # Initial speed and initialization of frames
            speed = 0.0
            vis = frame.copy()
            self.frames.append(frame_gray)

            # Calculates the optical flow and sets the new velocity tracks along with
            # drawing them out starting from second frame onwards.
            if len(self.tracks) > 0:
                img0, img1 = self.prev_gray, frame_gray
                speed = np.linalg.norm(img1-img0)
                # flow = cv2.calcOpticalFlowFarneback(img0, img1, 0.5, 3, 15, 2, 5, 1.2, 0)
                # deltaX = flow[:][:][0]
                # deltaY = flow[:][:][1]
                # vel_y = math.pow(np.linalg.norm(deltaY), 2)
                # vel_x = math.pow(np.linalg.norm(deltaX), 2)
                # print(math.sqrt(vel_y + vel_x))
                # c=0
                # ysum = 0
                # for i in range(0, 639):
                #     for j in range(0, 479):
                #         if (abs(flow[j][i][0]) < 0.5):
                #             ysum += abs(flow[j][i][1])
                #             c+=1
                # if c!=0:
                #     print(ysum/c)

                #main_input = Input(shape=(640,480,2), dtype='float64', name='main_input')
                p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
                p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
                p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
                d = abs(p0-p0r).reshape(-1, 2).max(-1)
                good = d < 1
                new_tracks = []
                for tr, (x, y), good_flag in zip(self.tracks, p1.reshape(-1, 2), good):
                    if not good_flag:
                        continue
                    tr.append((x, y))
                    if len(tr) > self.track_len:
                        del tr[0]
                    new_tracks.append(tr)
                    cv2.circle(vis, (x, y), 2, (0, 255, 0), -1)
                self.tracks = new_tracks

                # Roughly estimate speed by looking at pixel differences between current
                # and previous frames. Process image for contour detection by applying
                # a threshold and dilation on the difference between current and previous
                # frame.
                frameDelta = cv2.absdiff(img1, img0)
                #cv2.imwrite('absdiff_frames/%d.jpg' % self.frame_idx, frame)
                #cv2.imwrite('delta_frames/%d.jpg' % self.frame_idx-1, frame)
                thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
                dilate = cv2.dilate(thresh, None, iterations=2)
                cv2.imshow('Dilate', thresh)
                cv2.moveWindow('Dilate', 800, 100)
                cv2.imwrite('dilated_frames/%d.jpg' % self.frame_idx, img1-img0)
                # Calculate speed and find contours.
                (cnts, _) = cv2.findContours(dilate.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                speed = np.linalg.norm(frameDelta)/640

                # Draw out the contours on the frame if it spans a minimum area.
                for c in cnts:
            		# if the contour is too small, ignore it
            		if (cv2.contourArea(c) < 500):
            			continue
                	# compute the bounding box for the contour, draw it on the frame,
                	(x, y, w, h) = cv2.boundingRect(c)
                	cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)

                diffy = 0
                count = 0
                if (self.frame_idx > 1):
                    for c1 in cnts:
                        (x1, y1, w1, h1) = cv2.boundingRect(c1)
                        if (cv2.contourArea(c1) < 500):
                            continue
                        for c2 in self.old_cnts:
                            if (cv2.contourArea(c2) < 500):
                                continue
                            (x2, y2, w2, h2) = cv2.boundingRect(c2)
                            if (abs(cv2.contourArea(c2) - cv2.contourArea(c1))<50):
                                diffy += pow((y2-y1),2)
                                count += 1
                    if count != 0:
                        diffy /= count
                    diffy = (diffy+self.prev_diffy)/2
                self.old_cnts = cnts
                self.prev_diffy = diffy

                # Draw out all the information per frame.
                cv2.polylines(vis, [np.int32(tr) for tr in self.tracks], False, (0,255,0))
                # draw_str(vis, (20, 20), 'track count: %d' % len(self.tracks))
                # draw_str(vis, (20, 40), 'frame: %d' % self.frame_idx)
                # draw_str(vis, (20, 60), 'speed (actual): %f' % dataset[1][self.frame_idx])
                # draw_str(vis, (20, 80), 'speed (guess): %f' % speed)
                # draw_str(vis, (20, 100), 'contours: %d' % len(cnts))
                # draw_str(vis, (20, 120), 'contour diff: %d' % diffy)

            # Prints out the new frames and tracks.
            if self.frame_idx % self.detect_interval == 0:
                mask = np.zeros_like(frame_gray)
                mask[:] = 255
                for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
                    cv2.circle(mask, (x, y), 5, 0, -1)
                p = cv2.goodFeaturesToTrack(frame_gray, mask = mask, **feature_params)
                if p is not None:
                    for x, y in np.float32(p).reshape(-1, 2):
                        self.tracks.append([(x, y)])

            # Increments frame id and saves the previous gray frame.
            # Visualizes the new video with the optical flow, contours,
            # and text information layered on.
            self.frame_idx += 1
            self.prev_gray = frame_gray
            cv2.imshow('lk_track', vis)

            ch = 0xFF & cv2.waitKey(1)
            if ch == 27:
                break

def main():
    import sys
    video_src = "drive.mp4"

    App(video_src).run()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
