X_data = []
files = sorted(glob.glob ("absdiff_frames/*.jpg"), key=lambda name: int(name[15:-4]))

sorted(files, )
for file in files:
    print(file)
    img = cv2.imread (file)
    img = skimage.color.rgb2gray(img)
    img = skimage.transform.resize(img,(28,28))
    img = skimage.exposure.rescale_intensity(img, out_range=(0, 255))
    X_data.append (img)

print('X_data shape:', np.array(X_data).shape)
images = np.stack(X_data)

files = sorted(glob.glob ("bg_frames/*.jpg"), key=lambda name: int(name[15:-4]))
sorted(files, )
for file in files:
    print(file)
    img = cv2.imread(file,0)
    frames.append(img)

print(frames)


for i in range(1, len(frames)):
	img0, img1 = frames[i-1], frames[i]
	speed = np.linalg.norm(img1-img0)
	flow_matrix = cv2.calcOpticalFlowFarneback(img0, img1, 0.5, 3, 15, 2, 5, 1.2, 0)
	features = image_to_feature_vector(flow_matrix)
	speed = dataset[1][i]
	data.append(features)
	speeds.append(speed)

	if frame_idx > 0 and frame_idx % 500 == 0:
		print("[%d] processed." % frame_idx)

	frame_idx += 1
	prev_gray = frame_gray


# clf = KerasRegressor(build_fn=baseline_model, verbose=1)
# clf.fit(X_train,Y_train)
# res = clf.predict(X_test)
# print(res)
# score = mean_squared_error(Y_test, res)
# print("SCORE: %d" % score)


    print("[INFO] K-fold evaluation on test set...")
    estimators = []
    estimators.append(('standardize', StandardScaler()))
    estimators.append(('mlp', KerasRegressor(build_fn=baseline_model, nb_epoch=50, batch_size=128, verbose=0)))
    pipeline = Pipeline(estimators)
    kfold = KFold(n_splits=10, random_state=seed)
    results = cross_val_score(pipeline, trainData, trainSpeeds, cv=kfold)
    print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))



    # def train_and_evaluate__model(model, data[train], speeds[train], data[test], speeds[test):
    #     model.fit(data[train], speeds[train], nb_epoch=50, batch_size=128, verbose=1)

    # skf = StratifiedKFold(labels, n_folds=10, shuffle=True)
    # for i, (train, test) in enumerate(skf):
    # 	print "Running Fold", i+1, "/", 10
    # 	model = None # Clearing the NN.
    # 	model = baseline_model()
    # 	train_and_evaluate_model(model, data[train], speeds[train], data[test], speeds[test))
    #
    # print("[INFO] K-fold evaluation on test set...")
    # kfold = KFold(n_splits=10, random_state=seed)
    # results = cross_val_score(estimator, trainData, trainSpeeds, cv=kfold)
    # print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))
    #
    # print("[INFO] evaluating on testing set...")
    # (loss, accuracy) = model.evaluate(testData, testSpeeds,
    # 	batch_size=128, verbose=1)
    # print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,
    # 	accuracy * 100))
