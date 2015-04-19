from sklearn import svm, grid_search
from sklearn import cross_validation
from sklearn.svm import SVC
import pandas as pd
import numpy as np
import os
import cPickle
from config import *

data = pd.read_pickle(os.path.join(data_path, 'feature.pickle'))
data = data[data['feature'].map(lambda x: len(x) > 4000)]
X = np.array(data['feature'].tolist())
y = data['class'].values
images = data['filename'].values

param_grid = [
  {'C': [0.01, 0.1, 1, 10, 100, 1000], 'kernel': ['linear']},
  {'C': [0.01, 0.1, 1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
 ]

GRIDSEARCH=False
if GRIDSEARCH==True:
    clf = grid_search.GridSearchCV(svm.SVC(), param_grid, verbose=10)
    clf.fit(X, y)

    with open('best_estimator', 'wb') as f:
        cPickle.dump(clf.best_estimator_, f)

else:
    cv = cross_validation.ShuffleSplit(len(y), n_iter=1, test_size=0.2)
    for train, test in cv:
        train_X = X[train]
        train_y = y[train]

        test_X = X[test]
        test_y = y[test]
        test_image = images[test]

        clf = SVC(C=10, gamma=0.0001, kernel='rbf', random_state=None)
        clf.fit(train_X, train_y)

    result = clf.predict(test_X)
    cats = test_image[result == 0]
    dogs = test_image[result == 1]

    result_df = pd.DataFrame({'image':test_image})
    result_df['class'] = result
    result_df.to_pickle('./data/result.pickle')







