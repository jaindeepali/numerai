from sklearn.feature_selection import VarianceThreshold, RFECV, SelectKBest, SelectPercentile, f_classif
from sklearn.decomposition import PCA

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.lda import LDA
from sklearn.qda import QDA
from nnet import *

from sklearn import cross_validation

import pandas as pd
import numpy as np
import pickle
import os

classifiers = {
	'knn': KNeighborsClassifier( 3 ),
	'svm_linear': SVC(kernel="linear", C=0.025),
	'svm': SVC(gamma=2, C=1),
	'tree': DecisionTreeClassifier(max_depth=5),
	'rf': RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
	'adb': AdaBoostClassifier(),
	'etc': ExtraTreesClassifier(),
	'gauss': GaussianNB(),
	'lda': LDA(),
	'qda': QDA(),
	'ann': neuralNetwork( 16 )
}

def main():

	# Load Data:
	train = pd.read_csv('data/train.csv')
	valid = pd.read_csv('data/valid.csv')
	test = pd.read_csv('data/test.csv')

	classifiers['ann'].fit(train.iloc[:, 0:14].as_matrix(), train['target'].as_matrix())
	pred = classifiers['ann'].predict(test.iloc[:, 0:14].as_matrix())
	pred_df = pd.DataFrame(pred)
	pred_df.to_csv('submissions/python_nnet.csv')

if __name__ == '__main__':
	main()