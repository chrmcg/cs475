#!/usr/bin/python

# Neil Mallinar & Charlie McGeorge
# CS 475 Machine Learning -- Fall 2014

# This script takes a path to a directory of .pkl files containing Python dictionaries,
# each of which contains the vector values of an audio feature extracted using YAAFE,
# taken over all available instances (which were instrument recordings as WAVs)

# This script lets the user interactively choose which features, and which classifiers,
# should be used to classify the instances.


import sys
import os
import Queue
import math
import pickle

import numpy as np
from sklearn import svm, linear_model
from sklearn.neighbors import KNeighborsClassifier

import process_data
from accuracy import Accuracy


class FeatureSelector:
    def __init__(self, argv):
        if len(argv) != 2:
            print 'Usage: ./feature_select.py path_to_pkl_dir'
            exit()
        path = argv[1];
        if not os.path.isdir(path):
            print 'Directory not found.'
            exit()
        filenames = os.listdir(path)
        filenames = [i for i in filenames if i[-4:] == '.pkl']
        if len(filenames) == 0:
            print 'No PKL files found in directory.';
            exit()

        print filenames

        self.train = {}
        self.train['target'] = []
        self.train['data'] = []
        self.dev = {}
        self.dev['target'] = []
        self.dev['data'] = []

        # TODO: interactively choose a subset of these filenames
        for fn in filenames:
            f = open(os.path.join(path, fn), 'rb')
            pkl_dict = pickle.load(f)
            f.close()

            # Handle the training data
            for instance in range(0, len(pkl_dict.train['data'])):

                # Append these feature values to the correct instance
                if len(self.train['data']) <= instance:
                    self.train['data'].append([])
                self.train['data'][instance] += pkl_dict.train['data'][instance]

                # Make sure the labels agree
                if len(self.train['target']) <= instance:
                    self.train['target'].append(pkl_dict.train['target'][instance])
                else:
                    if self.train['target'][instance] != pkl_dict.train['target'][instance]:
                        print "Error: labels do not agree"
                        exit()


            # Handle the development data
            for instance in range(0, len(pkl_dict.dev['data'])):

                # Append these feature values to the correct instance
                if len(self.dev['data']) <= instance:
                    self.dev['data'].append([])
                self.dev['data'][instance] += pkl_dict.dev['data'][instance]

                # Make sure the labels agree
                if len(self.dev['target']) <= instance:
                    self.dev['target'].append(pkl_dict.dev['target'][instance])
                else:
                    if self.dev['target'][instance] != pkl_dict.dev['target'][instance]:
                        print "Error: labels do not agree"
                        exit()


        # TODO: choose classifier(s) interactively
        lin_clf = svm.LinearSVC()
        lin_clf.fit(np.array(self.train['data']), np.array(self.train['target']))
        svm_predictions = lin_clf.predict(np.array(self.dev['data']))

        actual = np.array(self.dev['target'])

        print "SVM:"
        svm_evaluator = Accuracy(svm_predictions, actual)

def main():
    fs = FeatureSelector(sys.argv)

if __name__ == "__main__":
    main()
