#!/usr/bin/python

# Neil Mallinar & Charlie McGeorge
# CS 475 Machine Learning -- Fall 2014

# This script takes (1) a label and (2) a path to a directory of
# CSV files output from the YAAFE feature extraction library
# and returns a single feature vector containing all extracted values

import sys
import os
import Queue
import math

class Vectorize:
    def __init__(self, argv):
        # Unpack program arguments and check correctness
        if len(argv) != 3:
            print 'Wrong number of arguments. Run as: ./vectorize_csv.py LABEL_NAME PATH_TO_CSV_DIRECTORY'
            exit()
        label = argv[1]
        path = argv[2]
        if not os.path.isdir(path):
            print 'Directory not found. Run as ./vectorize_csv.py LABEL_NAME PATH_TO_CSV_DIRECTORY'
            exit()
        filenames = os.listdir(path)
        filenames = [i for i in filenames if i[-4:] == '.csv']
        if len(filenames) == 0:
            print 'No CSV files found in directory.'
            exit()

        # Set up temporary data structure
        self.windows = []

        # We want all the time-dependent features contiguous,
        # so keep a queue of non-time-dependent features
        # and add those last.
        self.queue = Queue.Queue()

        # Read CSV files into the data structure
        y = 0
        for fn in filenames:
            filepath = os.path.join(path, fn)
            f = open(filepath, "r")
            csv_data = [line.strip().split(',') for line in f if len(line.strip()) > 0]
            f.close()
            if len(csv_data) > 10:
                self.add_feature(csv_data)
            else:
                self.queue.put(csv_data)

        # Concatenate the time-window vectors into one huge vector.
        # The vector's first entry should be the label.
        self.vec = []
        self.vec.append(label)
        for window_data in self.windows:
            self.vec += window_data

        # Now append the features that are not dependent on time-windows.
        while not self.queue.empty():
            csv_data = self.queue.get()
            for data_row in csv_data:
                for value in data_row:
                    if math.isnan(float(value)):
                        self.vec.append(0.0)
                    else:
                        self.vec.append(float(value))


    def get_vector(self):
        return self.vec

    def add_feature(self, csv_data):

        # Assuming csv_data represents a single extracted feature f_i
        # whose value at each time-window w_j is separated by newlines
        # and each line consists of comma-separated coefficients c_k,
        # we want to represent the final vector as
        # [ f1w1c1,f1w1c2...f1w1cK, f2w1c1...f2w1cK, ... fIw1c1...fIw1cK,
        #   f1w2c1,f1w2c2...f1w2cK, f2w2c1...f2w2cK, ... fIw2c1...fIw2cK,
        #   ...
        #   f1wJc1,f1wJc2...f1wJcK, f2wJc1...f2wJcK, ... fIwJc1...fIwJcK ]

        index = 0
        for window_data in csv_data:
            if len(self.windows) <= index:
                self.windows.append([])

            for coefficient in window_data:
                if math.isnan(float(coefficient)):
                    self.windows[index].append(0.0)
                else:
                    self.windows[index].append(float(coefficient))
            index += 1
            pass

def main():
    vector = Vectorize(sys.argv)
    print vector.get_vector()

if __name__ == "__main__":
    main()
