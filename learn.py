import pickle
import process_data
import sys
from sklearn import svm, linear_model
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from accuracy import Accuracy

class Model:
  def __init__(self, argv):
    if len(argv) == 2:
      data_file = argv[1]
      with open(data_file, 'rb') as filein:
        self.data = pickle.load(filein)
    elif len(argv) == 5:
      with open('processed_data.pkl', 'wb') as fileout:
        self.data = process_data.Data(argv)
        pickle.dump(self.data, fileout, pickle.HIGHEST_PROTOCOL)

def main():
  learner = Model(sys.argv)
  train = learner.data.train
  trainX = train['data']
  trainY = train['target']
  dev = learner.data.dev
  devX = dev['data']
  devY = dev['target']

  lin_clf = svm.LinearSVC()
  lin_clf.fit(np.array(trainX), np.array(trainY))
  svm_predictions = lin_clf.predict(np.array(devX))

  clf = svm.SVC()
  clf.fit(np.array(trainX), np.array(trainY))
  svm2_predictions = clf.predict(np.array(devX))
  
  knn = KNeighborsClassifier()
  knn.fit(np.array(trainX), np.array(trainY))
  knn_predictions = knn.predict(np.array(devX))

  log_reg = linear_model.LogisticRegression(C=1e5)
  log_reg.fit(np.array(trainX), np.array(trainY))
  log_predictions = log_reg.predict(np.array(devX))

  actual = np.array(devY);

  labelDictionary = {
    "acousticbass": 1,
    "acousticguitar": 2,
    "acousticpiano": 3,
    "electricbass": 4,
    "electricguitar": 5,
    "electricpiano": 6,
    "solohorns": 7,
    "solostrings": 8,
    "strings": 9
  }

  print labelDictionary

  print "SVM (1-in-K encoded): "
  svm_evaluator = Accuracy(svm_predictions, actual)
  print "KNN: "
  knn_evaluator = Accuracy(knn_predictions, actual)
  print "SVM (1-in-1 encoded): "
  svm2_evaluator = Accuracy(svm2_predictions, actual)
  print "Logistic Regression: "
  log_reg = Accuracy(log_predictions, actual)

if __name__ == "__main__":
  main()