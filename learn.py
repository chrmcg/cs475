import pickle
import process_data
import sys
from sklearn import svm, linear_model, metrics, cross_validation
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
  test = learner.data.test

  lin_clf = svm.LinearSVC()
  lin_clf.fit(np.array(train['data']), np.array(train['target']))
  svm_predictions = lin_clf.predict(np.array(dev['data']))

  sgd_clf = linear_model.SGDClassifier(n_iter=200, shuffle=True)
  sgd_clf.fit(np.array(train['data']), np.array(train['target']))
  sgd_predictions = sgd_clf.predict(np.array(dev['data']))

  log_reg = linear_model.LogisticRegression(C=1e5)
  log_reg.fit(np.array(train['data']), np.array(train['target']))
  log_predictions = log_reg.predict(np.array(dev['data']))

  knn = KNeighborsClassifier()
  knn.fit(np.array(train['data']), np.array(train['target']))
  knn_predictions = knn.predict(np.array(dev['data']))


  # Run all of the above tests on a cross-validated data set

  X = np.concatenate([train['data'], dev['data'], test['data']])
  Y = np.concatenate([train['target'], dev['target'], test['target']])
  X_train, X_dev, Y_train, Y_dev = cross_validation.train_test_split(X, Y, test_size=0.2, random_state=10)
  
  sv = svm.LinearSVC()
  sv.fit(X_train, Y_train)
  sv_predictions = sv.predict(X_dev)

  log_reg_cross_valid = linear_model.LogisticRegression(C=1e5)
  log_reg_cross_valid.fit(X_train, Y_train)
  log_predictions_cross_valid = log_reg_cross_valid.predict(X_dev)

  knn_cross_valid = KNeighborsClassifier()
  knn_cross_valid.fit(X_train, Y_train)
  knn_predictions_cross_valid = knn_cross_valid.predict(X_dev)

  actual = np.array(dev['target'])
  actual_cross_valid = np.array(Y_dev)

  # Metrics section
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

  labelDictionary2 = {
    "acousticbass": 1,
    "acousticguitar": 2,
    "acousticpiano": 3,
    "electricbass": 1,
    "electricguitar": 2,
    "electricpiano": 3,
    "solohorns": 4,
    "solostrings": 5,
    "strings": 5
  }

  print("SVM 1 in K:\n%s\n" % (metrics.classification_report(actual, svm_predictions)))
  print("Predictions:\n" + str(svm_predictions))
  print("Actual:\n" + str(actual))
  print(labelDictionary)

  print("Logistic Regression:\n%s\n" % (metrics.classification_report(actual, log_predictions)))
  print("Predictions:\n" + str(log_predictions))
  print("Actual:\n" + str(actual))
  print(labelDictionary)

  print("kNN:\n%s\n" % (metrics.classification_report(actual, knn_predictions)))
  print("Predictions:\n" + str(knn_predictions))
  print("Actual:\n" + str(actual))
  print(labelDictionary)

  print("SGD:\n%s\n" % (metrics.classification_report(actual, sgd_predictions)))
  print("Predictions:\n" + str(sgd_predictions))
  print("Actual:\n" + str(actual))
  print(labelDictionary)

  print("SVM 1 in K Cross-Validated:\n%s\n" % (metrics.classification_report(actual_cross_valid, sv_predictions)))
  print("Predictions:\n" + str(sv_predictions))
  print("Actual:\n" + str(actual_cross_valid))
  print(labelDictionary)

  print("Logistic Regression Cross-Validated:\n%s\n" % (metrics.classification_report(actual_cross_valid, log_predictions_cross_valid)))
  print("Predictions:\n" + str(log_predictions_cross_valid))
  print("Actual:\n" + str(actual_cross_valid))
  print(labelDictionary)

  print("kNN Cross-Validated:\n%s\n" % (metrics.classification_report(actual_cross_valid, knn_predictions_cross_valid)))
  print("Predictions:\n" + str(knn_predictions_cross_valid))
  print("Actual:\n" + str(actual_cross_valid))
  print(labelDictionary)

  #print labelDictionary

  #print "SVM (1-in-K encoded): "
  #svm_evaluator = Accuracy(svm_predictions, actual)
  #print "KNN: "
  #knn_evaluator = Accuracy(knn_predictions, actual)
  #print "SVM (1-in-1 encoded): "
  #svm2_evaluator = Accuracy(svm2_predictions, actual)
  #print "Logistic Regression: "
  #log_reg = Accuracy(log_predictions, actual)

if __name__ == "__main__":
  main()