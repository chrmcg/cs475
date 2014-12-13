class Accuracy:
  def __init__(self, predictions, actual):
    if len(predictions) != len(actual):
      print "Lengths of two label vectors not equal!"
      exit()

    print "Predictions:\n", predictions
    print "Actual:\n", actual
    numCorrect = 0
    numTotal = len(predictions)
    numIncorrect = 0
    for i in range (0,len(predictions)):
      if (predictions[i] == actual[i]):
        numCorrect = numCorrect + 1
      else:
        numIncorrect = numIncorrect + 1

    print "Correct: ", (100*numCorrect/float(numTotal)), "%"
    print "Incorrect: ", (100*numIncorrect/float(numTotal)), "%"