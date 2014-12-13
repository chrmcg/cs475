import vectorize_csv
import sys
import os
import subprocess
import shutil

# argv is as follows:
# training directory, dev directory test directory, featureplan
class Data:
  def __init__(self, argv):
    if len(argv) != 5:
      print "Wrong number of arguments. Correct is: ./process_data.py train_dir dev_dir test_dir feature_plan_full_path"
      exit()

    train_dir = argv[1]
    dev_dir = argv[2]
    test_dir = argv[3]

    feature_plan = argv[4]

    if not os.path.isdir(train_dir):
      print 'Training directory not found'
      exit()
    if not os.path.isdir(dev_dir):
      print 'Dev directory not found'
      exit()
    if not os.path.isdir(test_dir):
      print 'Test directory not found'
      exit()
    if not os.path.exists('temp'):
      os.makedirs('temp')

    training_filenames = os.listdir(train_dir)
    self.training_filenames = [i for i in training_filenames if i[-4:] == '.wav']
    
    dev_filenames = os.listdir(dev_dir)
    self.dev_filenames = [i for i in dev_filenames if i[-4:] == '.wav']
    
    test_filenames = os.listdir(test_dir)
    self.test_filenames = [i for i in test_filenames if i[-4:] == '.wav']

    self.train = {}
    self.train['target'] = []
    self.train['data'] = []
    self.dev = {}
    self.dev['target'] = []
    self.dev['data'] = []
    self.test = {}
    self.test['target'] = []
    self.test['data'] = []

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

    for instance in self.training_filenames:
      instanceLabel = instance[:instance.index('_')]
      self.train['target'].append(labelDictionary[instanceLabel])
      subprocess.call(["yaafe.py", "-c", feature_plan, "-r", "44100", train_dir+"/"+instance, "-o", "csv", "-b", "temp", "-p", "Precision=8", "-p", "Metadata=False"])
      vector = vectorize_csv.Vectorize(['blank', instanceLabel, 'temp/data/train'])
      vector = vector.get_vector()
      vector = vector[1:]
      self.train['data'].append(vector)
      shutil.rmtree('temp/data/train')

    for instance in self.dev_filenames:
      instanceLabel = instance[:instance.index('_')]
      self.dev['target'].append(labelDictionary[instanceLabel])
      subprocess.call(["yaafe.py", "-c", feature_plan, "-r", "44100", dev_dir+"/"+instance, "-o", "csv", "-b", "temp", "-p", "Precision=8", "-p", "Metadata=False"])
      vector = vectorize_csv.Vectorize(['blank', instanceLabel, 'temp/data/dev'])
      vector = vector.get_vector()
      vector = vector[1:]
      self.dev['data'].append(vector)
      shutil.rmtree('temp/data/dev')

    for instance in self.test_filenames:
      instanceLabel = instance[:instance.index('_')]
      self.test['target'].append(labelDictionary[instanceLabel])
      subprocess.call(["yaafe.py", "-c", feature_plan, "-r", "44100", test_dir+"/"+instance, "-o", "csv", "-b", "temp", "-p", "Precision=8", "-p", "Metadata=False"])
      vector = vectorize_csv.Vectorize(['blank', instanceLabel, 'temp/data/test'])
      vector = vector.get_vector()
      vector = vector[1:]
      self.test['data'].append(vector)
      shutil.rmtree('temp/data/test')

def main():
  data = Data(sys.argv)
  pickle.dump(data, 'processed_data.pkl')

if __name__ == "__main__":
  main();