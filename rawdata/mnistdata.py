# load database
from mnist import MNIST
from sklearn.externals import joblib  # save dataset

from rawdata.dataset import DataSet

# loading training and test data
mndata = MNIST()
training_images, training_labels = mndata.load_training()

test_images, test_labels = mndata.load_testing()

# Saving DataSet
dataset = DataSet(training_images, training_labels, test_images, test_labels, "Dados Iniciais")

joblib.dump(dataset, r"..\datasets\first_dataset.ds")
