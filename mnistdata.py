# load database
import os
from mnist import MNIST

# loading training and test data
mndata = MNIST('.\data')
training_images, training_labels = mndata.load_training()

test_images, test_labels = mndata.load_testing()
