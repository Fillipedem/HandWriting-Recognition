"""
Class for DataSet
"""


class DataSet:

    def __init__(self, training_data=[], training_labels=[], test_data=[], test_labels=[], description=""):

        self.training_data = training_data
        self.training_labels = training_labels
        self.test_data = test_data
        self.test_labels = test_labels
        self.description = description
