"""
Classiicando com MLP
"""
from sklearn.externals import joblib  # sabe model
from sklearn.neural_network import MLPClassifier

from mnistdata import training_images, training_labels

# Classifier Parameters
params = {'solver': 'lbfgs',
         'alpha': 1e-5,
         'hidden_layer_sizes': (10),
         'random_state': 1,
         'activation': 'logistic'}

# Classifier
clf = MLPClassifier(**params)

clf.fit(training_images, training_labels)

joblib.dump(clf, r'.\models\firstclf.pkl')
