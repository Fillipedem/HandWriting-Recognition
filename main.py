# Classifier Parameters
import os
import datetime
import errno
import sys

from rawdata import dataset

from sklearn.externals import joblib
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, confusion_matrix
from sklearn.neural_network import MLPClassifier



# Variaveis
dataset_name = "first_dataset"
dir_name = str(datetime.datetime.now())
dir_name = dir_name.replace(":", "")
dir_name = dir_name.replace(".", "")

dir_path = r"./models/" + dir_name + "/"
results_file = dir_path + r"results.txt"

# create dir
if not os.path.exists(os.path.dirname(dir_path)):
    try:
        os.makedirs(os.path.dirname(dir_path))
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise

# Load DataSet
print("Loading dataset: ", dataset_name)
dataset = joblib.load('./datasets/' + dataset_name + ".ds")

#
## Training
#
print("Training mlp")

# Parametros de Treinamento
params = {'solver': 'lbfgs',
          'alpha': 1e-5,
          'hidden_layer_sizes': (100, 50),
          'random_state': 1,
          'activation': 'logistic',
          'early_stopping': True,
          }

# Classifier
clf = MLPClassifier(**params)

clf.fit(dataset.training_data, dataset.training_labels)

# save clf
joblib.dump(clf, dir_path + "/clf.pkl")

#
## Testing
#
print("Testing")

# predict
predict_data = clf.predict(dataset.test_data)

test_labels = list(dataset.test_labels)

report = classification_report(test_labels, predict_data)
accuracy = accuracy_score(test_labels, predict_data)
#roc = roc_auc_score(test_labels, predict_data)
cmatrix = confusion_matrix(test_labels, predict_data)

# writing results
sys.stdout = open(results_file, "w")

print("Dataset: ", dataset_name)
print("Classifier Parameters: " + str(params))
print("\n\n")
print("Results: ")
print("Accuracy: ", accuracy)
#print("ROC - Area Under Curve: ", roc)
print("Confusion Matrix:\n", cmatrix)
print("Report: \n", report)

sys.stdout.close()


