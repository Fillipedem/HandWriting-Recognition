"""
Testing MLP Models
"""
from sklearn.externals import joblib
from mnistdata import test_images, test_labels

# reading model
clf = joblib.load(r'.\models\firstclf.pkl')

# predict
predict_images = clf.predict(test_images)

# results
miss_classified = []
for idx, pred in enumerate(predict_images):

    if pred != test_labels[idx]:

        miss_classified.append([test_images[idx], test_labels[idx], predict_images[idx]])

total = len(test_images)
miss = len(miss_classified)

print("Acuracia: ", (total - miss)/total)
