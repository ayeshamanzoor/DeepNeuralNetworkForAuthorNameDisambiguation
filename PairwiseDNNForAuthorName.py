from keras.preprocessing.text import Tokenizer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from keras.models import Sequential
import matplotlib.pyplot as plt
from keras import layers
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn import metrics
from sklearn.metrics import roc_auc_score,roc_curve, auc
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from scipy import interp
from itertools import cycle
def one_hot_encode(labels):
    n_labels=len(labels)
    n_unique_labels=len(np.unique(labels))
    one_hot_encode=np.zeros((n_labels,n_unique_labels))
    one_hot_encode[np.arange(n_labels),labels]=1
    return  one_hot_encode
url_2 = "E:/Allah Kareem HelpmeGuid me/DatasetForPairwiseclassification/Romano/Romaanopairwise.csv"

df1= pd.read_csv(url_2,header=0)

array1=df1.values
target1=df1[df1.columns[5]]
array1=df1[df1.columns[0:5]].values
encoder=LabelEncoder()
encoder.fit(target1)
target1= encoder.transform(target1)
target1= one_hot_encode(target1)

url_1 = "E:/Allah Kareem HelpmeGuid me/DatasetForPairwiseclassification/Romano/Romaanopairwise.csv"

df= pd.read_csv(url_1,header=0)
array=df.values
target=df[df.columns[5]]
array=df[df.columns[0:5]].values
encoder=LabelEncoder()
encoder.fit(target)
target= encoder.transform(target)
target= one_hot_encode(target)
print("target",target)
n_class=6
array_train,array_test,target_train,target_test= train_test_split(array1, target1, test_size=0.25, random_state=1000)

print(array_train)
print(target_train)
#array_train, array_test, target_train, target_test = train_test_split(array, target, test_size=0.25, random_state=1000)
#print("test data")
#print(target_test)
input_dim = array_train.shape[1]
print("input dimention",input_dim)
model = Sequential()
model.add(layers.Dense(6,input_dim=input_dim, activation='relu'))
model.add(layers.Dense(n_class, activation='sigmoid'))
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.summary()
history = model.fit(array_train, target_train,
                    epochs=250,
                    verbose=False,
                    validation_data=(array_test, target_test),
                    batch_size=108)
loss, accuracy = model.evaluate(array_train, target_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss_eva, accuracy = model.evaluate(array_test, target_test, verbose=2)
print("Testing Accuracy:  {:.4f}".format(accuracy))
test_predictions = model.predict(array)
test_predictions = np.round(test_predictions)
print("jjjjjjjjjjjjj",test_predictions)
print(target)
# Report the accuracy
accuracy = accuracy_score(target, test_predictions)
#print(cross_val_score(y_test, test_predictions,cv=5)) 
print("f1 score"+str(f1_score(target, test_predictions, average='macro')))
precision1 = precision_score(target, test_predictions, average='macro')
print(precision1,"macro precision")
precision = precision_score(target, test_predictions, average='micro')
recall = recall_score(target, test_predictions, average='micro')
f1 = f1_score(target, test_predictions, average='micro')
print("Micro-average quality numbers")
print("Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}".format(precision, recall, f1))
print("Accuracy of evaluation: " + str(accuracy*100))
print("Accuracy of prediction: " + str(accuracy*100))
print("The End")

print("Train and test curve")

def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
plot_history(history)
plt.show()

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_class):
    fpr[i], tpr[i], _ =metrics. roc_curve(target[:, i], test_predictions[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    print("roc of class==")
    print(i)
    print("======")
    print("roc of",roc_auc[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = metrics.roc_curve(target.ravel(), test_predictions.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
print("")
print("roc of micro",roc_auc["micro"])
plt.figure()
#lw = 1
plt.plot(fpr["micro"], tpr["micro"], color='darkorange',
          label='ROC curve (area = %0.2f)' % roc_auc["micro"])
plt.plot([0, 1], [0, 1], color='navy',  linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic (ROC)curve')
plt.legend(loc="lower right")
plt.show()
precision, recall, _ = precision_recall_curve(target.ravel(), test_predictions.ravel())

plt.figure(figsize = (10,8))
plt.plot([0, 1], [0.5, 0.5],label=' precision-recall AUC (area = %0.2f)' % auc(recall, precision))
plt.plot(recall, precision)
plt.xlabel('recall')
plt.ylabel('precision')
plt.title('precision and recall curve')
plt.legend(loc="lower right")
plt.show()
auc_prc = auc(recall, precision)
print("value of precision and recall curve",auc_prc)
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_class)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_class):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
print(target.ravel())
print(target)
print(confusion_matrix(target.ravel(), test_predictions.ravel()))
pd.crosstab(target.ravel(), test_predictions.ravel(), rownames = ['Actual'], colnames =['Predicted'], margins = True)
