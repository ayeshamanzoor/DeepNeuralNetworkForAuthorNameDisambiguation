Python 3.7.6 (tags/v3.7.6:43364a7ae0, Dec 19 2019, 00:42:30) [MSC v.1916 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>> 
=============== RESTART: E:\Allah Kareem HelpmeGuid me\pikears.py ==============
Using TensorFlow backend.



3
WARNING:tensorflow:From E:\Allah Kareem HelpmeGuid me\lib\site-packages\tensorflow_core\python\ops\resource_variable_ops.py:1630: calling BaseResourceVariable.__init__ (from tensorflow.python.ops.resource_variable_ops) with constraint is deprecated and will be removed in a future version.
Instructions for updating:
If using Keras pass *_constraint arguments to layers.
WARNING:tensorflow:From E:\Allah Kareem HelpmeGuid me\lib\site-packages\tensorflow_core\python\ops\nn_impl.py:183: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_1 (Dense)              (None, 4)                 16        
_________________________________________________________________
dense_2 (Dense)              (None, 2)                 10        
=================================================================
Total params: 26
Trainable params: 26
Non-trainable params: 0
_________________________________________________________________
WARNING:tensorflow:From E:\Allah Kareem HelpmeGuid me\lib\site-packages\keras\backend\tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Training Accuracy: 0.7698
Testing Accuracy:  0.7619


f1 score0.43750000000000006
Micro-average quality numbers
Precision: 0.7778, Recall: 0.7500, F1-measure: 0.7636
Accuracy of evaluation: 75.0
Accuracy of prediction: 75.0
The End
Train and test curve
roc of class==
0
======
roc of 0.5714285714285714
roc of class==
1
======
roc of 0.5

roc of micro 0.7678571428571428
Traceback (most recent call last):
  File "E:\Allah Kareem HelpmeGuid me\pikears.py", line 130, in <module>
    label='ROC curve (area = %0.2f)' % roc_auc[2])
KeyError: 2
>>> 
=============== RESTART: E:\Allah Kareem HelpmeGuid me\pikears.py ==============
Using TensorFlow backend.



3
WARNING:tensorflow:From E:\Allah Kareem HelpmeGuid me\lib\site-packages\tensorflow_core\python\ops\resource_variable_ops.py:1630: calling BaseResourceVariable.__init__ (from tensorflow.python.ops.resource_variable_ops) with constraint is deprecated and will be removed in a future version.
Instructions for updating:
If using Keras pass *_constraint arguments to layers.
WARNING:tensorflow:From E:\Allah Kareem HelpmeGuid me\lib\site-packages\tensorflow_core\python\ops\nn_impl.py:183: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_1 (Dense)              (None, 4)                 16        
_________________________________________________________________
dense_2 (Dense)              (None, 2)                 10        
=================================================================
Total params: 26
Trainable params: 26
Non-trainable params: 0
_________________________________________________________________
WARNING:tensorflow:From E:\Allah Kareem HelpmeGuid me\lib\site-packages\keras\backend\tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.

Training Accuracy: 0.7460
Testing Accuracy:  0.7619


f1 score0.42857142857142855
Micro-average quality numbers
Precision: 0.7500, Recall: 0.7500, F1-measure: 0.7500
Accuracy of evaluation: 75.0
Accuracy of prediction: 75.0
The End
Train and test curve
roc of class==
0
======
roc of 0.5
roc of class==
1
======
roc of 0.5

roc of micro 0.75
Traceback (most recent call last):
  File "E:\Allah Kareem HelpmeGuid me\pikears.py", line 170, in <module>
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
NameError: name 'lw' is not defined
>>> 
