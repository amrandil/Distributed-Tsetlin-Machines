#!/usr/bin/python
# Granmo's original Binary Iris demo, from https://github.com/cair/TsetlinMachine
# (commit 5678644, 21 April 2018; later only a typo fix in a comment).
# MIT licence: see ../LICENSE.
#
# Changes (compare with BinaryIrisDemo.orig.py):
#   1. Python 2 -> 3: print statements written as print() calls, xrange -> range.
#   2. One added print line per ensemble (marked ADDED) with that ensemble's own
#      test and training accuracy; the original only prints running averages.
# Run it from a folder that contains BinaryIrisData.txt; repeat_iris_demo.py
# does this.

import numpy as np

import pyximport; pyximport.install(setup_args={
                              "include_dirs":np.get_include()},
                            reload_support=True)
import MultiClassTsetlinMachine

# Ensembles

ensemble_size = 1000

# Parameters for the Tsetlin Machine
T = 10
s = 3.0
number_of_clauses = 300
states = 100 

# Parameters of the pattern recognition problem
number_of_features = 16
number_of_classes = 3

# Training configuration
epochs = 500

# Loading of training and test data
data = np.loadtxt("BinaryIrisData.txt").astype(dtype=np.int32)

accuracy_training = np.zeros(ensemble_size)
accuracy_test = np.zeros(ensemble_size)

for ensemble in range(ensemble_size):
	print("ENSEMBLE", ensemble + 1)
	print()

	np.random.shuffle(data)

	X_training = data[:int(data.shape[0]*0.8),0:16] # Input features
	y_training = data[:int(data.shape[0]*0.8),16] # Target value

	X_test = data[int(data.shape[0]*0.8):,0:16] # Input features
	y_test = data[int(data.shape[0]*0.8):,16] # Target value

	# This is a multiclass variant of the Tsetlin Machine, capable of distinguishing between multiple classes
	tsetlin_machine = MultiClassTsetlinMachine.MultiClassTsetlinMachine(number_of_classes, number_of_clauses, number_of_features, states, s, T, boost_true_positive_feedback = 1)

	# Training of the Tsetlin Machine in batch mode. The Tsetlin Machine can also be trained online
	tsetlin_machine.fit(X_training, y_training, y_training.shape[0], epochs=epochs)

	# Some performacne statistics
	accuracy_test[ensemble] = tsetlin_machine.evaluate(X_test, y_test, y_test.shape[0])
	accuracy_training[ensemble] = tsetlin_machine.evaluate(X_training, y_training, y_training.shape[0])

	print("Ensemble accuracy: test %.6f train %.6f" % (accuracy_test[ensemble], accuracy_training[ensemble]))  # ADDED
	print("Average accuracy on test data: %.1f +/- %.1f" % (np.mean(100*accuracy_test[:ensemble+1]), 1.96*np.std(100*accuracy_test[:ensemble+1])/np.sqrt(ensemble+1)))
	print("Average accuracy on training data: %.1f +/- %.1f" % (np.mean(100*accuracy_training[:ensemble+1]), 1.96*np.std(100*accuracy_training[:ensemble+1])/np.sqrt(ensemble+1)))
	print()
