#!/usr/bin/env python3

"""
This module implements a simplified training process for boston housing price data.
"""


from pathlib import Path
from copy import deepcopy

import pickle as pickle
import matplotlib.pyplot as plt
import numpy as np


### SETTINGS #############################################################################
"""

The following global constants are the settings for training:

:ALPHA: learning rate
:BATCH_SIZE: Number of samples fed into model per iter
:MAX_ITERS: Number of learning iterations we train the model on
:DECAY: The 'weight decay' (??)

"""

ALPHA = 0.001
BATCH_SIZE = 10
MAX_ITERS = 100
DECAY = 0.0


### TRAINING #############################################################################

def predict(X, w, y):
    """

    Make a prediction on the test data given the matrix of features and vector of weights.
    :param X: A matrix of sample features ( Dim: M x (d+1) )
    :param w: A vector of weights ( Dim: (d+1) x 1 )
    :param y: A vector of targets ( Dim: M x 1 )

    :returns (y_hat, loss, risk):
    - y_hat >>  The new prediction, y_hat
    - loss >> The MSE result for the prediction
    - risk >> The risk of the prediction

    """

    M = len(y)

    y_hat = np.dot(X, w)
    loss = ( 1 / (2 * M) ) * np.sum((y_hat - y) ** 2)
    risk = (1/M) * np.sum(np.abs(y_hat - y))
    return y_hat, loss, risk

def train(X_train, y_train, X_val, y_val):

    #### INITIALIZATION ####

    n_train = X_train.shape[0]

    # w: (d+1)x1
    w = np.zeros([X_train.shape[1], 1])

    losses_train = []
    risks_val = []
    best_validation_epoch = float('inf')

    w_best = None
    risk_best = float('inf')
    loss_best = float('inf')

    for epoch in range(MAX_ITERS):

        loss_this_epoch = 0
        M = 1
        for b in range(int(np.ceil(n_train/BATCH_SIZE))):

            X_batch = X_train[b*BATCH_SIZE : (b+1)*BATCH_SIZE]
            y_batch = y_train[b*BATCH_SIZE : (b+1)*BATCH_SIZE]

            y_hat_batch, loss_batch, _ = predict(X_batch, w, y_batch)
            loss_this_epoch += loss_batch

            #### Mini-Batch Gradient Descent ####

            M = len(y_batch)

            # Diff prediction and target parts and divide by M to get gradient of dJ/dw
            gradient = (1/M)*np.dot(X_batch.T, y_hat_batch - y_batch)

            # Then we finally merge this all together to update w
            w = w - ALPHA * gradient

        #### Monitor Model Behavior After Each Epoch ####
        # 1. Compute the training loss by averaging loss_this_epoch
        # 2. Perform validation on the validation set by the risk
        # 3. Keep track of the best validation epoch, risk, and the weights

        # Keep track of the best validation epoch and (1.)
        curr_loss = loss_this_epoch / int(np.ceil(n_train/BATCH_SIZE))
        loss_best = min(loss_best, curr_loss)

        # Keep track of the best risk and (2.)
        _, val_loss, val_risk = predict(X_val, w, y_val)
        if (risk_best >= val_risk):
            # Keep track of the best weights and associated best risk
            risk_best = val_risk
            w_best = deepcopy(w)
            best_validation_epoch = epoch

        # Monitor model behaviour after each epoch
        risks_val.append(val_risk)
        losses_train.append(curr_loss)

    return (best_validation_epoch, val_risk, w_best, losses_train, risks_val) 

### MAIN #################################################################################

if __name__ == "__main__":

    #### LOAD DATA ####

    pkl_path = Path.cwd() / "src" / "public" / "housing.pkl"
    X = None
    y = None
    try:
        with open(pkl_path, "rb") as f:
            (X, y) = pickle.load(f)
    except (OSError):
        print(f"File {pkl_path} cannot be discovered. Have you entered the correct path?")
        raise OSError

    assert X is not None and y is not None, f"Failed to initialize X, y"


    #### NORMALIZATION + FEATURE AUGMENTATION ####

    # X: sample x dimension
    # y: sample x 1
    mean_X = np.mean(X, axis=0)
    std_X = np.std(X, axis=0)
    X = (X - mean_X) / std_X

    # normalize features:
    mean_y = np.mean(y)
    std_y = np.std(y)
    y = (y - mean_y) / std_y

    # X_: Nsample x (d+1)
    X_ = np.concatenate((np.ones([X.shape[0], 1]), X), axis=1)


    #### SAMPLE FROM TRAINING SET ####

    # Randomly shuffle the data
    np.random.seed(314)
    np.random.shuffle(X_)
    np.random.seed(314)
    np.random.shuffle(y)

    X_train = X_[:300]
    y_train = y[:300]

    X_val = X_[300:400]
    y_val = y[300:400]

    X_test = X_[400:]
    y_test = y[400:]


    #### TRAINING ####

    #### Perform test using the weights yielding the best test performance ####

    (
        best_val_epoch,
        best_risk,
        w_best,
        training_loss,
        validation_risk
    ) = train(X_train, y_train, X_val, y_val)

    prediction, loss, risk = predict(X_test, w_best, y_test)


    #### Report numbers ####

    print("### DATA STATISTICS #########################################################")
    print("Target mean", mean_y)
    print("Target std", std_y)
    print("")

    print("### TESTING PERFORMANCE #####################################################")
    print("Best validation epoch: ", best_val_epoch)
    print("Best validation risk: ", best_risk)
    print("Test Loss: ", loss)
    print("Test Risk: ", risk)
    print("")

    #### Draw plots ####

    fig, (axis_1, axis_2) = plt.subplots(1, 2)

    # Plot training loss on the first subplot
    axis_1.plot(range(len(training_loss)), training_loss, label='Training Loss', color='blue', marker='o')
    axis_1.set_title('Training Loss')
    axis_1.set_ylabel('Training Loss')
    axis_1.set_xlabel('Epochs')

    # Plot validation risk on the second subplot
    axis_2.plot(range(len(validation_risk)), validation_risk, label='Validation Risk', color='red', marker='x')
    axis_2.set_title('Validation Risk')
    axis_2.set_ylabel('Validation Risk')
    axis_2.set_xlabel('Epochs')

    plt.savefig('output/ca2pa2.png')  # Save the plot as an image

    pass
