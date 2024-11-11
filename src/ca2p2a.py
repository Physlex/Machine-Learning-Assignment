#!/usr/bin/env python3

"""
This module implements a simplified training process for boston housing price data.
"""


from pathlib import Path
from copy import deepcopy

import pickle as pickle
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
    n_val = X_val.shape[0]

    # w: (d+1)x1
    w = np.zeros([X_train.shape[1], 1])

    losses_train = []
    risks_val = []
    weights = []

    w_best = None
    risk_best = float('inf')
    loss_best = float('inf')

    for _ in range(MAX_ITERS):

        loss_this_epoch = 0
        risk_this_epoch = 0
        M = 1
        for b in range(int(np.ceil(n_train/BATCH_SIZE))):

            X_batch = X_train[b*BATCH_SIZE : (b+1)*BATCH_SIZE]
            y_batch = y_train[b*BATCH_SIZE : (b+1)*BATCH_SIZE]

            y_hat_batch, loss_batch, risk_batch = predict(X_batch, w, y_batch)
            loss_this_epoch += loss_batch
            risk_this_epoch += risk_batch

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
        curr_loss = loss_this_epoch / M
        loss_best = min(loss_best, curr_loss)

        # Keep track of the best risk and (2.)
        curr_risk = risk_this_epoch
        risk_best = min(risk_best, curr_risk)
        if (risk_best == curr_risk):
            # Keep track of the best weights
            w_best = deepcopy(w)

        # Monitor model behaviour after each epoch
        risks_val.append(curr_risk)
        losses_train.append(curr_loss)
        weights.append(w)

    return (w_best, risk_best, loss_best), (risks_val, losses_train, weights)


### MAIN #################################################################################

if __name__ == "__main__":

    #### LOAD DATA ####

    pkl_path = Path.cwd() / "public" / "housing.pkl"
    X = None
    y = None
    try:
        with open(pkl_path, "rb") as f:
            (X, y) = pickle.load(f)
    except (OSError):
        print(f"File {pkl_path} cannot be discovered. Have you entered the correct path?")
        raise OSError
    finally:
        print(f"File {pkl_path} loaded successfully")

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
    
    # print(X.shape, y.shape) # It's always helpful to print the shape of a variable


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

    (bests, monitor) = train(X_train, y_train, X_val, y_val)


    #### Perform test by the weights yielding the best validation performance ####


    #### Report numbers and draw plots as required ####


    pass
