#neural network to train on binary genotypes saved fitness evalauted data from random evaluations of midi sounds
#using k fold cross validation and saving model to use in GA

from tensorflow import keras
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras import regularizers
from matplotlib import pyplot as plt

# Load data
train_data = pd.read_csv('GAdata6.csv')
#test_data = pd.read_csv('test_data.csv')

# Preprocess data
X = train_data.loc[:, "bin"]
Y = train_data.loc[:, "fit"]
Y = (Y - np.min(Y)) / (np.max(Y) - np.min(Y))
new_X = []
for row in X.to_numpy():
    new_X.append([int(x) for x in row])
X = np.array(new_X)

# Split data into 10 folds
num_folds = 10
kfolds = KFold(n_splits=num_folds, shuffle=True)

# Define model
def build_model():
    model = Sequential()
    model.add(Dense(90, input_dim=45, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(180, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='mse', optimizer='adam', metrics=['mse'])
    return model

# Train and evaluate model on each fold
mae_scores = []
corr_scores = []
fold = 0
for train_idx, val_idx in kfolds.split(X):
    fold += 1
    print(f"Fold {fold}/{num_folds}")

    # Split data into training and validation sets for this fold
    x_train, y_train = X[train_idx], Y[train_idx]
    x_val, y_val = X[val_idx], Y[val_idx]

    # Build and train model
    model = build_model()
    early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
    history = model.fit(x_train, y_train, epochs=500, batch_size=32, validation_data=(x_val, y_val), verbose=1, callbacks=[early_stop, reduce_lr])

    # Evaluate model on validation set
    predictions = np.array(model.predict(x_val))
    actuals = np.array(y_val)
    errors = [abs(x - y) for x, y in zip(predictions, actuals)]
    mae = np.average(errors)
    correlation = np.corrcoef([x[0] for x in predictions], actuals)
    print(f"MAE: {mae}")
    print(f"Correlation: {correlation}")

    # Add scores for this fold to overall scores
    mae_scores.append(mae)
    corr_scores.append(correlation[0][1])
    model.save('GANNsig{0}.h5'.format(str(fold)))

# Print average MAE and correlation over all folds
print(f"Average MAE: {np.mean(mae_scores)}")
print(f"Average Correlation: {np.mean(corr_scores)}")

