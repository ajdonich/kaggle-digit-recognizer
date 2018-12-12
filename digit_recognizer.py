import numpy as np
import pandas as pd

import model_data as md
import visualize_data as vs

############################################################
# Section 1: Importing/loading data
############################################################

# Load test data
# data_test = pd.read_csv("data/test.csv")
# X_test = data_test.iloc[:, 0:data_test.shape[1]]

# Load training/cross-validation data
data_block = pd.read_csv("data/train.csv")
m, data_cols = data_block.shape

# X_train = data_block.iloc[:, 1:data_cols]
# Y_train = data_block['label']

# Assign a ramdom third of the data to cross-validation
data_cval = data_block.sample(frac=0.33, random_state=3)
data_train = data_block.drop(data_cval.index, axis=0)

# Separate labels from examples
X_cval = data_cval.iloc[:, 1:data_cols]
X_train = data_train.iloc[:, 1:data_cols]

Y_cval = data_cval['label']
Y_train = data_train['label']

# print(X_cval)
# print(X_train)

############################################################
# Section 2: Visualizing data
############################################################

# # Plot MNIST numerals as one big image (RxC)
# vs.plot_mnist_image(X_train, Y_train, 10, 10)

# # Plot MNIST label distribution
# vs.plot_label_dist(Y_train)

############################################################
# Section 3: Modeling/evaluating data
############################################################

# Pre-process training data
Y_train_vec = md.vectorize_labels(Y_train)
X_train_scaled, mu, sig2 = md.normalize_data(X_train)

# Pre-pocess cross-validation data
Y_cval_vec = md.vectorize_labels(Y_cval)
X_cval_scaled, mu, sig2 = md.normalize_data(X_cval, mu, sig2)

# Create models and plot training history
flat_model, flat_history = md.generate_flat_model(X_train_scaled, Y_train_vec, X_cval_scaled, Y_cval_vec)
conv_model, conv_history = md.generate_conv_model(X_train_scaled, Y_train_vec, X_cval_scaled, Y_cval_vec)
vs.plot_train_hist(flat_history, conv_history)

############################################################
# Section 4: Evaluate model against test data
############################################################

# Evaluate model
# loss_and_metrics = model.evaluate(X_cval, Y_cval_vec, batch_size=b_size)
# print(loss_and_metrics)

pred_10d = conv_model.predict(X_cval.values.reshape(-1, 28, 28, 1), batch_size=128)

# Massage predictions back into Series format
pred_labels_np = md.vec_to_label(pred_10d)
pred_labels_pd = pd.Series(pred_labels_np, index=Y_cval.index, dtype=np.int64)

vs.plot_error_image(X_cval, Y_cval, pred_labels_pd, 20, 20)
