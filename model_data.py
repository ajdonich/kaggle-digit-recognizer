import numpy as np
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, Flatten

def normalize_data(X, mu=None, sig2=None):

    # Nudge all data by 1 to prevent div by zero below
    X = X.add(np.ones(X.shape))

    # Calc mu and sig2 on orginal data
    if mu is None: mu = X.mean(axis=0)
    if sig2 is None: sig2 = X.mul(X, axis=0).mean(axis=0)    
    #print(str(mu) + ", " + str(sig2))

    # Center and scale xi = (xi - mui) / sig2i
    X = X.sub(mu, axis=1).div(sig2, axis=1)
    #print(X)

    return X, mu, sig2

def vectorize_labels(Y):

    m = Y.shape[0]
    Y_vec = np.zeros((m, 10))

    # Translate to 10d vector
    for i in range(0, m):
        Y_vec[i, Y.iloc[i]] = 1

    return Y_vec 

def vec_to_label(Y_vec):

    m, lindex = Y_vec.shape
    Y = np.zeros((m, ))

    # Translate to 1d vector
    for i in range(0, m):

        for l in range(0, lindex):
            if (Y_vec[i,l] >= 0.9 ):
                Y[i] = l; break
            
    return Y 

def generate_flat_model(X_train, Y, X_cval, Y_cval, num_epochs=100):

    n = X_train.shape[1]

    # Build model
    model = Sequential()
    model.add(Dense(units=n, activation='relu', input_dim=n))
    model.add(Dense(units=int(n/2), kernel_regularizer=regularizers.l2(0.01), activation='relu'))
    model.add(Dense(units=int(n/2), kernel_regularizer=regularizers.l2(0.01), activation='relu'))
    model.add(Dense(units=int(n/4), kernel_regularizer=regularizers.l2(0.01), activation='relu'))

    model.add(Dense(units=10, activation='softmax'))

    # Configure and train model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(X_train, Y, validation_data=(X_cval, Y_cval), epochs=num_epochs, batch_size=128)

    return model, history

def generate_conv_model(X_train, Y, X_cval, Y_cval, num_epochs=5):

    model = Sequential()

    model.add(Conv2D(filters = 32, kernel_size = (5,5), kernel_regularizer=regularizers.l2(0.01), padding = 'Same', activation ='relu', input_shape = (28,28,1)))
    model.add(Conv2D(filters = 32, kernel_size = (5,5), kernel_regularizer=regularizers.l2(0.01), padding = 'Same', activation ='relu'))
    model.add(Conv2D(filters = 64, kernel_size = (3,3), kernel_regularizer=regularizers.l2(0.01), padding = 'Same', activation ='relu'))
    model.add(Conv2D(filters = 64, kernel_size = (3,3), kernel_regularizer=regularizers.l2(0.01), padding = 'Same', activation ='relu'))

    model.add(Flatten())
    model.add(Dense(256, activation = "relu"))
    model.add(Dense(10, activation = "softmax"))

    # Reshape data
    X_train_2D = X_train.values.reshape(-1, 28, 28, 1)
    X_cval_2D = X_cval.values.reshape(-1, 28, 28, 1)

    # Configure and train model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(X_train_2D, Y, validation_data=(X_cval_2D, Y_cval), epochs=num_epochs, batch_size=128)

    return model, history