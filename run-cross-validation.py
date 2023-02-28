# -*- coding: utf-8 -*-
from unidecode import unidecode

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding, LSTM

from keras import backend as K
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.metrics import hamming_loss
import pickle

def remove_non_ascii(text):
    return unidecode(text)

def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc

## https://blog.manash.me/multi-task-learning-in-keras-implementation-of-multi-task-classification-loss-f1d42da5c3f6
def multitask_loss(y_true, y_pred):
    # Avoid divide by 0
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    # Multi-task loss
    return K.mean(K.sum(- y_true * K.log(y_pred) - (1 - y_true) * K.log(1 - y_pred), axis=1))

def exact_match(y_true, y_pred):
    np_y_true = K.get_value(y_true)
    np_y_pred = K.get_value(y_pred)

    return K.variable(accuracy_score(np_y_true, np_y_pred))

def to_categorical_multi(y, num_classes):
    ret = []
    for yi in y:
        temp = [0] * num_classes
        for yij in yi:
            temp[yij] = 1
        ret.append(temp)
    return np.array(ret)

def hamming_loss_metric(y_true, y_pred):
    return hamming_loss(y_true, y_pred)

def hamming_loss_metric2(y_true, y_pred):
    np_y_true = K.get_value(y_true)
    np_y_pred = K.get_value(y_pred)

    return K.variable(hamming_loss(np_y_true, np_y_pred))

def hamming_loss_keras(y_true, y_pred):
    nonzero = K.cast(K.tf.count_nonzero(y_true - y_pred, axis=-1), K.tf.float32)
    return K.abs(nonzero / K.cast((K.shape(y_true)[0] * K.shape(y_true)[1]), K.tf.float32))

max_words = 100000 # 569136
batch_size = 512
epochs = 30
i = 0

x_train = []
y_train = []
x_test = []
y_test = []

x_films = [[],[],[],[],[]]
y_films = [[],[],[],[],[]]
key_films = [[],[],[],[],[]]

num_classes = 28
print(num_classes, 'classes')

print('Building model...')
model = Sequential()
model.add(LSTM(128, input_shape=(1, max_words), dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
model.add(LSTM(256, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(num_classes))
model.add(Activation('softmax'))
model.summary()
start_weights = model.get_weights()

with open('dataset.pickle', 'rb') as output:
    x_films = pickle.load(output)

with open('results.pickle', 'rb') as output:
    y_films = pickle.load(output)

with open('keys.pickle', 'rb') as output:
    key_films = pickle.load(output)

print('%s Subtitle loaded' % len(x_films))
print('Learning will start with %s films' % len(x_films))

for cv_round in range(0, 5):
    print(len(x_films[cv_round]), 'fold ' + str(cv_round) + ' sequences')
    x_test = np.array(x_films[cv_round], copy=True)
    y_test = np.array(y_films[cv_round], copy=True)
    x_train = False
    y_train = False

    for i in range(0, 5):
        if i != cv_round:
            if type(x_train) == bool:
                x_train = np.array(x_films[i], copy=True)
                y_train = np.array(y_films[i], copy=True)
            else:
                x_train = np.concatenate((x_train, x_films[i]), axis=0)
                y_train = np.concatenate((y_train, y_films[i]), axis=0)

    print('Fold %s' % cv_round)
    print(len(x_train), 'train sequences')
    print(len(x_test), 'test sequences')

    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)
    print(x_train[0])

    print('Convert class vector to binary class matrix (for use with categorical_crossentropy)')

    # y_train = keras.utils.to_categorical(y_train, num_classes)
    # y_test = keras.utils.to_categorical(y_test, num_classes)
    y_train = to_categorical_multi(y_train, num_classes)
    y_test = to_categorical_multi(y_test, num_classes)
    print('y_train shape:', y_train.shape)
    print('y_test shape:', y_test.shape)

    last = 0
    model.compile(loss=multitask_loss, #'binary_crossentropy', #multitask_loss
              optimizer='adam',
              metrics=['accuracy', 'categorical_accuracy', auc])

    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        shuffle=True,
                        verbose=1,
                        validation_split=0.1)
    score = model.evaluate(x_test, y_test,
                           batch_size=batch_size, verbose=1)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    test_predict_class = model.predict_classes(x_test)
    test_predict = model.predict(x_test)
    ret = []
    for t in test_predict:
        row = []
        for i in t:
            if i> 5e-01:
                row.append(1)
            else:
                row.append(0)
        ret.append(row)

    test_predict_softmax = ret

    ascore = accuracy_score(y_test, np.array(test_predict_softmax))
    hamm = hamming_loss_metric(y_test, np.array(test_predict_softmax))

    print("Exact Match Score : %s " % ascore)
    print("Hamming Loss : %s " % hamm)

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for i in range(0, len(y_test)):
        for j in range(0, 28):
            if y_test[i][j]:
                if y_test[i][j] == test_predict_softmax[i][j]:
                    tp += 1
                else:
                    fn += 1
            else:
                if y_test[i][j] == test_predict_softmax[i][j]:
                    tn += 1
                else:
                    fp += 1

    print('TP %s' % tp)
    print('TN %s' % tn)
    print('FP %s' % fp)
    print('FN %s' % fn)
    print('ACC : %s' % (float(tp+tn) / float(tp+tn+fp+fn)))
    model.save_weights("fold_" + str(cv_round) + "_model.h5")

    last = ascore
    test = open("result.txt","a+")
    test.write('x[' + str(cv_round) + '] = ')
    for t in test_predict:
        test.write(str(t) + "\n")

    test.write('===========\n')

    test.write('y[' + str(cv_round) + '] = ')
    for t in test_predict_softmax:
        test.write(str(t) + "\n")

    test.write('===========\n')

    test.write('z[' + str(cv_round) + '] = ')
    for t in y_test:
        test.write(str(t) + "\n")
    test.write('###############\n')

    print("###########################")
    model.set_weights(start_weights)

    del x_train
    del y_train
    del y_test
    del x_test
    print("Cleaned.......")
