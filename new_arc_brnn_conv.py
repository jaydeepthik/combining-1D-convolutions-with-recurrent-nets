# -*- coding: utf-8 -*-
"""
Created on Sat Jun  9 15:07:31 2018

@author: jaydeep thik
"""

from keras import layers, models
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras import optimizers, callbacks

max_features = 10000
max_len = 500

(X_train, y_train),(X_test, y_test) = imdb.load_data(num_words=max_features)

X_train = sequence.pad_sequences(X_train, max_len)
X_test = sequence.pad_sequences(X_test, max_len)

print("reduced BLSTM ,lr, and LSTM activation:relu....\n")

model = models.Sequential()
model.add(layers.Embedding(max_features, 50, input_length=max_len))
model.add(layers.Bidirectional(layers.LSTM(16, dropout=0.1, recurrent_dropout=0.5, return_sequences=True)))
model.add(layers.Conv1D(16,10, padding='same'))
#model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling1D(5))

model.add(layers.Bidirectional(layers.LSTM(16, dropout=0.1, recurrent_dropout=0.5, return_sequences=True)))

model.add(layers.Conv1D(16, 10, padding='same'))
#model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))

model.add(layers.LSTM(16, dropout=0.1, recurrent_dropout=0.4, return_sequences=False))
#model.add(layers.LSTM(16, dropout=0.1, recurrent_dropout=0.4))
#model.add(layers.Dense(128, activation='relu'))
#model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))


model.compile(optimizer=optimizers.RMSprop(), loss='binary_crossentropy', metrics=['acc'])

#callbacks = [callbacks.TensorBoard(log_dir='./logs', histogram_freq=1, embeddings_freq=1)]

history  = model.fit(X_train, y_train, batch_size=128, epochs=10, validation_split=0.2)
