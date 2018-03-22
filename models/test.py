import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.optimizers import Adam
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split

random_seed = 42
vocab_size = 30000
x = np.load('D:/MachineLearning/Dataset/x_nn12.npy')
y = np.load('D:/MachineLearning/Dataset/y_data12.npy')
classes = len(y[0])
vocab_size = 30000
max_length = 30
embedding_vector_length = 300
dropout = 0.1
learning_rate = 0.01
nb_epochs = 20
batch_size = 64
x_train, x_test, y_train, y_test = (train_test_split(x,y, test_size = 0.2, random_state = random_seed))

del(x)
del(y)
x_train = sequence.pad_sequences(x_train, maxlen=max_length, padding = 'post')
x_test = sequence.pad_sequences(x_test, maxlen=max_length, padding = 'post')

model = Sequential()
model.add(Embedding(vocab_size, embedding_vector_length, input_length=max_length))
model.add(Dropout(dropout))
model.add(LSTM(64))
#model.add(Dropout(dropout))
model.add(Dense(classes, activation='softmax'))
# optimizer=RMSprop(lr=learning_rate),
model.compile(loss='categorical_crossentropy', optimizer = 'Adam', metrics=['accuracy'])
# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
#                               patience=2, min_lr=0.001)
#  callbacks = [reduce_lr],
model.fit(x_train, y_train, batch_size = batch_size, 
          epochs = nb_epochs, verbose=1, validation_data=(x_test, y_test))

json_string = model.to_json() # as json 
open('D:/MachineLearning/Dataset/lstm1_model.json', 'w').write(json_string) 
model.save_weights('D:/MachineLearning/Dataset/lstm1_wts.h5')
print("Model and weights Saved")
