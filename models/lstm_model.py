import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
import nltk
from tqdm import tqdm

random_seed = 42
vocab_size = 30000
x = np.load('D:/MachineLearning/Dataset/x_data12.npy')
y = np.load('D:/MachineLearning/Dataset/y_data12.npy')
print("Data Loaded Successfully")

texts = [token for lines in x for token in lines]
fdist = nltk.FreqDist(texts)
words = fdist.most_common(vocab_size)
dictionary = [pair[0] for pair in words]
data = x
x = []
for i in tqdm(range(len(data))):
    line = data[i]
    temp = []
    for word in line:
        if word in dictionary:
            temp.append(dictionary.index(word)+1)
        else:
            temp.append(0)
    x.append(temp)

print("X_data generated, totally {} tweets and {} corresponding hashtags".format(len(x),len(y)))
classes = len(y[0])
vocab_size = 30000
max_length = 40
embedding_vector_length = 300
dropout = 0.2
learning_rate = 0.01
nb_epochs = 5
batch_size = 128


x_train, x_test, y_train, y_test = (train_test_split(x,y, test_size = 0.2, random_state = random_seed))

del(x)
del(y)
x_train = sequence.pad_sequences(x_train, maxlen=max_length, padding = 'post')
x_test = sequence.pad_sequences(x_test, maxlen=max_length, padding = 'post')
np.save('D:/MachineLearning/Dataset/test_data/x_test_lstm',x_test)
np.save('D:/MachineLearning/Dataset/test_data/y_test_lstm',y_test)

model = Sequential()
model.add(Embedding(vocab_size, embedding_vector_length, input_length=max_length))
model.add(LSTM(128,return_sequences=True))
model.add(Dropout(dropout))
model.add(LSTM(128))
model.add(Dropout(dropout))
model.add(Dense(classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', 
              optimizer=RMSprop(lr=learning_rate), 
              metrics=['accuracy'])
model.fit(x_train, y_train, batch_size = batch_size, 
          epochs = nb_epochs, verbose=1, validation_data=(x_test, y_test))

json_string = model.to_json() # as json 
open('D:/MachineLearning/Dataset/lstm_model.json', 'w').write(json_string) 
model.save_weights('D:/MachineLearning/Dataset/lstm_wts.h5')
print("Model and weights Saved")
