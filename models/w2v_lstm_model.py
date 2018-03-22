from keras.models import Sequential 
from keras.layers import Dense, Activation, Dropout, LSTM 
from sklearn import preprocessing
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
import numpy as np
from gensim.models import Word2Vec
from tqdm import tqdm
from keras.preprocessing import sequence

data_path = 'D:/MachineLearning/Dataset/'

y = np.load(data_path + 'y_data12.npy')
x = np.load(data_path + 'x_data12.npy')

print("Data loaded: totally {} tweets and {} corresponding hashtags".format(len(x),len(y)))

w2v_model = Word2Vec(x, sg = 1, size = 100, alpha = 0.005, window = 2, 
                 min_count = 6, workers = 8, negative = 2)

x_data = []
for i in tqdm(range(len(x))):
	temp = []
	for word in x[i]:
		if word in w2v_model.wv.vocab:
			temp.append(w2v_model.wv[word])
	x_data.append(temp)

del(x)
del(w2v_model)
classes = len(y[0])
random_seed = 42
vocab_size = 30000
max_length = 20
input_demension = len(x_data[0][0])
dropout = 0.1
#learning_rate = 0.01
nb_epochs = 20
batch_size = 64

x_train, x_test, y_train, y_test = (train_test_split(x_data,y, test_size = 0.2, random_state = random_seed))

del(x_data)
del(y)



x_train = sequence.pad_sequences(x_train, maxlen=max_length, padding = 'post')
x_test = sequence.pad_sequences(x_test, maxlen=max_length, padding = 'post')
np.save('D:/MachineLearning/Dataset/test_data/x_test_w2vnn',x_test)
np.save('D:/MachineLearning/Dataset/test_data/y_test_w2vnn',y_test)

model = Sequential()
#model.add(Dropout(dropout))
model.add(LSTM(512, input_shape=(max_length, input_demension)))
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
open('D:/MachineLearning/Dataset/w2vlstm_model.json', 'w').write(json_string) 
model.save_weights('D:/MachineLearning/Dataset/w2vlstm_wts.h5')
print("Model and weights Saved")

