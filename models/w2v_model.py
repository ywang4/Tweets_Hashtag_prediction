from keras.models import Sequential 
from keras.layers import Dense, Activation 
from sklearn import preprocessing
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
import numpy as np
from gensim.models import Word2Vec
from tqdm import tqdm
import tensorflow as tf

data_path = 'D:/MachineLearning/Dataset/'

y = np.load(data_path + 'y_data12.npy')
x = np.load(data_path + 'x_data12.npy')

print("Data loaded: totally {} tweets and {} corresponding hashtags".format(len(x),len(y)))

'''
word2vec
sg = 1 means using skikgram
size = 300 means shape of matrix for each word is 1x300
alpha is learning rate
min_count = 6 means words count less than 6 will be marked as rare unknown word 
'''

w2vmodel = Word2Vec(x, sg = 1, size = 300, alpha = 0.005, window = 2, 
                 min_count = 6, workers = 8, negative = 2)
w2vmodel.save(data_path + 'w2vmodel.bin')
#take average for each word in a sentence to generate a 1x300 vector for each tweets
x_data = []

print('Generating vector form for each tweets:')
for i in tqdm(range(len(x))):
    temp = np.zeros((300,), dtype=float)
    for j in range(len(x[i])):
        word = x[i][j]
        if(word in w2vmodel.wv.vocab):
            temp += w2vmodel[word]
    x_data.append(temp/float(len(x[i])))
del(x)


#input X_data, Y_data, then use logistic regression to fit the data and evaluate the model
#ouput recall, precision, accuracy
#logistic model
input_dim = len(x_data[0])
output_dim = len(y[0])
x_train, x_test, y_train, y_test = train_test_split(np.asarray(x_data), y, test_size = 0.2, random_state = 42)
np.save(data_path + 'test_data/x_test_w2v', x_test)
np.save(data_path + 'test_data/y_test_w2v', y_test)
del(x_data)
del(y)
print("data ready to be trained")
model = Sequential() 
model.add(Dense(output_dim, input_dim=input_dim, activation='softmax')) 
batch_size = 100 
nb_epoch = 30

model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
print("model compiled")

history = model.fit(x_train, y_train, batch_size=batch_size, epochs=nb_epoch,verbose=1,validation_data=(x_test, y_test))


json_string = model.to_json() # as json 
open(data_path + 'w2v_Logistic_model.json', 'w').write(json_string) 
model.save_weights(data_path + 'w2v_Logistic_wts.h5')
print(" the model and weights saved")

