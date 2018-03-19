from keras.models import Sequential 
from keras.layers import Dense, Activation 
from sklearn import preprocessing
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
import numpy as np
import scipy.sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm


x_data = np.load('D:/MachineLearning/Dataset/x_data12.npy')
y_data = np.load('D:/MachineLearning/Dataset/y_data12.npy')



#input X_data, Y_data, then use logistic regression to fit the data and evaluate the model
#ouput recall, precision, accuracy
def logistic_model(x, y):
	input_dim = x.shape[1]
	output_dim = len(y[0])
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)
	scipy.sparse.save_npz('D:/MachineLearning/Dataset/test_data/x_test_tfidf.npz',x_test)
	np.save('D:/MachineLearning/Dataset/test_data/y_test_tfidf',y_test)
	print("data ready to be trained")
	model = Sequential() 
	model.add(Dense(output_dim, input_dim=input_dim, activation='softmax')) 
	batch_size = 512
	nb_epoch = 30

	print("start to train model")
	model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
	model.fit(x_train.todense(), y_train, 
		batch_size=batch_size, nb_epoch=nb_epoch,verbose=1, validation_data=(x_test.todense(), y_test))

	print("save the model and weights")
	json_string = model.to_json() # as json 
	open('D:/MachineLearning/Dataset/tfidf_Logistic_model.json', 'w').write(json_string) 
	model.save_weights('D:/MachineLearning/Dataset/tfidf_Logistic_wts.h5')


corpus = []
for lines in x_data:
	corpus.append(" ".join(lines))
tfidf = TfidfVectorizer(max_df = 0.5,max_features = 2000)
tfs = tfidf.fit_transform(corpus)

logistic_model(tfs,y_data)
