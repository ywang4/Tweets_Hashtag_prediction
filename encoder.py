'''
Input: Raw tweets line by line
Preprocess: remove stop word, html, metion, url, hashtags;
			tokenize
Output: list of words list in raw tweets as x_data
		list of binary vector as y_data
		list of hashtags with frequency more than 500 as labels for further uses

'''

import re
import sys
import io
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from keras.preprocessing import text
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn import preprocessing
from keras.utils import np_utils


#input file is raw tweets in 2009 December
in_file = 'D:/MachineLearning/Dataset/tweets-12.txt'
x_file = 'D:/MachineLearning/Dataset/test/x_data12'
raw_y_file = 'D:/MachineLearning/Dataset/test/lable12'
y_file = 'D:/MachineLearning/Dataset/test/y_data12'



regex_str = [
    r'<[^>]+>', # HTML tags
    r'(?:@[\w_]+)', # @-mentions
    r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", # hash-tags
    r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', # URLs
 
    r'(?:(?:\d+,?)+(?:\.?\d+)?)', # numbers
    r"(?:[a-z][a-z'\-_]+[a-z])", # words with - and '
    r'(?:[\w_]+)', # other words
    r'(?:\S)+' # anything else
]

tokens_re = re.compile(r'('+'|'.join(regex_str)+')', re.VERBOSE | re.IGNORECASE)

def tokenize(s):
    return tokens_re.findall(s)

def preprocess(s, lowercase=True):
    tokens = tokenize(s)
    tokens = [token.lower() for token in tokens]

    html_regex = re.compile('<[^>]+>')
    tokens = ['' if html_regex.match(token) else token for token in tokens]

    mention_regex = re.compile('(?:@[\w_]+)')
    tokens = ['@user' if mention_regex.match(token) else token for token in tokens]

    url_regex = re.compile('http[s]?://(?:[a-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+')
    tokens = ['' if url_regex.match(token) else token for token in tokens]

    hashtag_regex = re.compile("(?:\#+[\w_]+[\w\'_\-]*[\w_]+)")
    hashtags = []
    for token in tokens:
        if hashtag_regex.match(token):
            token = token.replace('#', '')
            hashtags.append(token)

    flag = False
    for item in tokens:
        if item=='rt':
            flag = True
            continue
        if flag and item=='@user':
            return ''
        else:
            flag = False

    if hashtags:
        tweets = str(hashtags[0]) + str(' ') + ' '.join([t for t in tokens if t]).replace('rt @user: ','')
        return tweets
    else:
        return ''


#Step1: Extract hashtags with more than 500 frequency
words = []
with open(in_file, 'r',encoding="utf-8") as in_lines:
    for line in in_lines:
        a = preprocess(line.rstrip())
        if a != '':
            words.append(a.split(' ',1)[0])
fdist = nltk.FreqDist(words)
a = list(filter(lambda x: x[1]>=500,fdist.items()))
hashtags = []
for i in range(len(a)):
    hashtags.append(a[i][0])

print('hashtags extraction finished')
del(words)
del(fdist)
del(a)

#Step2: preprocess the raw tweets and tokenize tweets with hashtags in list
# input and output files
stop_words = set(stopwords.words('english'))
tweets = []
y = []

with open(in_file, 'r', encoding = 'utf-8') as file_in:
	for line in file_in:
		line = preprocess(line.rstrip())
		if line != '':
			if(line.split(' ',1)[0] in hashtags):
				words = text.text_to_word_sequence(line,
				                           filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
				                           lower=True,
				                           split=" ")
				new_words = []
				for x in words:
					if (not x in stop_words) and (not x in hashtags):
						new_words.append(x)
				if new_words:
					y.append(words[0])
					tweets.append(new_words)

del(hashtags)
np.save(x_file, tweets)
np.save(raw_y_file, y)
del(tweets)
print('tweets and labels saved')

#build y data
le = preprocessing.LabelEncoder()
le.fit(y)
nb_classes = len(list(le.classes_))
output_dim = nb_classes
y_temp = le.transform(y)
y_data = np_utils.to_categorical(y_temp, nb_classes)

np.save(y_file, y_data)
print('y_data saved, preprocessing completed')
			
