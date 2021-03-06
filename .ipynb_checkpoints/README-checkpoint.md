# Tweets Hashtag Prediction
## Dataset: 476 million Twitter tweets
467 million Twitter posts from 20 million users covering a 7 month period from June 1 2009 to December 31 2009. We estimate this is about 20-30% of all public tweets published on Twitter during the particular time frame.

For each public tweet the following information is available: 

- Author
- Time
- Content

## Project Idea
The hashtags of tweets usually represent the main information of the content, so predicting hashtags could help users to classify their tweets into a particular topic. In this way, implementing an efficient hashtag prediction algorithm is essential. Old techniques like K nearest neighborhoods could be time consuming and may not reach a state-of-art accuracy; whereas neural networks such as sequence model could help avoid these drawback and hopefully provide a better result.

## The approaches
### Preprocessing
 - **TweetstoVec**
 - **Stemming**
 - **TF/IDF**
 - **Remove Stop Words**
 - **WordtoVec:SkipGram**

### Evaluation model 
 - **Recall/Precision**
 - **Accuracy**
 
### Machine learning and deep learning models
 - **Naive Bayes**
 - **KNN**
 - **Sequence Model**

## Softwares
- Python 3.6
- Numpy
- Keras using tensorflow backend
- Pandas
- Scikit Learn

## References
- J. Yang, J. Leskovec. Temporal Variation in Online Media. ACM International Conference on Web Search and Data Mining (WSDM '11), 2011.
- Dhingra, Bhuwan, Zhong Zhou, Dylan Fitzpatrick, Michael Muehl, and William W. Cohen. "Tweet2Vec: Character-Based Distributed Representations for Social Media." ACL (2016).
- Li, Tianxi and Yu Wu. “Twitter Hash Tag Prediction Algorithm.” (2010).

 