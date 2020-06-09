import json
import csv
import sys
import getopt
import os
import math
import operator
import numpy as np
from nltk.tokenize import TweetTokenizer 
from collections import defaultdict
import random
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models.keyedvectors import KeyedVectors
import scipy 
from scipy.sparse import hstack, csr_matrix
import nltk
from nltk.tokenize import word_tokenize



# reads items in json file as a list of dicts, where the entries are "article_link", "headline", "is_sarcastic"
def read_headlines(filename = 'data/headlines/Sarcasm_Headlines_Dataset.json'):
    artList = []
    with open(filename) as f:
        for jsonObj in f:
            art = json.loads(jsonObj)
            artList.append(art)
    return artList

def naive_headlines(articles=read_headlines()):
    artlist = {}
    for art in articles:
        artlist[art["headline"]] = art["is_sarcastic"]
    return artlist

# extract tweets as dict mapping tweet to sarcasm = 1, not sarcasm = 0 
def read_tweets(filename = 'data/twitter/sarcasm-dataset.txt'):
    tweetlist = {}
    with open(filename, 'r') as fp:
        for twt in fp:
            tweetlist[twt[0:len(twt)-2].strip()] = twt[-2]
    return tweetlist

# returns json file as dict
def read_reddit(filename = 'data/reddit/comments.json'):
    with open(filename) as f:
        for jsonObj in f:
            c = json.loads(jsonObj)
    return c

def read_reddit_label(filename = 'data/reddit/train-balanced.csv'):
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        lab = {}
        for row in csv_reader:
            line_count+=1
            sp = row[0].split('|')
            for i,s in enumerate(sp[1].split()):
                lab[s] = int(sp[2].split()[i])
        print(f'Processed {line_count} lines.')
        return lab

def read_tweets_csv(filename = 'data/twitter/dataset_csv.csv'):
    tweetlist = {}
    numTweets = 0
    with open(filename, 'r') as fp:
        for i,twt in enumerate(fp):
            numTweets = i  
            tweetlist[twt[0:len(twt)-2].strip()] = twt[-2]
    print("num tweets (csv file): {}".format(numTweets+1))
    return tweetlist

def main():
    # evaluate(False, True)
    # data = naive_headlines()
    # X_train, X_test, y_train, y_test = train_test_split(list(data.keys()), list(data.values()), random_state=1)
    # print(X_test)
    # cv = CountVectorizer()
    # X_train_cv = cv.fit_transform(X_train)
    # X_test_cv = cv.transform(X_test)
    # nb = MultinomialNB()
    # nb.fit(X_train_cv, y_train)
    # predictions = nb.predict(X_train_cv)
    # print('Accuracy score: ', accuracy_score(y_train, predictions))
    # print('Precision score: ', precision_score(y_train, predictions))
    # print('Recall score: ', recall_score(y_train, predictions))

    # lr = LogisticRegression()
    # lr.max_iter = 10000
    # lr.fit(X_train_cv,y_train)
    # predictions = lr.predict(X_train_cv)
    # print('Accuracy score: ', accuracy_score(y_train, predictions))
    # print('Precision score: ', precision_score(y_train, predictions))
    # print('Recall score: ', recall_score(y_train, predictions))

    comments = read_reddit()
    labels = read_reddit_label()
    y_train = list(labels.values())
    X_train = [comments[i]['text'] for i in list(labels.keys())]
    labels = read_reddit_label('data/reddit/test-balanced.csv')
    cv = CountVectorizer(ngram_range=(1, 2))
    cv = CountVectorizer(analyzer='char', ngram_range=(1, 6))
    y_test = list(labels.values())
    X_test = [comments[i]['text'] for i in list(labels.keys())]
    X_train_cv = cv.fit_transform(X_train)
    X_test_cv = cv.transform(X_test)
    nb = MultinomialNB()
    nb.fit(X_train_cv, y_train)
    predictions = nb.predict(X_test_cv)
    print('Accuracy score: ', accuracy_score(y_test, predictions))
    print('Precision score: ', precision_score(y_test, predictions))
    print('Recall score: ', recall_score(y_test, predictions))

    lr = LogisticRegression()
    lr.fit(X_train_cv,y_train)
    feature_names = cv.get_feature_names()
    coefs_with_fns = sorted(zip(lr.coef_[0], feature_names))
    top = zip(coefs_with_fns[:20], coefs_with_fns[:-(20 + 1):-1])

    for (coef_1, fn_1), (coef_2, fn_2) in top:
        print ("\t%.4f\t%-15s\t\t%.4f\t%-15s" % (coef_1, fn_1, coef_2, fn_2))

    predictions = lr.predict(X_test_cv)
    print('Accuracy score: ', accuracy_score(y_test, predictions))
    print('Precision score: ', precision_score(y_test, predictions))
    print('Recall score: ', recall_score(y_test, predictions))
    
    # Naive Bayes
    # nb = MultinomialNB()
    # nb.fit(X_train_cv2, y_train)
    # predictions = nb.predict(X_test_cv2)
    # print('NB Accuracy score: ', accuracy_score(y_test, predictions))
    # print('NB Precision score: ', precision_score(y_test, predictions))
    # print('NB Recall score: ', recall_score(y_test, predictions))

    # Logistic Regression
    # lr = LogisticRegression(max_iter=10000)
    # lr.fit(X_train_cv2, y_train)
    # predictions = lr.predict(X_test_cv2)
    # print('LR Accuracy score: ', accuracy_score(y_test, predictions))
    # print('LR Precision score: ', precision_score(y_test, predictions))
    # print('LR Recall score: ', recall_score(y_test, predictions))

    # Adaboost Classifier
    abc = AdaBoostClassifier(n_estimators=1000, learning_rate = 0.9, random_state=0)
    abc.fit(X_train_cv, y_train)
    predictions = abc.predict(X_test_cv)
    print('Adaboost Accuracy score: ', accuracy_score(y_test, predictions))
    print('Adaboost Precision score: ', precision_score(y_test, predictions))
    print('Adaboost Recall score: ', recall_score(y_test, predictions))

    # Random Forest - precision score is wonky, produces errors
    # rf = RandomForestClassifier(max_depth=2, random_state=0)
    # rf.fit(X_train_cv, y_train)
    # predictions = rf.predict(X_test_cv)
    # print('RandomForest Accuracy score: ', accuracy_score(y_test, predictions))
    # print('RandomForest Precision score: ', precision_score(y_test, predictions))
    # print('RandomForest Recall score: ', recall_score(y_test, predictions))

    # Gradient Boosting
    gb = GradientBoostingClassifier(loss='deviance')
    gb.fit(X_train_cv, y_train)
    predictions = gb.predict(X_test_cv)
    print('GB Accuracy score: ', accuracy_score(y_test, predictions))
    print('GB Precision score: ', precision_score(y_test, predictions))
    print('GB Recall score: ', recall_score(y_test, predictions))

    # K-neighbors
    # kn = KNeighborsClassifier()
    # kn.fit(X_train_cv2, y_train)
    # predictions = kn.predict(X_test_cv2)
    # print('KN Accuracy score: ', accuracy_score(y_test, predictions))
    # print('KN Precision score: ', precision_score(y_test, predictions))
    # print('KN Recall score: ', recall_score(y_test, predictions))

if __name__ == "__main__":
    main()