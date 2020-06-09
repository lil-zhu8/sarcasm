import json, csv, sys, getopt, os, math, operator, random, scipy, nltk
import numpy as np
from nltk.tokenize import TweetTokenizer 
from collections import defaultdict
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import FunctionTransformer
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models.keyedvectors import KeyedVectors
from scipy.sparse import hstack, csr_matrix
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

votes_train = []
votes_test = []
# returns json file as dict
def read_reddit(filename = 'data/reddit/comments.json'):
    with open(filename) as f:
        for jsonObj in f:
            c = json.loads(jsonObj)
    return c

def read_reddit_label(filename = 'data/reddit/train-balanced.csv'):
    comments = read_reddit()
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        lab = {}
        for row in csv_reader:
            line_count+=1
            sp = row[0].split('|')
            for i,s in enumerate(sp[1].split()):
                lab[s] = int(sp[2].split()[i])
        if filename == 'data/reddit/train-balanced.csv':
            global votes_train
            votes_train = [[abs(comments[i]['ups']),abs(comments[i]['downs'])] for i in list(lab.keys())]
        else:
            global votes_test
            votes_test = [[abs(comments[i]['ups']),abs(comments[i]['downs'])] for i in list(lab.keys())]
        print(f'Processed {line_count} lines.')
        return ([comments[i]['text'] for i in list(lab.keys())], list(lab.values()))

def read_tweets_csv(filename = 'data/twitter/dataset_csv.csv'):
    tweetlist = {}
    numTweets = 0
    with open(filename, 'r') as fp:
        for i,twt in enumerate(fp):
            numTweets = i  
            tweetlist[twt[0:len(twt)-2].strip()] = twt[-2]
    print("num tweets (csv file): {}".format(numTweets+1))
    return tweetlist

def get_length(x):
    return np.array([len(t) for t in x]).reshape(-1, 1)

posW = set()
negW = set()
def buildSentiment():
    with open('data/sentiment/NRC-emotion-lexicon.txt', 'r') as fp:
        for line in fp:
            word,emotion,value = line.split('\t')
            if emotion == 'positive' and int(value) == 1:
                posW.add(word)
            if emotion == 'negative' and int(value) == 1:
                negW.add(word)

def use_sentiment(x):
    sent = []
    for t in x:
        words = t.split()
        poscount = negcount = 0
        for word in words:
            if word in posW:
                poscount += 1
            if word in negW:
                negcount += 1
        sent.append([poscount,negcount])
    return np.array(sent).reshape(-1, 2)

def use_score(x):
    if len(x) == len(votes_train):
        return np.array(votes_train).reshape(-1, 2)
    else:
        return np.array(votes_test).reshape(-1, 2)

def clf(classifier = MultinomialNB()):
    c = Pipeline([
        ('features', FeatureUnion([
            ('text', Pipeline([
                ('vectorizer', CountVectorizer(analyzer='char', ngram_range=(1, 6))),
                # ('tfidf', TfidfTransformer()),
            ])),
            ('length', Pipeline([
                ('count', FunctionTransformer(get_length, validate=False)),
            ])),
            ('sentiment', Pipeline([
                ('sentcount', FunctionTransformer(use_sentiment, validate=False)),
            ])),
            ('votes', Pipeline([
                ('updown', FunctionTransformer(use_score, validate=False)),
            ]))
        ])),
        ('clf', classifier)])
    return c

def main():
    # data = naive_headlines()
    # X_train, X_test, y_train, y_test = train_test_split(list(data.keys()), list(data.values()), random_state=1)
    # cv = CountVectorizer()
    # X_train_cv = cv.fit_transform(X_train)
    # X_test_cv = cv.transform(X_test)

    # setup for SARC 2.0 dataset
    comments = read_reddit()
    X_train,y_train = read_reddit_label()
    X_test, y_test = read_reddit_label('data/reddit/test-balanced.csv')
    buildSentiment()

    # cv = CountVectorizer(ngram_range=(1, 2))
    cv = CountVectorizer(analyzer='char', ngram_range=(1, 6))
    X_train_cv = cv.fit_transform(X_train)
    X_test_cv = cv.transform(X_test)

    # Naïve Bayes

    nb = clf()
    nb.fit(X_train, y_train)
    predictions = nb.predict(X_test)
    print('NB Accuracy score: ', accuracy_score(y_test, predictions))
    print('NB Precision score: ', precision_score(y_test, predictions))
    print('NB Recall score: ', recall_score(y_test, predictions))
    print('NB F1 score: ', f1_score(y_test, predictions))


    # Logistic Regression
    lr = clf(LogisticRegression())
    lr.fit(X_train,y_train)

    # print top 20 weights
    coeff = lr.named_steps['clf'].coef_[0]
    fnames = dict(lr.named_steps['features'].transformer_list).get('text').named_steps['vectorizer'].get_feature_names()
    coefs_with_fns = sorted(zip(coeff[0:len(fnames)+1], fnames))
    top = zip(coefs_with_fns[:20], coefs_with_fns[:-(20 + 1):-1])

    for (coef_1, fn_1), (coef_2, fn_2) in top:
        print ("\t%.4f\t%-15s\t\t%.4f\t%-15s" % (coef_1, fn_1, coef_2, fn_2))

    predictions = lr.predict(X_test)
    print('LR Accuracy score: ', accuracy_score(y_test, predictions))
    print('LR Precision score: ', precision_score(y_test, predictions))
    print('LR Recall score: ', recall_score(y_test, predictions))
    print('LR F1 score: ', f1_score(y_test, predictions))

    # Adaboost Classifier
    abc = clf(AdaBoostClassifier(n_estimators=1000, learning_rate = 0.9, random_state=0))
    abc.fit(X_train, y_train)
    predictions = abc.predict(X_test)
    print('Adaboost Accuracy score: ', accuracy_score(y_test, predictions))
    print('Adaboost Precision score: ', precision_score(y_test, predictions))
    print('Adaboost Recall score: ', recall_score(y_test, predictions))
    print('Adaboost F1 score: ', f1_score(y_test, predictions))

    # Random Forest
    rf = clf(RandomForestClassifier(max_depth=2, random_state=0))
    rf.fit(X_train, y_train)
    predictions = rf.predict(X_test)
    print('RandomForest Accuracy score: ', accuracy_score(y_test, predictions))
    print('RandomForest Precision score: ', precision_score(y_test, predictions))
    print('RandomForest Recall score: ', recall_score(y_test, predictions))
    print('RandomForest F1 score: ', f1_score(y_test, predictions))

    # Gradient Boosting
    gb = clf(GradientBoostingClassifier(loss='deviance'))
    gb.fit(X_train, y_train)
    predictions = gb.predict(X_test)
    print('GB Accuracy score: ', accuracy_score(y_test, predictions))
    print('GB Precision score: ', precision_score(y_test, predictions))
    print('GB Recall score: ', recall_score(y_test, predictions))
    print('GB F1 score: ', f1_score(y_test, predictions))

    # K-neighbors
    kn = clf(KNeighborsClassifier())
    kn.fit(X_train, y_train)
    predictions = kn.predict(X_test)
    print('KN Accuracy score: ', accuracy_score(y_test, predictions))
    print('KN Precision score: ', precision_score(y_test, predictions))
    print('KN Recall score: ', recall_score(y_test, predictions))
    print('KN F1 score: ', f1_score(y_test, predictions))

if __name__ == "__main__":
    main()