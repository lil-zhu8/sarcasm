import json
import sys
import getopt
import os
import math
import operator
from nltk.tokenize import TweetTokenizer 
from collections import defaultdict
import random


# reads items in json file as a list of dicts, where the entries are "article_link", "headline", "is_sarcastic"
def read_headlines(filename = 'data/headlines/Sarcasm_Headlines_Dataset.json'):
	artList = []
	with open(filename) as f:
	    for jsonObj in f:
	        art = json.loads(jsonObj)
	        artList.append(art)
	return artList

def naive_headlines(articles):
	artlist = {}
	for art in articles:
		artlist[art[headline]] = art[is_sarcastic]
	return artlist

# extract tweets as dict mapping tweet to sarcasm = 1, not sarcasm = 0 
def read_tweets(filename = 'data/twitter/sarcasm-dataset.txt'):
	tweetlist = {}
	with open(filename, 'r') as fp:
		for twt in fp:
	  		tweetlist[twt[0:len(twt)-2].strip()] = twt[-2]
	return tweetlist

class NaiveBayes:
    class tsplit:
        """Represents a set of training/testing data. self.train is a list of Examples, as is self.dev and self.test. 
        """
        def __init__(self):
            self.train = []
            self.dev = []
            self.test = []

    class instance:
        """Represents a document with a label. klass is 'sarcastic' or 'not' by convention.
             words is a list of strings.
        """
        def __init__(self):
            self.klass = ''
            self.words = []

    def __init__(self):
        """NaiveBayes initialization"""
        self.USE_BIGRAMS = False
        self.BEST_MODEL = False

        self.not_docs = 0
        self.sarcasm_docs = 0
        self.total_docs = 0

        # unigrams
        self.words_sarcasm = defaultdict(int)
        self.words_not = defaultdict(int)
        self.vocab = set()
        self.count_sarcasm = 0
        self.count_not = 0

        # bigrams
        self.bi_vocab = set()
        self.bigrams_sarcasm = defaultdict(int)
        self.bigrams_not = defaultdict(int)
        self.count_sarcasm_bigram = 0
        self.count_not_bigram = 0

    def buildSplit(self):
        split = self.tsplit()
        k = ['not','sarcasm']
        data = read_tweets()
        d = list(data.keys())
        s = list(data.values())
        tt = TweetTokenizer()
        for i in range(len(data)):
            inst = self.instance()
            inst.words = tt.tokenize(d[i])
            inst.klass = k[int(s[i])]
            if random.random() < 0.7:
                split.train.append(inst)
            else:
                split.dev.append(inst)
        return split

    def classify(self, words):
        
        if(self.USE_BIGRAMS):
            words.insert(0,'<s>')
            words.append('</s>')

            pr_sarcasm = pr_not = 0
            if not self.sarcasm_docs == 0:
                pr_sarcasm += math.log(self.sarcasm_docs) - math.log(self.total_docs)
            if not self.not_docs == 0:
                pr_not = math.log(self.not_docs) - math.log(self.total_docs)

            sarcasm_denom = self.count_sarcasm_bigram + len(self.bi_vocab)
            not_denom = self.count_not_bigram + len(self.bi_vocab)

            for fword,sword in zip(words[:-1],words[1:]):
                if (fword,sword) in self.bi_vocab:
                    pr_sarcasm += math.log((self.bigrams_sarcasm[(fword,sword)] + 1)) - math.log(sarcasm_denom)
                    pr_not += math.log((self.bigrams_not[(fword,sword)] + 1)) - math.log(not_denom)

            if pr_sarcasm > pr_not:
                return 'sarcasm'
            return 'not'

        else:
            pr_sarcasm = pr_not = 0
            if not self.sarcasm_docs == 0:
                pr_sarcasm += math.log(self.sarcasm_docs) - math.log(self.total_docs)
            if not self.not_docs == 0:
                pr_not += math.log(self.not_docs) - math.log(self.total_docs)

            sarcasm_denom = self.count_sarcasm + len(self.vocab)
            not_denom = self.count_not + len(self.vocab)

            for word in words:
                if word in self.vocab:
                    pr_sarcasm += math.log((self.words_sarcasm[word] + 1)) - math.log(sarcasm_denom)
                    pr_not += math.log((self.words_not[word] + 1)) - math.log(not_denom)

            if pr_sarcasm > pr_not:
                return 'sarcasm'
            return 'not'

    def addExample(self, klass, words):
        """
         * TODO
         * Train your model on an example document with label klass ('sarcasm' or 'not') and
         * words, a list of strings.
         * You should store whatever data structures you use for your classifier 
         * in the NaiveBayes class.
         * Returns nothing
        """
        if klass == 'sarcasm':
            self.sarcasm_docs += 1
        else:
            self.not_docs += 1
        self.total_docs += 1

        if(self.USE_BIGRAMS):
            words.insert(0,'<s>')
            words.append('</s>')
            
            for fword,sword in zip(words[:-1],words[1:]):
                if klass == 'sarcasm':
                    self.count_sarcasm_bigram += 1
                    self.bigrams_sarcasm[(fword,sword)] += 1
                else:
                    self.count_not_bigram += 1
                    self.bigrams_not[(fword,sword)] += 1
                self.bi_vocab.add((fword,sword))

        else:
            for word in words:
                if klass == 'sarcasm':
                    self.count_sarcasm += 1
                    self.words_sarcasm[word] += 1
                else:
                    self.count_not += 1
                    self.words_not[word] += 1
                self.vocab.add(word)

def evaluate(USE_BIGRAMS):
    classifier = NaiveBayes()
    classifier.USE_BIGRAMS = USE_BIGRAMS
    split = classifier.buildSplit()
   
    for example in split.train:
        classifier.addExample(example.klass,example.words)

    train_accuracy = calculate_accuracy(split.train,classifier)
    dev_accuracy = calculate_accuracy(split.dev,classifier)

    print('Train Accuracy: {}'.format(train_accuracy))
    print('Dev Accuracy: {}'.format(dev_accuracy))


def calculate_accuracy(dataset,classifier):
    acc = 0.0
    if len(dataset) == 0:
        return 0.0
    else:
        for example in dataset:
            guess = classifier.classify(example.words)
            if example.klass == guess:
                acc += 1.0
        return acc / len(dataset)

def main():
	USE_BIGRAMS = False
	evaluate(USE_BIGRAMS)

if __name__ == "__main__":
        main()
