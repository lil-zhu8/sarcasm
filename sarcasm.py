#!/usr/bin/python

import random
import collections
import math
import sys
from util import *

import string

def extractWordFeatures(x):
    """
    Example: "I am what I am" --> {'I': 2, 'am': 2, 'what': 1}
    """
    return dict(Counter(x.split()))


def extractCharacterFeatures(n):
    '''
    EXAMPLE: (n = 3) "I like tacos" --> {'Ili': 1, 'lik': 1, 'ike': 1, ...
    '''
    def extract(x):
        phi = collections.defaultdict(int)

        strClean = x.replace(" ","")       

        if n > len(strClean):
            return phi
        elif n == len(strClean):
            phi[strClean] = 1
            return phi
        for i in range(0, len(strClean) - n + 1):
            j = i + n
            ngram = strClean[i: j]      # gets the ngram
            phi[ngram] += 1

        return phi
    return extract


def extractPunctuationFeatures(n):
    '''
    EXAMPLE: (n = 3) "Hello! My name is bob. " --> {'!': 1, '.': 1}
    '''
    def extract(x):
        phi = collections.defaultdict(int)
        puncts = string.punctuation

        strClean = x.replace(" ","")       

        if n > len(strClean):
            return phi
        elif n == len(strClean):
            phi[strClean] = 1
            return phi
        for i in range(len(strClean)):
            if strClean[i] in string.punctuation: 
            phi[strClean[i]] += 1

        return phi
    return extract


def extractCapitalizationFeatures(n):
    '''
    EXAMPLE: "Hello! My name is bob. " --> {'H': 1, 'M': 1}
    '''
    def extract(x):
        phi = collections.defaultdict(int)

        strClean = x.replace(" ","")       

        if n > len(strClean):
            return phi
        elif n == len(strClean):
            phi[strClean] = 1
            return phi
        for i in range(len(strClean)):
            if strClean[i].isUpper(): 
            phi[strClean[i]] += 1

        return phi
    return extract


def extractElongatedWordFeatures(x):
    '''
    EXAMPLE: "Oh that's reallllly goooodd. " --> {('really','l'): 1, ('good','o'): 1}
    '''

    phi = []
    # strips punctuation, then splits by word
    strClean = x.translate(None, string.punctuation).split()     

    for word in range(len(strClean)):
        if len(word) > 5:
            for i in range(1, len(word) - 1):
                charBefore = word[i-1]
                charCurr = word[i]
                charAfter = word[i+1]

            # making an assumption that we won't have multiple of the same words in a sentence with this 
            # type of feature (3+ chars in a row)
            if charBefore == charCurr and charCurr == charAfter:
                # 3 characters in a row are the same
                phi.append((word, charCurr), 1)

    return phi



def learnPredictor(trainExamples, validationExamples, featureExtractor, numIters, eta):
    '''
    Given |trainExamples| and |validationExamples| (each one is a list of (x,y)
    pairs), a |featureExtractor| to apply to x, and the number of epochs to
    train |numIters|, the step size |eta|, return the weight vector (sparse
    feature vector) learned.

    You should implement stochastic gradient descent.

    Note: only use the trainExamples for training!
    You should call evaluatePredictor() on both trainExamples and validationExamples
    to see how you're doing as you learn after each epoch. Note also that the 
    identity function may be used as the featureExtractor function during testing.
    '''
    #print('featureExtractor: {}'.format(featureExtractor))
    weights = {}  # feature => weight
    # BEGIN_YOUR_CODE (our solution is 12 lines of code, but don't worry if you deviate from this)
    def predictor(input):

    	phi_x = featureExtractor(input)
    	k = dotProduct(weights, phi_x)

    	if k < 0: return -1
    	else: return 1
    
    for i in range(numIters):						# each epoch???
        for ex in trainExamples:
        	phi_x = featureExtractor(ex[0])
        	k = dotProduct(weights, phi_x)
        	
        	#print("phi type: {}".format(type(phi_x)))
        	# print("phi_x: {}".format(phi_x))

        	if k * ex[1] < 1: 	# hinge loss
        		#print("old weights: {}".format(weights))
        		increment(weights, eta * ex[1], phi_x)	# update weights
        		#print("new weights: {}".format(weights))
    # END_YOUR_CODE
    return weights



def generateDataset(numExamples, weights):
    '''
    Return a set of examples (phi(x), y) randomly which are classified correctly by
    |weights|.
    '''
    random.seed(42)
    # Return a single example (phi(x), y).
    # phi(x) should be a dict whose keys are a subset of the keys in weights
    # and values can be anything (randomize!) with a nonzero score for the given weight vector.
    # y should be 1 or -1 as classified by the weight vector.

    # Note that the weight vector can be arbitrary during testing. 
    def generateExample():
        # BEGIN_YOUR_CODE (our solution is 2 lines of code, but don't worry if you deviate from this)
        a = random.randint(1, len(list(weights)))		# random size of subset
        phi = {}

        for i in range(a):
        	b = random.randint(1, 500)				# random value
        	key = random.sample(list(weights), 1)	# grab a random key from weights
        	phi[key[0]] = b							# extract just the key and not a list
        	
        	# print('item: {}'.format(key))
        	# print('phi[item]: {}'.format(b))
        
        if dotProduct(weights, phi) > 1:
        	y = 1
        else:
        	y = -1

        # raise Exception("Not implemented yet")
        # # END_YOUR_CODE
        return (phi, y)

    return [generateExample() for _ in range(numExamples)]






