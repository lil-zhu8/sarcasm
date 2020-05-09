#!/usr/bin/python

import random
import collections
import math
import sys
from util import *

############################################################
# Problem 3: binary classification
############################################################

############################################################
# Problem 3a: feature extraction

def extractWordFeatures(x):
    """
    Extract word features for a string x. Words are delimited by
    whitespace characters only.
    @param string x: 
    @return dict: feature vector representation of x.
    Example: "I am what I am" --> {'I': 2, 'am': 2, 'what': 1}
    """
    # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
    
    # print(dict(Counter(x.split())))
    return dict(Counter(x.split()))
    raise Exception("Not implemented yet")
    # END_YOUR_CODE

############################################################
# Problem 3b: stochastic gradient descent

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


############################################################
# Problem 3c: generate test case

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

############################################################
# Problem 3e: character features

def extractCharacterFeatures(n):
    '''
    Return a function that takes a string |x| and returns a sparse feature
    vector consisting of all n-grams of |x| without spaces mapped to their n-gram counts.
    EXAMPLE: (n = 3) "I like tacos" --> {'Ili': 1, 'lik': 1, 'ike': 1, ...
    You may assume that n >= 1.
    '''
    def extract(x):
    	#print("incoming: {}".format(x))
    	# BEGIN_YOUR_CODE (our solution is 6 lines of code, but don't worry if you deviate from this)

    	phi = collections.defaultdict(int)

    	strClean = x.replace(" ","")		# remove all spaces from the string

    	if n > len(strClean):
    		return phi
    	elif n == len(strClean):
    		phi[strClean] = 1
    		return phi
    	for i in range(0, len(strClean) - n + 1):
    		j = i + n
    		ngram = strClean[i: j]		# gets the ngram
    		phi[ngram] += 1

    	return phi
        #raise Exception("Not implemented yet")
        # END_YOUR_CODE
    return extract

############################################################
# Problem 4: k-means
############################################################

def vectorSelfProd(v1):
	selfProd = []
	for entry in v1:
		a = collections.defaultdict(float)
		for i, j in entry.items():
			a[i] = j ** 2
		selfProd.append(a)
	return selfProd

def zeroOut(c):
	for mu in c: 		# zero out the existing mu to prepare for updates
	    for key in mu:
	        mu[key] = 0
	return c

def kmeans(examples, K, maxIters):
    '''
    examples: list of examples, each example is a string-to-double dict representing a sparse vector.
    K: number of desired clusters. Assume that 0 < K <= |examples|.
    maxIters: maximum number of epochs to run (you should terminate early if the algorithm converges).
    Return: (length K list of cluster centroids,
            list of assignments (i.e. if examples[i] belongs to centers[j], then assignments[i] = j)
            final reconstruction loss)
    '''
    # BEGIN_YOUR_CODE (our solution is 25 lines of code, but don't worry if you deviate from this)

    # print("examples: {}".format(examples))

    previousMatch = []
    centers = list(subset.copy() for subset in random.sample(examples,K))
    squaredEx = vectorSelfProd(examples) # precompute the squares of the examples, only needed once per program
    
    # print("centers: {}".format(centers))
    # match each example to a cluster
    # store that as a previous match
    # update the centers
    # repeat

    for n in range(maxIters):
    	squaredCent = []
    	squaredCent = vectorSelfProd(centers)	# precompute the squares of the centroids, only needed once per iteration (aka mu update)
    	
    	distanceToCentroids = [1000 for example in examples]
    	bestCentroidMatches = [1000 for example in examples]

    	for i, example in enumerate(examples):
    		for j, mu in enumerate(centers):			# calculating a distace from each example point to each mu
    			d = -2 * dotProduct(mu, example) + sum(squaredEx[i].values()) + sum(squaredCent[j].values()) # the precomputed parts and -2ab
            	
    			if d < distanceToCentroids[i]:	# if the distance less than what's already in distances for that example, replace with new, smaller distance & update the mu match
    				distanceToCentroids[i] = d
    				bestCentroidMatches[i] = j

    	# print('distance: {}'.format(distanceToCentroids))
    	# print('matches: {}'.format(bestCentroidMatches))

    	if previousMatch == bestCentroidMatches:	# check if we're done
        	break
    	
    	else:
            previousMatch = bestCentroidMatches		# update our previous match
            groupNums = [0 for cluster in centers]	# prepare to count number of examples that fit in each cluster
            centers = zeroOut(centers)				# prepare to update centers with new mu's

            #print("blank centers: {}".format(centers))

            for i, ex in enumerate(examples):					
                groupNums[bestCentroidMatches[i]] += 1		# add 1 to the cluster to which that example belong # add to that particular mu's zeroed out value, the value of this example
                #print("cluster: {}".format(mu))
                for key, val in ex.items():					# for each key and value in the example, update the mu 
                    if key in centers[bestCentroidMatches[i]]:
                    	centers[bestCentroidMatches[i]][key] += val
            #         print("key,val: {}, {}".format(key, val))
            #         print("updated centers: {}".format(centers))
            # print("groupNums: {}".format(groupNums))

            for i, mu in enumerate(centers):	# calculate new mu's
            	#print("cluster: {}".format(mu))
            	for key in mu:
            		mu[key] /= groupNums[i]

            # print("previousMatch: {}".format(previousMatch))
            # print("centers before next iteration: {}".format(centers))

    return centers, bestCentroidMatches, sum(distanceToCentroids)
    raise Exception("Not implemented yet")
    # END_YOUR_CODE
    





