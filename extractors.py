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



