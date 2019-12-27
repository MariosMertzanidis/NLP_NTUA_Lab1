import math
import string
import os

def identity_preprocess(myString):
    return myString

def file_preprocess(myPath,preprocess=identity_preprocess):
    file = open(myPath,"r")
    l=list(map(preprocess,file.readlines()))
    return l

def tokenize(s):
    s=s.strip().lower()
    s=s.translate(s.maketrans("","",string.punctuation+string.digits))
    s=s.replace('\n',' ')
    s=s.replace('\ufeff','')
    s=s.split()
    return s

corpus=file_preprocess("./corpus.txt", tokenize)

def get_tokens(corpus):
    tokens=set()
    for i in corpus:
        for j in i:
            tokens.add(j)
    return tokens

def get_alphabet(corpus):
    alpha=set()
    for i in corpus:
        for j in i:
            for z in j:
                alpha.add(z)
    return alpha

tokens = list(get_tokens(corpus))

ab = list(get_alphabet(corpus))

def token_prob(corpus):
    wordFreq={}
    totalWords=0
    for i in corpus:
        for j in i:
            if j in wordFreq.keys():
                wordFreq[j]+=1
            else:
                wordFreq[j]=1
            totalWords+=1
    for i in wordFreq.keys():
        wordFreq[i]=wordFreq[i]/totalWords
    return wordFreq

tokenProb = token_prob(corpus)

def word_level_acceptor(token):
    n=1

    for i in token.keys():
        for num, j in enumerate(i):
            if num==0:
                print("1 %d %s %s %f" % (n+1,j,j,-math.log(token[i])))
            elif num==len(i)-1:
                print("%d 0 %s %s 0" % (n,j,j))
            else:
                print("%d %d %s %s 0" % (n,n+1,j,j))
            n+=1
    print(0)

word_level_acceptor(tokenProb)
