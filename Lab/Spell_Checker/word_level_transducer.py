import math
import string
import os

def identity_preprocess(myString):
    return myString

def file_preprocess(myPath, preprocess=identity_preprocess):
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

corpus = file_preprocess("./corpus.txt", tokenize)

def get_alphabet(corpus):
    alpha=set()
    for i in corpus:
        for j in i:
            for z in j:
                alpha.add(z)
    return alpha

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

def mean_cost(myDict):
    temp=0
    for i in myDict.values():
        temp+= math.log(i)
    return -temp/len(myDict)

w = mean_cost(tokenProb)

lev_weights = {"d":w, "i":w, "r":w}

def make_fst(alphabet, weights):

    for i in alphabet:
        print("0 0 %s %s %.4f" % (i,i,0))

    for i in alphabet:
        print("0 0 %s <epsilon> %f" % (i,weights["d"]))

    for i in alphabet:
        print("0 0 <epsilon> %s %f" % (i,weights["i"]))

    for i in alphabet:
        for j in alphabet:
            if i!=j:
                print("0 0 %s %s %f" % (i,j,weights["r"]))
    print(0)

make_fst(ab, lev_weights)
