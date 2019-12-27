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

corpus=file_preprocess("/home/theoni/Documents/emp/nlp/pre_lab/corpus.txt",tokenize)

def get_tokens(corpus):
    tokens=set()
    for i in corpus:
        for j in i:
            tokens.add(j)
    return tokens

tokens = list(get_tokens(corpus))

def bigram_prob(corpus):
    biFreq={}
    total=0
    for i in corpus:
        for j in i:
            for n,z in enumerate(j):
                if n==0:
                    prev=z
                else:
                    if prev+z in biFreq:
                        biFreq[prev+z]+=1
                    else:
                        biFreq[prev+z]=1
                    prev=z
                    total+=1
    for i in biFreq.keys():
        biFreq[i]/=total
    return biFreq

def ab_prob(corpus):
    abFreq={}
    totalAb=0
    for i in corpus:
        for j in i:
            for ab in j:
                if ab in abFreq.keys():
                    abFreq[ab]+=1
                else:
                    abFreq[ab]=1
                totalAb+=1
    for i in abFreq.keys():
        abFreq[i]=abFreq[i]/totalAb
    return abFreq

abProb = ab_prob(corpus)

biProb = bigram_prob(corpus)

def bigram_acceptor(token,bi_prob,ab_prob):
    n=1
    for i in token:
        cost=0
        for num,j in enumerate(i):
            if num==0:
                prev=j
            else:
                cost+=-math.log(bi_prob[prev+j]/ab_prob[prev])
                prev=j
        for num, j in enumerate(i):
            if num==0:
                print("1 %d %s %s %f" % (n+1,j,j,cost))
            elif num==len(i)-1:
                print("%d 0 %s %s 0" % (n,j,j))
            else:
                print("%d %d %s %s 0" % (n,n+1,j,j))
            n+=1
    print(0)


bigram_acceptor(tokens,biProb,abProb)
