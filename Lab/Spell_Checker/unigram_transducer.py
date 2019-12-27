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

corpus=file_preprocess("./corpus.txt",tokenize)

def get_alphabet(corpus):
    alpha=set()
    for i in corpus:
        for j in i:
            for z in j:
                alpha.add(z)
    return alpha

ab = list(get_alphabet(corpus))

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

tokenProb= ab_prob(corpus)

def mean_cost(myDict):
    temp=0
    for i in myDict.values():
        temp+= math.log(i)
    return -temp/len(myDict)

wUni = mean_cost(tokenProb)

lev_weights = {"d":wUni,"i":wUni,"r":wUni}

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


make_fst(ab,lev_weights)
