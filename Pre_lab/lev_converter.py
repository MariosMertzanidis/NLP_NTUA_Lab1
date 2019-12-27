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

corpus=file_preprocess("/home/theoni/Documents/emp/fwni/corpus.txt",tokenize)

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

def create_syms(alphabet):
    f = open('chars.syms',"w")

    f.write('<epsilon> 0\n')
    n=1
    for i in alphabet:
        f.write(i+" "+str(n)+"\n")
        n+=1

    f.close()

#create_syms(ab)

lev_weights = {"d":1.0,"i":1.0,"r":1.0}


def make_fst(alphabet,weights):

    for i in alphabet:
        print("0 0 %s %s %.3f" % (i,i,0))


    for i in alphabet:
        print("0 0 %s <epsilon> %.3f" % (i,weights["d"]))

    for i in alphabet:
        print("0 0 <epsilon> %s %.3f" % (i,weights["i"]))

    for i in alphabet:
        for j in alphabet:
            if i!=j:
                print("0 0 %s %s %.3f" % (i,j,weights["r"]))
    print(0)


make_fst(ab,lev_weights)
