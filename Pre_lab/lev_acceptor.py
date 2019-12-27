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

def create_lexicon(tok):
    n=1
    for i in tok:
        for num,c in enumerate(i):
            n+=1
            if num==0:
                print("1 "+str(n)+" "+c+" "+c+" 0\n")
            elif (num+1)==len(i):
                print(str(n-1)+" 0 "+c+" "+c+" 0\n")
            else:
                print(str(n-1)+" "+str(n)+" "+c+" "+c+" 0\n")
    print(0)


create_lexicon(tokens)
