#!/usr/bin/env python
# coding: utf-8

# In[1]:


def identity_preprocess(myString):
    return myString


# In[2]:


def file_preprocess(myPath, preprocess = identity_preprocess):
    file = open(myPath, "r")
    l = list(map(preprocess, file.readlines()))
    return l


# In[3]:


import string


# In[4]:


def tokenize(s):
    s = s.strip().lower()
    s = s.translate(s.maketrans("", "" ,string.punctuation+string.digits))
    s = s.replace('\n', ' ')
    s = s.replace('\ufeff', '')
    s = s.split()
    return s


# In[5]:


from nltk.tokenize import WordPunctTokenizer

def nltk_tokenizer(s):
    l = []
    s = WordPunctTokenizer().tokenize(s)
    for i in s:
        if i.isalpha():
            l.append(i.lower())
    return l


# In[6]:


test_string = "!! Title: Moby Dick; or The Whale !!11"
l1 = nltk_tokenizer(test_string)
l2 = tokenize(test_string)
print("nltk_tokenizer: ", l1)
print("my tokenizer: ", l2)


# In[8]:


corpus=file_preprocess("/home/theoni/Documents/emp/fwni/pre_lab/corpus.txt",tokenize)


# In[9]:


def get_tokens(corpus):
    tokens = set()
    for i in corpus:
        for j in i:
            tokens.add(j)
    return list(tokens)


# In[10]:


def get_alphabet(corpus):
    alpha = set()
    for i in corpus:
        for j in i:
            for z in j:
                alpha.add(z)
    return list(alpha)


# In[11]:


tokens = get_tokens(corpus)

ab = get_alphabet(corpus)


# In[12]:


def create_syms(alphabet):
    f = open('chars.syms',"w")

    f.write('<epsilon> 0\n')
    index = 1
    for i in alphabet:
        f.write(i + " " + str(index) + "\n")
        index+=1

    f.close()


# In[13]:


create_syms(ab)


# In[14]:


lev_weights = {"d":1.0,"i":1.0,"r":1.0}


# In[15]:


def make_fst(alphabet, weights):

    for i in alphabet:
        print("0 0 %s %s %.3f" % (i,i,0))

    for i in alphabet:
        print("0 0 %s <epsilon> %.3f" % (i, weights["d"]))

    for i in alphabet:
        print("0 0 <epsilon> %s %.3f" % (i, weights["i"]))

    for i in alphabet:
        for j in alphabet:
            if i!=j:
                print("0 0 %s %s %.3f" % (i,j, weights["r"]))


# In[16]:


import os


# In[20]:


os.system("python3 lev_converter.py| fstcompile --isymbols=chars.syms --osymbols=chars.syms > lev.fst")


# In[21]:


def create_lexicon(tokens):
    n = 1
    #for each token
    for i in tokens:
        #for each character of the token
        for num, c in enumerate(i):
            n+=1
            #we begin from state 1 to state 2
            if num == 0:
                print("1 " + str(n) + " " + c + " " + c + " 0\n")
            #if it is the final character we return to state 0
            elif (num+1) == len(i):
                print(str(n-1) + " 0 " + c + " " + c + " 0\n")
            #in any other case, we crate a new state
            else:
                print(str(n-1) + " " + str(n) + " " + c + " " + c + " 0\n")
    print(0)


# In[22]:


os.system("python3 lev_acceptor.py| fstcompile --isymbols=chars.syms --osymbols=chars.syms > myFst.fst")


# In[23]:


os.system("fstrmepsilon myFst.fst | fstdeterminize | fstminimize > lexicon.fst")


# In[24]:


os.system("fstarcsort --sort_type=olabel lev.fst arcsort_lev.fst")


# In[25]:


os.system("fstarcsort --sort_type=ilabel lexicon.fst arcsort_lexicon.fst")


# In[26]:


os.system("fstcompose arcsort_lexicon.fst arcsort_lev.fst > spell_check.fst")


# In[30]:


def create_T(s):
    for index,char in enumerate(s):
        if index+1==len(s):
            string = str(index+1)+" 0 "+char+" "+char+" 0\n"
        else:
            string = str(index+1)+" "+str(index+2)+" "+char+" "+char+" 0\n"
        print(string)

    print(str(0))


# In[70]:


string1 = "python help_spell_checker.py "
string2 = "| fstcompile --isymbols=chars.syms --osymbols=chars.syms            | fstcompose spell_check.fst -            | fstshortestpath            | fstrmepsilon            | fsttopsort            | fstprint -isymbols=chars.syms            | cut -f3| grep -v '<epsilon>'            | head -n -1            | tr -d '\n'"


# In[54]:


def spell_check(s):
    print(os.popen(string1 + s + string2).read())




from gensim.models import Word2Vec

model = Word2Vec(corpus, size=100, window=5, min_count=1, workers=5)

model.train(corpus, total_examples=len(corpus), epochs=4000)

model.save("word2vec.model")
