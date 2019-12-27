#!/usr/bin/env python
# coding: utf-8

# ## ΒΗΜΑ 10: Εξαγωγή στατιστικών
#
# Στο βήμα αυτό θα κατασκευάσουμε 2 πηγές στατιστικών για τα γλωσσικά μας μοντέλα, μία __word/token level__ και μία __character level__.
#
# Θα χρειαστούμε το corpus, τα tokens και το αλφάβητο (ab) από την προπαρασκευή και για το λόγο αυτό εισάγουμε ξανά τις παρακάτω συναρτήσεις:

# In[1]:


def identity_preprocess(myString):
    return myString

def file_preprocess(myPath, preprocess = identity_preprocess):
    file = open(myPath, "r")
    l = list(map(preprocess, file.readlines()))
    return l

import string

def tokenize(s):
    s = s.strip().lower()
    s = s.translate(s.maketrans("", "" ,string.punctuation+string.digits))
    s = s.replace('\n', ' ')
    s = s.replace('\ufeff', '')
    s = s.split()
    return s

corpus = file_preprocess("./corpus.txt",tokenize)

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


# α) Θέλουμε να εξάγουμε την πιθανότητα εμφάνισης κάθε token και να την αποθηκεύσουμε σε ένα λεξικό με key το token και value την πιθανότητα αμφάνισης του. Χρησιμοποιούμε, λοιπόν, την συνάρτηση __token_prob__ η οποία δέχεται σαν όρισμα ένα corpus και κάνει το παραπάνω:

# In[2]:


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


# α) Θέλουμε να εξάγουμε την πιθανότητα εμφάνισης κάθε χαρακτήρα και να την αποθηκεύσουμε σε ένα λεξικό με key τον  χαρακτήρα και value την πιθανότητα αμφάνισης του. Χρησιμοποιούμε, λοιπόν, την συνάρτηση __ab_prob__ η οποία δέχεται σαν όρισμα ένα corpus και κάνει το παραπάνω:

# In[3]:


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


# ## ΒΗΜΑ 11: Κατασκευή μετατροπέων FST
#
# Για τη δημιουργία του ορθογράφου θα χρησιμοποιήσουμε μετατροπείς βασισμένους στην
# απόσταση Levenshtein (https://en.wikipedia.org/wiki/Levenshtein_distance). Θα
# χρησιμοποιήσουμε 3 τύπους από edits: εισαγωγές χαρακτήρων, διαγραφές χαρακτήρων και
# αντικαταστάσεις χαρακτήρων. Κάθε ένα από αυτά τα edits χαρακτηρίζεται από ένα κόστος.
#
# α) Για να υπολογίσουμε το κόστος των edits θα βρούμε την μέση τιμή των βαρών του word level μοντέλου που κατασκευάσαμε στι βήμα 10α.
#
# Θα χρησιμοποιήσουμε την συνάρτηση __mean_cost__ η οποία δέχεται σαν είσοδο ένα dictionary (όπως αυτά που δημιουργήσαμε στο βήμα 10) και επιστρέφει την μέση τιμή των values. Επειδή στην τιμή των values έχουμε πιθανότητες και εμείς θέλουμε βάρη εφαρμόζουμε τον αρνητικό λογάριθμο σε κάθε value.

# In[4]:


import math

def mean_cost(myDict):
    temp=0
    for i in myDict.values():
        temp+= math.log(i)
    return -temp/len(myDict)


# Έτσι για την μέση τιμή των βαρών του word level μοντέλου που είναι το βάρος των edit, w, έχουμε:

# In[5]:


w = mean_cost(tokenProb)


# β) Θα κατασκευάσουμε τώρα ένα μετατροπέα μίας κατάστασης που υλοποιεί την απόσταση Levenstein αντιστοιχίζοντας:
# - Κάθε χαρακτήρα στον εαυτό του με βάρος 0 (no edit),
# - Kάθε χαρακτήρα στο epsilon με βάρος w (deletion),
# - Tο epsilon σε κάθε χαρακτήρα με βάρος w (insertion),
# - Kάθε χαρακτήρα σε κάθε άλλο χαρακτήρα με βάρος w.
#
# Αυτός είναι ο μετατροπέας Levenshtein για το word level μονέλο. Για την δημιουεγία του έχουμε την παρακάτω εντολή:

# In[15]:


import os

os.system("python3 word_level_transducer.py| fstcompile --isymbols=chars.syms --osymbols=chars.syms > wlTrans.fst")


# Στο αρχείο word_level_transducer.py έχουμε τις συναρτήσεις δημιουργίας του corpus, του ab και του tokenProb όπως αναφέρθηκαν παραπάνω, και επιπλέον χρησιμοποιούμε την __make_fst__ της προπαρασκεύης (την οποία παραθέτουμε ξανά παρακάτω) ενώ έχουμε *"lev_weights = {"d":w, "i":w, "r":w}"*, όπου w το βάρος που υπολογίσαμε για τα edits.

# In[6]:


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


# γ) Επαναλαμβάνουμε τα (β), (γ) με μόνη διαφορά ότι αυτή τη φορά χρησιμοποιούμε ως βάρος των edits αυτό που προκύπτει από το unigram γλωσσικό μοντέλο του Βήματος 10β. Δηλαδή είναι:

# In[7]:


wUni=mean_cost(abProb)


# In[18]:


os.system("python3 unigram_transducer.py| fstcompile --isymbols=chars.syms --osymbols=chars.syms > uniTrans.fst")


# δ) Κάναμε μια αφελή επιλογή βαρών levestein. Ένας καλύτερος τρόπος επιλογής βαρών θα ήταν με βάση το __πόσο συχνά εμφανίζεται το εκάστοτε λάθος__. Έτσι, αν είχαμε ένα επαρκή αριθμό κείμενα γραμμένα από έναν άνθρωπο και την σωστή τους έκδoση τους θα μπορούσαμε να υπολογίσουμε μέσω του τύπου του Bayes την συχνότητα και κατ' επέκταση την πιθανότητα εμφάνισης κάθε λάθους . Στην συνέχεια, παίρνοντας τον αρνητικό λογάριθμο μετατρέπουμε κάθε μία από τις προηγούμενες πιθανότητες σε κόστη και άρα έχουμε πλέον τα βάρη για κάθε σύμβολο τόσο στο deletion όσο και στο insertion και replacement.
#
# ## BHMA 12: Κατασκευή γλωσσικών μοντέλων
#
# α) Θα κατασκευάσουμε έναν αποδοχέα μίας αρχικής κατάστασης που αποδέχεται κάθε λέξη του λεξικού που φτάξαμε στο βήμα 3α. Στο βήμα αυτό όμως θα χρησιμοποιήσουμε τον αρνητικό λογάριθμο της πιθανότητας εμφάνισης κάθε λέξης. Επίσης, η πρώτη ακμή εχει βάρος -logP(w) και οι υπόλοιπες 0. Χρησιμοποιούμε, λοιπόν, την παρακάτω συνάρτηση:

# In[19]:


def word_level_acceptor(token):
    n=1

    for i in token.keys():
        for num, j in enumerate(i):
            if num==0:
                print("1 %d %s %s %f" % (n+1, j, j, -math.log(token[i])))
            elif num==len(i)-1:
                print("%d 0 %s %s 0" % (n, j, j))
            else:
                print("%d %d %s %s 0" % (n, n+1, j, j))
            n+=1
    print(0)


# Και για την δημιουργία του αποδοχέα έχουμε:

# In[20]:


os.system("python3 word_level_acceptor.py| fstcompile --isymbols=chars.syms --osymbols=chars.syms > wlAcc.fst")


# β) Καλούμε τις fstrmepsilon, fstdeterminize και fstminimize για την βελτιστοποίηση του μονελου και τελικά παίρνουμε τον αποδοχέα στο αρχείο __lexicon2.fst__.

# In[21]:


os.system("fstrmepsilon wlAcc.fst | fstdeterminize | fstminimize > lexicon2.fst")


# γ) Επναλαμβάνουμε τα (α) και (β) για το Unigram γλωσσικό μοντέλο.
#
# Έστι έχουμε την συνάρτηση __unigram_acceptor__:

# In[8]:


def unigram_acceptor(token, ab_prob):
    n=1
    for i in token:
        cost=0
        for j in i:
            cost+=-math.log(ab_prob[j])
        for num, j in enumerate(i):
            if num==0:
                print("1 %d %s %s %f" % (n+1,j,j,cost))
            elif num==len(i)-1:
                print("%d 0 %s %s 0" % (n,j,j))
            else:
                print("%d %d %s %s 0" % (n,n+1,j,j))
            n+=1
    print(0)


# Και δημιουργούμε τον αποδοχέα μας στον αρχείο __lexicon_uni2.fst__

# In[23]:


os.system("python3 unigram_acceptor.py| fstcompile --isymbols=chars.syms --osymbols=chars.syms > uniAcc.fst")

os.system("fstrmepsilon uniAcc.fst | fstdeterminize | fstminimize > lexicon_uni2.fst")


# ## ΒΗΜΑ 13: Κατασκευή ορθογράφων
#
# α) Όπως και στο βήμα 7, κατασκευάζουμε έναν ορθογράφο  με το word-level γλωσσικό μοντέλο και το word-level μετατροπέα:

# In[24]:


os.system("fstarcsort --sort_type=olabel wlTrans.fst arcsort_wlTrans.fst")
os.system("fstarcsort --sort_type=ilabel lexicon2.fst arcsort_lexicon2.fst")


# In[25]:


os.system("fstcompose arcsort_lexicon2.fst arcsort_wlTrans.fst > spell_check2.fst")


# Για να μορούμε τώρα να ελέγξουμε την έξοδο του ορθρογράφου σε κάποια είσοδο φτιάχνουμε την συνάρτηση __spell_check2__ ως εξής:

# In[21]:


string3 = "python help_spell_checker.py "
string4 = "| fstcompile --isymbols=chars.syms --osymbols=chars.syms            | fstcompose spell_check2.fst -            | fstshortestpath            | fstrmepsilon            | fsttopsort            | fstprint -isymbols=chars.syms            | cut -f3| grep -v '<epsilon>'            | head -n -1            | tr -d '\n'"

def spell_check2(s):
    return os.popen(string3 + s + string4).read()


# Έτσι για την λέξη __cit__ έχουμε:

# In[30]:


spell_check2("cit")


# β) Αντίστοιχα με το βήμα 7, κατασκευάζουμε τον ορθογράφο με το unigram γλωσσικό μοντέλο και το word-level μετατροπέα:

# In[31]:


os.system("fstarcsort --sort_type=ilabel lexicon_uni2.fst arcsort_lexicon_uni2.fst")


# In[32]:


os.system("fstcompose arcsort_lexicon_uni2.fst arcsort_wlTrans.fst > spell_check_uni2.fst")


# Για να μορούμε τώρα να ελέγξουμε την έξοδο του ορθρογράφου σε κάποια είσοδο φτιάχνουμε την συνάρτηση __spell_check_uni2__ ως εξής:

# In[18]:


string5 = "python help_spell_checker.py "
string6 = "| fstcompile --isymbols=chars.syms --osymbols=chars.syms            | fstcompose spell_check_uni2.fst -            | fstshortestpath            | fstrmepsilon            | fsttopsort            | fstprint -isymbols=chars.syms            | cut -f3| grep -v '<epsilon>'            | head -n -1            | tr -d '\n'"

def spell_check_uni2(s):
    return os.popen(string5 + s + string6).read()


# Έτσι για την λέξη __cit__ έχουμε:

# In[19]:


a=spell_check_uni2("cit")


# γ) Η διαφορά των 2 ορθογράφων βρίσκεται στο γλωσσικό μοντέλο που χρησιμοποιούν. Πιο συγκεκριμένα:
#
# - Στον πρώτο ορθογράφο που χρησιμοποιεί το __word level γλωσσικό μοντέλο__, η διόρθωση γίνεται με βάση την συχνότητα εμφάνισης κάθε λέξης. Έτσι, διορθώνει μία λεη σε μία άλλη η οποία είναι πιο πιθανό να έχει εμφανιστεί κοιτώντας παράλληλα να επιτύχει ελάχιστον αριθμό edits.
# - Στον δεύτερο ορθογράφο που χρησιμοποιεί το __unigram γλωσσικό μοντέλο__, η διόρθωση γίνεται με βάση τη συχνότητα εμφάνισης του κάθε γράμματος. Έτσι, διορθώνει ένα γράμμα κάποιας λέξης σε κάποιο άλλο το οποίο είναι πιο πιθανό να έχει εμφανιστεί στην διορθωμένη λέξη, κοιτώντας παράλληλα να επιτύχει ελάχιστον αριθμό edits.
#
# Για παράδειγμα για την λέξη __'wok'__ έχουμε διαφορετικά αποτελέσματα από τους δύο ορθογράφους.

# In[15]:


spell_check2("wok")
spell_check_uni2("wok")


# ## ΒΗΜΑ 14: Αξιολόγηση των ορθογράφων
#
# α) Κατεβάζουμε το σύνολο δεδομένων που βρίσκουμε στο https://raw.githubusercontent.com/georgepar/python-lab/master/spell_checker_test_set και το αποθηκεύμουμε με το όνομα __test.txt__.
#
# β) Για να κάνουμε αξιολόγηση των ορθογράφων γράφουμε την παρακάτω συνάρτηση η οποία επιλέγει 20 τυχαίες λέξεις από το test.txt και βρίσκει το ποσοστό των λέξεων που βρίσκει σωστά ο ορθογράφος:

# In[38]:


import numpy as np

def test(text, func):
    f = open(text, "r")
    total=0
    correct=0
    lines = f.readlines()
    for i in range(20):
        rnd = np.random.randint(0, 121)
        words = tokenize(lines[rnd])
        lines.pop(rnd)
        for j in words[1:]:
            predict=func(j)
            if predict==words[0]:
                correct+=1
            total+=1
    return correct/total



# In[39]:


test("test.txt", spell_check2)


# In[40]:


test("test.txt", spell_check_uni2)


# __*Σχολιασμός αποτελεσμάτων*__
#
# Για τον word level ορθογράφο βρίσκουμε ότι προβλέπει σωστά το __84%__ των λέξεων, ενώ για το unigram έχουμε σωστό το __47,36%__.

# ## Βήμα 15: Bigram ορθογράφος (Extra Credit)
#
# Στο βήμα αυτό θέλουμε να φτιάξουμε έναν ορθογράφο που χρησιμοποιεί το bigram γλωσσικό μοντέλο για την εξαγωγή αποτελεσμάτων.
#
# Αρχικά χρειάζεται να φτιάξουμε έναν bigram αποδοχέα. Για το σκοπό αυτό χρησιμοποιούμε την συνάρτηση bigram_acceptor:

# In[49]:


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


# Στην συνέχεια ακολοθούμε τα βήματα που ακολουθήσαμε και στην δημιουργία των προγούμενων ορθογράφων:

# In[44]:


os.system("python3 bigram_acceptor.py| fstcompile --isymbols=chars.syms --osymbols=chars.syms > biAcc.fst")

os.system("fstrmepsilon biAcc.fst | fstdeterminize | fstminimize > lexicon_v3.fst")

os.system("fstarcsort --sort_type=olabel wlTrans.fst arcsort_wlTrans.fst")

os.system("fstarcsort --sort_type=ilabel lexicon_v3.fst arcsort_lexicon_v3.fst")

os.system("fstcompose arcsort_lexicon_v3.fst arcsort_wlTrans.fst > spell_check_v3.fst")


# In[47]:


string7 = "python help_spell_checker.py "
string8 = "| fstcompile --isymbols=chars.syms --osymbols=chars.syms            | fstcompose spell_check_v3.fst -            | fstshortestpath            | fstrmepsilon            | fsttopsort            | fstprint -isymbols=chars.syms            | cut -f3| grep -v '<epsilon>'            | head -n -1            | tr -d '\n'"

def spell_check_bi(s):
    return os.popen(string7 + s + string8).read()


# Αξιολογούμε τα αποτελέσματα του bigram ορθογράφου με την συνάρτηση test και παίρνουμε ποσοστό __62,79%__ σωστών προβλέψεων.

# In[48]:


test("test.txt", spell_check_bi)
