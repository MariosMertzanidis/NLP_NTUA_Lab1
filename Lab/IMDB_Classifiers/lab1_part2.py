#!/usr/bin/env python
# coding: utf-8

# ### ΒΗΜΑ 16: Δεδομένα και προεπεξεργασία
# α) Αρχικά κατεβάζουμε τα δεδομένα από το επόμενο link: http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz και το αποσιμπιέζουμε αποθηκεύοντας το σε φάκελο με το όνομα aclImdb_v1.
#
# Από τα περιεχόμενα του αρχείου αυτά που μας ενδιαφέρουν είναι:
#
# 1) Τα **train data** τα οποία περιέχουν:
#   - __pos data__: είναι οι θετικές κριτικές -
#   - __neg data__: είναι οι αρνητικές κριτικές -
#
# 2) Τα **test data** τα οποία με την σειρά τους περιέχουν:
#   - __pos data__: είναι οι θετικές κριτικές -
#   - __neg data__: είναι οι αρνητικές κριτικές -
#
# β) Για το διάβασμα και την προεπεξεργασία των δεδομένων θα χρησιμοποιήσουμε τον έτοιμο κώδικα που δίνεται στο link https://gist.github.com/georgepar/eba00343e7ddc995898a7f075dcfc445 και φαίνεται παρακάτω:

# - Αρχικά δίνουμε στα directories που θα χρησιμοποιήσουμε,  όπως και σε κάποιες άλλες μεταβλητές, ονόματα για να είναι ευκολότερο στην χρήση.

# In[1]:


import os

data_dir = './aclImdb_v1/aclImdb'
train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'test')
pos_train_dir = os.path.join(train_dir, 'pos')
neg_train_dir = os.path.join(train_dir, 'neg')
pos_test_dir = os.path.join(test_dir, 'pos')
neg_test_dir = os.path.join(test_dir, 'neg')

# For memory limitations. These parameters fit in 8GB of RAM.
# If you have 16G of RAM you can experiment with the full dataset / W2V
MAX_NUM_SAMPLES = 5000
# Load first 1M word embeddings. This works because GoogleNews are roughly
# sorted from most frequent to least frequent.
# It may yield much worse results for other embeddings corpora
NUM_W2V_TO_LOAD = 1000000

import numpy as np

SEED = 42

# Fix numpy random seed for reproducibility
np.random.seed(42)

try:
    import glob2 as glob
except ImportError:
    import glob

import re


# - Η __strip_panctuation__ δέχεται ως είσοδο ένα string και αντικαθιστά κάθε σύμβολο του (εκτός από τα '\n' κλπ) που δεν είναι γράμμα με το κενό. Έτσι επιστρέφει ένα string που αποτελείται μόνο από πεζούς και κεφαλαίους αλφαριθμητικούς χαρακτήρες.

# In[2]:


def strip_punctuation(s):
    return re.sub(r'[^a-zA-Z\s]', ' ', s)


# - Η __preprocess__ δέχεται ως είσοδο ένα string και απαλοίφει από αυτό τα σημεία στήξης με την strip_punctuation ενώ επιπλέον μετατρέπει όλα τα γράμματα σε πεζά και αντικαθιστά όλα τα συνεχόμενα κενά με ένα μόνο κενό.

# In[3]:


def preprocess(s):
    return re.sub('\s+',' ', strip_punctuation(s).lower())


# - Η __tokenize__ δέχεται σαν είσοδο ένα string και το διασπάει στις λέξεις από τις οποίες αποτελείται με βάση την θέση των κενών, επιστρέφοντας μία λίστα από τις λέξεις του αρχικού string.

# In[4]:


def tokenize(s):
    return s.split(' ')


# - Η __preproc_tok__ δέχεται ένα string και επιστρέφει μία λίστα από tokens.

# In[5]:


def preproc_tok(s):
    return tokenize(preprocess(s))


# - Η __read_samples__ δέχεται σαν ορίσματα ένα path που περιέχει τα samples και μία συνάρτηση preprocess (η οποία έχει σαν default μία συνάρτηση που επιστρέφει το όρισμά της όπως είναι). Για κάθε ένα από τα samples που περιέχονται στο path και είναι σε μορφή .txt καλεί την preprocess και την εφαρμόζει σε αυτά. Τέλος, επιστρέφει το data το οποίο είναι μία λίστα της οποίας κάθε στοιχείο είναι το αποτέλεσμα της preprocess για κάθε κριτική(samples).

# In[6]:


def read_samples(folder, preprocess=lambda x: x):
    samples = glob.iglob(os.path.join(folder, '*.txt'))
    data = []
    for i, sample in enumerate(samples):
        if MAX_NUM_SAMPLES > 0 and i == MAX_NUM_SAMPLES:
            break
        with open(sample, 'r') as fd:
            x = [preprocess(l) for l in fd][0]
            data.append(x)
    return data


# - H __create_corpus__ δέχεται σαν ορίσματα δύο λίστες από τις οποίες η πρώτη περιέχει θετικές κριτικές και η δεύτερη αρνητικές. Επιστρέφει μία λίστα που περιέχει τις δοσμένες κριτικές με τυχαία σειρά και μία λίστα που περιέχει το label την κάθε κριτικής στην αντίστοιχη σειρά.

# In[7]:


def create_corpus(pos, neg):
    corpus = np.array(pos + neg)
    y = np.array([1 for _ in pos] + [0 for _ in neg])
    indices = np.arange(y.shape[0])
    np.random.shuffle(indices)
    return list(corpus[indices]), list(y[indices])


# Στο σημείο αυτό θα διαβάσουμε τα δεδομένα μας. Δημιουργούμε, λοιπόν, τις εξής λίστες:
# - __imdbCorpus__: Στην λίστα αυτή θα περιέχονται όλες οι κριτικές που θα χρησιμοποιήσουμε για το train του μοντέλου μας.
# - __Y_train__: Στην λίστα αυτή θα περιέχονται τα labels των κριτικών που θα χρησιμοποιηθούν για το train του μοντέλου.
# - __testCorpus__: Στην λίστα αυτή θα περιέχονται όλες οι κριτικές που θα χρησιμοποιήσουμε για το test του μοντέλου μας.
# - __Y_test__: Στην λίστα αυτή θα περιέχονται τα labels των κριτικών που θα χρησιμοποιηθούν για το test του μοντέλου.
#
# Έτσι έχουμε τον επόμενο κώδικα:

# In[8]:


posData = read_samples(pos_train_dir)
negData = read_samples(neg_train_dir)

imdbCorpus, Y_train = create_corpus(posData, negData)

posTest = read_samples(pos_test_dir)
negTest = read_samples(neg_test_dir)

testCorpus, Y_test = create_corpus(posTest, negTest)


# ### ΒΗΜΑ 17: Κατασκευή BOW αναπαραστάσεων και ταξινόμηση
# α) Στην __Bag Of Words__ αναπαράσταση υπολογίζουμε το πόσες φορές υπάρχει κάποια λέξη στην κάθε κριτική με αποτέλεσμα να προκύπτει ένας μεγάλος και αραιός πίνακας (μεγέθους ίσου με το μέγεθος του λεξικού). Η αναπαράσταση αυτή παρουσιαάζει δύο βασικά μειονεκτήματα τα οποία αντιμετωπίζονται με την χρήση __TF_IDF__. Πιο συγκεκριμένα έχουμε:
# - Το μέγεθος της κριτικής δεν λαμβάνεται υπόψιν με αποτέλεσμα σε μία κριτική μικρού μεγέθους κάποιες λέξεις έχουν σημαντικότερο ρόλο να μην αντιμετωπίζονται με την σωστή βαρύτητα. Αντίθετα, στην TF_IDF αναπαράσταση. αφού υπολογίσουμε το πόσες φορές υπάρχει μία λέξη στην πρόταση διαιρούμε με το συνολικό πλήθος των λέξεων και άρα δεν παρουσιάζεται το παραπάνω πρόβλημα. Αυτό είναι το __term frequency__.
# - Ευρέως χρησιμοποιούμενες λέξεις, όπως πχ το *a, and, ...*, παίζουν σημαντικό ρόλο λόγω του πλήθους του χωρίς όμως να προσθέτουν χρήσιμο νόημα, ενώ σπανιότερες λέξεις που μπορεί να είναι πιο σημαντικές δεν έχουν αντίστοιχη βαρύτητα. To πρόβλημα αυτό λύνεται στην TF_IDF αναπαράσταση μέσω του όρου __inverse document frequency__. Σύμφωνα με αυτόν ο συνολικός αριθμός των κριτικών διαιρείται με τον αριθμό των κριτικών στις οποίες βρίσκεται η λέξη που μας ενδιαφέρει και το inverse document frequency αυξάνεται όσο πιο σπάνια είναι κάποια λέξη.

# β) Θα χρησιμοποιήσουμε τον __transformer CountVectorizer__ του sklearn για να εξάγουμε μην σταθμισμένες BOW αναπαραστάσεις. Έχουμε, λοιπόν, τον εξής κώδικα:

# In[10]:


from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(analyzer = preproc_tok)
vectorizer.fit(imdbCorpus+testCorpus)

X_train = vectorizer.transform(imdbCorpus)
X_test = vectorizer.transform(testCorpus)


# γ) Εδώ εκπαιδεύουμε τον ταξινομητή LogisticRegression του sklearn για να ταξινομήσουμε τα σχόλια σε
# θετικά και αρνητικά:

# In[11]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import zero_one_loss

clf = LogisticRegression()
clf.fit(X_train, Y_train)


# δ) Επαναλαμβάνουμε την διαδικασία των βημάτων (β) και (γ) χρησιμοποιώντας τον TfidfVectorizer για την εξαγωγή TF-IDF αναπαραστάσεων. Έτσι έχουμε:

# In[47]:


from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(analyzer = preproc_tok)

X2_train = tfidf.fit_transform(imdbCorpus)
X2_test = tfidf.transform(testCorpus)

clf2 = LogisticRegression()
clf2.fit(X2_train, Y_train)


# __*Σύγκριση Αποτελεσμάτων*__
#
# Για να συγκρίνουμε τα αποτελέσματα των δύο παραπάνω ταξινομητών θα χρησιμοποιήσουμε το __confusion_matrix__ της sklearn.

# In[51]:


from sklearn.metrics import confusion_matrix


# - Για τον CountVectorizer προκύπτουν τα εξής αποτελέσματα:

# In[13]:


predictions = clf.predict(X_test)
print(confusion_matrix(Y_test, predictions))


# Δηλαδή το ποσοστό λάθους είναι __14,45%__.
# - Αντίστοιχα για τον TfidfVectorizer έχουμε:

# In[14]:


prediction2 = clf.predict(X2_test)
print(confusion_matrix(Y_test, prediction2))


# Δηλαδή το ποσοστό λάθους είναι __15,56%__
#
# Παρατηρούμε πως στην περίπτωση του TfidfVectorizer έχουμε μεγαλύτερο ποσοστό λάθος παρόλο που περιμέναμε βελτίωση όπως αναφέραμε θεωρητικά στο 17α. Καθώς, όμως η διαφορά δεν είναι σημαντικού μεγέθους, το ότι έχουμε αύξηση στο ποσοστό λάθους ωφείλεται στην τυχαιότητα του fit.
#
# ## ΒΗΜΑ 18: Χρήση Word2Vec αναπαραστάσεων για ταξινόμηση
#
# α) Στο σημείο αυτό θα χρησιμοποιήσουμε τις αναπαραστάσεις που υπολογίσαμε στο βήμα 9 προκειμένου να υπολογίσουμε το ποσοστό __Out Of Vocabulary (OOV) words__ και για τον λόγο αυτό επαναλαμβάνουμε αρχικά τον κώδικα που χρησιμοποιήσαμε στο βήμα 9.

# In[15]:


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


# In[16]:


corpus = file_preprocess("./corpus.txt", tokenize)


# In[17]:


from gensim.models import Word2Vec

model = Word2Vec(corpus, size=100, window=5, min_count=1, workers=5)

model.train(corpus, total_examples=len(corpus), epochs=50)

model.save("word2vec.model")


# Αποθηκεύουμε το λεξιλόγιο του embedding που χρησιμοποιήσαμε στην μεταβλητή dictionary:

# In[18]:


dictionary = model.wv.vocab


# Για να υπολογίσουμε, λοιπόν, τις OOV words θα χρησιμοποιήσουμε αρχικά την συνάρτηση __get_vocab__. Η get_vocab δέχεται σαν όρισμα ένα corpus και επιστρέφει μία λίστα από τις λέξεις πoυ υπάρχουν στο corpus. To αποτέλεσμα της συνάρτησης αυτής αποθηκεύουμε στην λίαστα vocabulary γιατί αυτό θα είναι το λεξιλόγιο του corpus μας.

# In[19]:


def get_vocab(corpus):
    tokens = set()
    for i in corpus:
        for j in i:
            tokens.add(j)
    return list(tokens)

vocabulary = get_vocab(imdbCorpus+testCorpus)


# Η συνάρτηση __count_oov__ δέχεται σαν ορίσματα τα tokens τα οποία είναι το λεξιλόγιο του corpus και το vocab που είναι ένα dictionary που περιέχει το vocabulary του embedding. H count_oov υπολογίζει το ποσοστό των tokens που δεν υπάρχουν στο vocab.

# In[20]:


def count_oov(tokens, vocab):
    count = 0
    for i in tokens:
        if i not in vocab:
            count+=1
    return count/len(tokens)

count_oov(vocabulary, dictionary)


# Υπολογίζουμε, λοιπόν, ότι το ποσοστό oov words είναι __87.36%__.
#
# β) Τώρα χρησιμποιώντας αυτές τις αναπαραστάσεις θα κατασκευάσουμε ένα __Neural Bag of Words__ αναπαραστάσεων για κάθε σχόλιο στο corpus και θα εκπαιδεύσουμε ένα Logistic Regression μοντέλο για ταξινόμηση.
# Θα χρησιμοποιήσουμε την συνάρτηση __create_cbow__. Η create_cbow δέχεται σαν είσοδο ένα corpus, ένα model όπως αυτό που εκπαιδεύσαμε στο βήμα 9 και μία συνατηση που έχει σαν default την preproc_tok. Η create_cbow επιστρέφει μία λίστα με το μέσο όρο των w2v διανυσμάτων κάθε λέξης που περιέχεται στο corpus.

# In[21]:


def create_cbow(corpus, model, size, proc=preproc_tok):
    dictionary = model.wv.vocab
    out = np.zeros((len(corpus), size))
    for row, sample in enumerate(corpus):
        sample_toks = proc(sample)
        for tok in sample_toks:
            if tok in dictionary:
                out[row] += model.wv[tok]
        out[row] = out[row]/len(sample_toks)
    return out

Χρησιμοποιώντας τώρα την προηγούμενη συνάρτηση διαβάζουμε τα train και test δεδομένα μας προκειμένου να δούμε τα αποτελέσματα αυτού του ταξινομητή. Έτσι έχουμε:
# In[22]:


X3_train = create_cbow(imdbCorpus, model, 100)


# In[23]:


X3_test = create_cbow(testCorpus, model, 100)


# In[24]:


clf3 = LogisticRegression()
clf3.fit(X3_train, Y_train)

predictions = clf3.predict(X3_test)
print(confusion_matrix(Y_test, predictions))


# __*Σχολιασμός αποτελεσμάτων*__
#
# Παρατηρούμε πως το ποσοστό λάθους (__= 29,43%__) είναι πολύ υψηλό. Η εξήγηση σε αυτό είναι ότι έχουμε χρησιμοποιήσει ένα πολύ μικρό corpus, με πολύ περιορισμένο λεξιλόγιο και άρα είναι δύσκολο να δημιουργηθούν παρόμοιες αναπαραστάσεις για κοντινές σημασιολογικά λέξεις
#
# γ) Κατεβάζουμε τα προεκπαδευμένα α GoogleNews vectors: https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit.
#
# δ) Φορτώνουμε τα δεδομένα που κατεβάσαμε με gensim:

# In[13]:


from gensim.models import KeyedVectors

model2 = KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin',
binary=True, limit=NUM_W2V_TO_LOAD)


# Και επαναλαμβάνουμε το ερώτημα 9γ παίρνωντας τα εξής αποτελέσματα για το googleModel:

# In[75]:


model2.wv.most_similar('nightmare')


# In[76]:


model2.wv.most_similar('craft')


# In[78]:


model2.wv.most_similar('originally')


# In[79]:


model2.wv.most_similar('savage')


# In[80]:


model.wv.most_similar('wonder')


# In[81]:


model.wv.most_similar('impregnable')


# In[82]:


model.wv.most_similar('conducted')


# In[83]:


model.wv.most_similar('indemnify')


# In[84]:


model.wv.most_similar('chagrined')


# In[85]:


model.wv.most_similar('impression')


# __*Σχολιασμός αποτελεσμάτων*__
#
# Ανατρέχοντας στα αποτελέσματα του βήματος 9γ παρατηρούμε πως έχουμε εντυπωσιακή βελτίωση αφού στην περίπτωση του googleModel όλες οι λέξεις είναι όντως κοντινές σημασιολογικά.

# ε) Εκπαιδεύουμε τώρα ένα LogisticRegression ταξινομητή προκειμένου να συγκρίνουμε τα αποτελέσματα:

# In[26]:


X4_train = create_cbow(imdbCorpus, model2, 300)


# In[27]:


X4_test = create_cbow(testCorpus, model2, 300)


# In[28]:


clf4 = LogisticRegression()
clf4.fit(X4_train, Y_train)

predictions = clf4.predict(X4_test)
print(confusion_matrix(Y_test, predictions))


# __*Σύγκριση αποτελεσμάτων*__
#
# Παρατηρούμε πως στην περίπτωση χρήσης του googleModel έχουμε μείωση του ποσοστού λάθους στο __17,26%__, όπως ήταν αναμενόμενο.
#
# στ) Στο σημείο αυτό θα χρησιμοποιήσουμε τα TF-IDF βάρη των λέξεων για να δημιουργήσουμε αναπαραστάσεις κριτικών με βάση τις w2v αναπαραστάσεις των λέξεων που περιέχουν.
#
# Γράφουμε την συνάρτηση __create_cbow_weighted__ για να υπολογίσουμε το neural bow με βάρη του tfidf vectorizer.

# In[43]:


def create_cbow_weighted(corpus, model, tfidf, trans, size, proc=preproc_tok):
    weights = trans.toarray()
    out = np.zeros((len(corpus), size))
    for row,sample in enumerate(corpus):
        sample_toks = proc(sample)
        total_weights=0
        for j in sample_toks:
            if j in model.wv.vocab and j in tfidf.vocabulary_:
                out[row]+=model.wv[j]*weights[row][tfidf.vocabulary_[j]]
                total_weights+=weights[row][tfidf.vocabulary_[j]]
        out[row]/=total_weights
    return out


# In[48]:


X5_train = create_cbow_weighted(imdbCorpus, model2, tfidf, X2_train, 300)


# In[49]:


X5_test = create_cbow_weighted(testCorpus, model2, tfidf, X2_test, 300)


# In[52]:


clf5 = LogisticRegression()
clf5.fit(X5_train, Y_train)

predictions = clf5.predict(X5_test)
print(confusion_matrix(Y_test, predictions))


# Εδώ βρίσκουμε ποσοστό λάθους __19,17%__.

# ## ΒΗΜΑ 19: Extra credit
#
# α) Θα χρησιμοποιήσουμε διαφορετικούς ταξινομητές προκειμένου να συγκρίνουμε την επίδοσή τους.
#
# - Ταξινομητής __LinearSVC__:

# In[31]:


from sklearn.svm import LinearSVC

clf6 = LinearSVC()
clf6.fit(X4_train, Y_train)

predictions = clf6.predict(X4_test)
print(confusion_matrix(Y_test, predictions))


# ποσοστό λάθους = __14,95%__.
#
# - Ταξινομητής __KNeighborsClassifier__:

# In[196]:


from sklearn.neighbors import KNeighborsClassifier

clf7 = KNeighborsClassifier(n_neighbors=3)
clf7.fit(X4_train, Y_train)

predictions = clf7.predict(X4_test)
print(confusion_matrix(Y_test, predictions))


# ποσοστό λάθους = __26,81%__.
#
# - Ταξινομητής __MLPClassifier__:

# In[187]:


from sklearn.neural_network import MLPClassifier

clf8 = MLPClassifier(alpha=1e-5,hidden_layer_sizes=(5, 3))
clf8.fit(X4_train, Y_train)

predictions = clf8.predict(X4_test)
print(confusion_matrix(Y_test, predictions))


# ποσοστό λάθους = __14,55%__.
#
# - Ταξινομητής __RandomForestClassifier__:

# In[32]:


from sklearn.ensemble import RandomForestClassifier

clf9 = RandomForestClassifier(max_depth=10)
clf9.fit(X4_train, Y_train)

predictions = clf9.predict(X4_test)
print(confusion_matrix(Y_test, predictions))


# ποσοστό λάθους = __24,64%__.
#
# - Ταξινομητής __GaussianNB__:

# In[195]:


from sklearn.naive_bayes import GaussianNB

clf10 = GaussianNB()
clf10.fit(X4_train, Y_train)

predictions = clf10.predict(X4_test)
print(confusion_matrix(Y_test, predictions))


# ποσοστό λάθους = __27,07%__.
#
# β) Μπορούμενα επίσης να χρησιμοποιήσουμε και διαφορετικά μοντέλα προκειμένου να εξετάσουμε την επίδοση των ταξινομητών:
#
# - Μοντέλο wiki-news:

# In[33]:


model3 = KeyedVectors.load_word2vec_format('./wiki-news-300d-1M.bin',
binary=True,encoding='unicode_escape', limit=NUM_W2V_TO_LOAD)


# In[214]:


X6_train = create_cbow(imdbCorpus, model3, 300)


# In[215]:


X6_test = create_cbow(testCorpus, model3, 300)


# In[216]:


clf11 = LogisticRegression()
clf11.fit(X6_train, Y_train)

predictions = clf11.predict(X6_test)
print(confusion_matrix(Y_test, predictions))


# ποσοστό λάθους = __41,82%__.
#
# Το παραπάνω ποσοστό φαίνεται να είναι τυχαίο. Αν εξετάσουμε τα embedings τα οποία πήραμε θα παρατηρήσουμε πως είναι σε κωδικοποιήση που δεν μπορεί να αναγνωρίσει ο υπολογιστής μας και άρα καμία από τις λέξεις δεν βρίσκεται στο vocabulary του embeding και άρα αυτό εξηγεί το ότι ο ταξινομητής κάνει ταξινόμηση με τυχαίο τρόπο.

# In[ ]:
