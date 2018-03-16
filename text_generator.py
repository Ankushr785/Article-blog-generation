
# coding: utf-8

# In[57]:


import gensim
import fasttext as ft
import re
import itertools
from keras.layers.core import Activation, Dense, Dropout, SpatialDropout1D
from keras.layers.embeddings import Embedding
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
import collections 
import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import string


# In[58]:


articles1 = pd.read_csv('articles1.csv')
articles = articles1.iloc[:100,:]


# In[59]:


#topic generator
def topic_generation(x):
    doc_complete = nltk.sent_tokenize(x)
    stop = set(nltk.corpus.stopwords.words('english'))
    exclude = set(string.punctuation) 
    lemma = nltk.stem.wordnet.WordNetLemmatizer()
    def clean(doc):
        stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
        punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
        normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
        return normalized

    doc_clean = [clean(doc).split() for doc in doc_complete]
    dictionary = gensim.corpora.Dictionary(doc_clean)

    # Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]
    Lda = gensim.models.ldamodel.LdaModel

    # Running and Trainign LDA model on the document term matrix.
    ldamodel = Lda(doc_term_matrix, num_topics=len(doc_complete), id2word = dictionary, passes=20)
    topic_list = []

    topics = ldamodel.show_topics(num_topics = len(doc_complete), num_words = 1)
    for i in range(len(topics)):
        if topics[i][1][7:-1] not in topic_list:
            topic_list.append(topics[i][1][7:-1])
    return topic_list

articles['topics'] = articles.content.map(lambda x:topic_generation(x))


# In[60]:


stopset = set(nltk.corpus.stopwords.words('english'))
def cleaner(x):
    x = nltk.word_tokenize(str(x))
    x = [w for w in x if not w in stopset]
    x = [z.lower() for z in x]
    return x

articles.content = articles.content.map(lambda x:cleaner(x))
articles.title = articles.title.map(lambda x:cleaner(x))


# In[61]:


articles.head()


# In[62]:


vocab = []
word = []
for i in range(len(articles.content)):
    for j in range(len(articles.content[i])):
        word.append(articles.content[i][j])
        if articles.content[i][j] not in vocab:
            vocab.append(articles.content[i][j])
#chars = list(set(data))
VOCAB_SIZE = len(vocab)

seq_length = 1
print('Data length: {} words'.format(len(word)))
print('Vocabulary size: {} words'.format(VOCAB_SIZE))

ix_to_word = {ix:word for ix, word in enumerate(vocab)}
word_to_ix = {word:ix for ix, word in enumerate(vocab)}
    
X = np.zeros((int(len(word)/seq_length), seq_length, VOCAB_SIZE))
y = np.zeros((int(len(word)/seq_length), seq_length, VOCAB_SIZE))
y_bar = np.zeros((int(len(word)/seq_length), seq_length, VOCAB_SIZE))
for i in range(0, int(len(articles.content)/seq_length)):
    X_sequence = word[i*seq_length:(i+1)*seq_length]
    X_sequence_ix = [word_to_ix[value] for value in X_sequence]
    input_sequence = np.zeros((seq_length, VOCAB_SIZE))
    for j in range(seq_length):
        input_sequence[j][X_sequence_ix[j]] = 1.
        X[i] = input_sequence

    y_sequence = word[i*seq_length+1:(i+1)*seq_length+1]
    y_sequence_ix = [word_to_ix[value] for value in y_sequence]
    target_sequence = np.zeros((seq_length, VOCAB_SIZE))
    for j in range(seq_length):
        target_sequence[j][y_sequence_ix[j]] = 1
        y[i] = target_sequence
        
    y_bar_sequence = word[i*seq_length-1:(i+1)*seq_length-1]
    y_bar_sequence_ix = [word_to_ix[value] for value in y_sequence]
    target_sequence_bar = np.zeros((seq_length, VOCAB_SIZE))
    for j in range(seq_length):
        target_sequence_bar[j][y_sequence_ix[j]] = 1
        y_bar[i] = target_sequence_bar


# In[63]:


xtrain = X[:int(.75*len(X)), :, :]
ytrain = y[:int(.75*len(X)), :, :]
xtest = X[int(.75*len(X)):, :, :]
ytest = y[int(.75*len(X)):, :, :]


# In[64]:


x_train = X[:int(.75*len(X)), :, :]
y_train = y_bar[:int(.75*len(X)), :, :]
x_test = X[int(.75*len(X)):, :, :]
y_test = y_bar[int(.75*len(X)):, :, :]


# In[ ]:


model = Sequential()
model.add(LSTM(5000, input_shape=(None, VOCAB_SIZE), return_sequences=True))
for i in range(6):
  model.add(LSTM(int(4500/(i+1)), return_sequences=True))
model.add(TimeDistributed(Dense(VOCAB_SIZE)))
model.add(Activation('softmax'))
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics = ["accuracy"])


# In[ ]:


BATCH_SIZE = 1000
history = model.fit(xtrain, ytrain, batch_size=BATCH_SIZE, verbose=1, epochs=10, validation_data=(xtest, ytest))


# In[ ]:


model.save_weights('model.hdf5', overwrite = True)


# In[ ]:


#word_sequence = pd.Series(words)
length = 5
ix = [np.random.randint(VOCAB_SIZE)]
y_word = [ix_to_word[ix[-1]]]
x = np.zeros((1, length, VOCAB_SIZE))
for i in range(length):
    # appending the last predicted character to sequence
    x[0, i, :][ix[-1]] = 1
    #print(ix_to_word[ix[-1]], end="")
    ix = np.argmax(model.predict(x[:, :i+1, :])[0], 1)
    y_word.append(ix_to_word[ix[-1]])
    if y_word[-1] in ['.', '?', '!']:
        break
print((' ').join(y_word))


# In[ ]:


model2 = Sequential()
model2.add(LSTM(5000, input_shape=(None, VOCAB_SIZE), return_sequences=True))
for i in range(6):
  model2.add(LSTM(int(4500/(i+1)), return_sequences=True))
model2.add(TimeDistributed(Dense(VOCAB_SIZE)))
model2.add(Activation('softmax'))
model2.compile(loss="categorical_crossentropy", optimizer="adam", metrics = ["accuracy"])


# In[ ]:


BATCH_SIZE_BAR = 1000
history_bar = model2.fit(x_train, y_train, batch_size=BATCH_SIZE_BAR, verbose=1, epochs=10, validation_data=(x_test, y_test))


# In[ ]:


model2.save_weights('model2.hdf5', overwrite = True)


# In[ ]:


#word_sequence = pd.Series(words)
length_bar = 5
ix_bar = [np.random.randint(VOCAB_SIZE)]
y_word_bar = [ix_to_word[ix_bar[-1]]]
x_bar = np.zeros((1, length_bar, VOCAB_SIZE))
for i in range(length_bar):
    # appending the last predicted character to sequence
    x_bar[0, i, :][ix_bar[-1]] = 1
    #print(ix_to_word[ix[-1]], end="")
    ix_bar = np.argmax(model2.predict(x_bar[:, :i+1, :])[0], 1)
    y_word_bar.append(ix_to_word[ix_bar[-1]])
    if y_word_bar[-1] in ['.', '?', '!']:
        del y_word_bar[-1]
        break
    
y_word_bar = reversed(y_word_bar)
print((' ').join(y_word_bar))


# In[ ]:


import matplotlib.pyplot as plt
plt.subplot(211)
plt.title("Accuracy")
plt.plot(history.history["acc"], color="g", label="Train")
plt.plot(history.history["val_acc"], color="b", label="Validation")
plt.legend(loc="best")

plt.subplot(212)
plt.title("Loss")
plt.plot(history.history["loss"], color="g", label="Train")
plt.plot(history.history["val_loss"], color="b", label="Validation")
plt.legend(loc="best")
plt.tight_layout()

plt.show()


# In[ ]:


plt.subplot(211)
plt.title("Accuracy")
plt.plot(history_bar.history["acc"], color="g", label="Train")
plt.plot(history_bar.history["val_acc"], color="b", label="Validation")
plt.legend(loc="best")

plt.subplot(212)
plt.title("Loss")
plt.plot(history_bar.history["loss"], color="g", label="Train")
plt.plot(history_bar.history["val_loss"], color="b", label="Validation")
plt.legend(loc="best")
plt.tight_layout()

plt.show()


# In[ ]:


#Reload model
model.load_weights('model.hdf5')


# In[ ]:


#BATCH_SIZE = 1000
#history = model.fit(xtrain, ytrain, batch_size=BATCH_SIZE, verbose=1, epochs=10, validation_data=(xtest, ytest))


# In[ ]:


#word_sequence = pd.Series(words)
length = 5
ix = [np.random.randint(VOCAB_SIZE)]
y_word = [ix_to_word[ix[-1]]]
x = np.zeros((1, length, VOCAB_SIZE))
for i in range(length):
    # appending the last predicted character to sequence
    x[0, i, :][ix[-1]] = 1
    #print(ix_to_word[ix[-1]], end="")
    ix = np.argmax(model.predict(x[:, :i+1, :])[0], 1)
    y_word.append(ix_to_word[ix[-1]])
    if y_word[-1] in ['.', '?', '!']:
        break
print((' ').join(y_word))


# In[ ]:


#Reload model
model2.load_weights('model2.hdf5')


# In[ ]:


#BATCH_SIZE_BAR = 1000
#history = model2.fit(x_train, y_train, batch_size=BATCH_SIZE_BAR, verbose=1, epochs=10, validation_data=(x_test, y_test))


# In[ ]:


#word_sequence = pd.Series(words)
length_bar = 5
ix_bar = [np.random.randint(VOCAB_SIZE)]
y_word_bar = [ix_to_word[ix_bar[-1]]]
x_bar = np.zeros((1, length_bar, VOCAB_SIZE))
for i in range(length_bar):
    # appending the last predicted character to sequence
    x_bar[0, i, :][ix_bar[-1]] = 1
    #print(ix_to_word[ix[-1]], end="")
    ix_bar = np.argmax(model2.predict(x_bar[:, :i+1, :])[0], 1)
    y_word_bar.append(ix_to_word[ix_bar[-1]])
    if y_word_bar[-1] in ['.', '?', '!']:
        del y_word_bar[-1]
        break
    
y_word_bar = reversed(y_word_bar)
print((' ').join(y_word_bar))


# In[ ]:


keywords = input(str("Enter some keywords or sentences. In case you're entering keywords, don't use comma separation."))


# In[ ]:


keys = keywords.split()
for i in keys:
    if i not in word_to_ix:
        keys.remove(i)


# In[ ]:


if keys is None:
    print('Error! No keyword recognized!')
else:
    forward_sentences = []
    backward_sentences = []
    for m in keys:
        length = 5
        ix = [word_to_ix[m]]
        y_word = [ix_to_word[ix[-1]]]
        relevant_words = []
        for j in range(len(articles)):
            if y_word in articles.topics[j]:
                for k in range(len(articles.content[j])):
                    relevant_words.append(articles.content[j][k])
        x = np.zeros((1, length, VOCAB_SIZE))
        for i in range(length):
            # appending the last predicted character to sequence
            x[0, i, :][ix[-1]] = 1
            #print(ix_to_word[ix[-1]], end="")
            ix = np.argmax(model.predict(x[:, :i+1, :])[0], 1)
            for l in range(len(ix)):
                if ix_to_word[ix[-1-l]] in relevant_words:
                    y_word.append(ix_to_word[ix[-1-l]])
                    break
            if y_word[-1] in ['.', '?', '!']:
                break
        forward_sentences.append((' ').join(y_word))
    
    for m in keys:
        length_bar = 5
        ix_bar = [word_to_ix[m]]
        y_word_bar = [ix_to_word[ix_bar[-1]]]
        relevant_words_bar = []
        for j in range(len(articles)):
            if y_word_bar in articles.topics[j]:
                for k in range(len(articles.content[j])):
                    relevant_words_bar.append(articles.content[j][k])
        x_bar = np.zeros((1, length_bar, VOCAB_SIZE))
        for i in range(length_bar):
            # appending the last predicted character to sequence
            x_bar[0, i, :][ix_bar[-1]] = 1
            #print(ix_to_word[ix[-1]], end="")
            ix_bar = np.argmax(model2.predict(x_bar[:, :i+1, :])[0], 1)
            for l in range(len(ix_bar)):
                if ix_to_word[ix_bar[-1-l]] in relevant_words_bar:
                    y_word_bar.append(ix_to_word[ix_bar[-1-l]])
                    break
            if y_word_bar[-1] in ['.', '?', '!']:
                del y_word_bar[-1]
                break
        y_word_bar = reversed(y_word_bar)
        backward_sentences.append((' ').join(y_word_bar))


# In[ ]:


generated_paragraph = []
for i in range(len(forward_sentences)):
    generated_paragraph.append(backward_sentences[i] + ' '+ forward_sentences[i]) 


# In[ ]:


generated_paragraph

