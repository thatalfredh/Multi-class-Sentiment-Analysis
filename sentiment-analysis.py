""" Sentiment Analysis (multi-class rating) on E-commerce product reviews """

import pandas as pd
import numpy as np 
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
from googletrans import Translator
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout, SpatialDropout1D
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

raw_data = pd.read_csv('train.csv', encoding='latin-1')

""" ------------------------- Text Preprocessing ------------------------- """
# bad reviews are statistically longer - more info to preprocess
samples = raw_data.head(100)
# reviews seem to be translated inaccurately from malaysian melayu to english
# (did random selection of translation)

"""
<<< Pre-processing pipeline >>> :
(1) To lowercase (DONE)

(2) Remove: (DONE)
        (2.1) - non ASCII printable characters
              - emojis
        (2.2) - punctuations
        
(3) Make single space (DONE)

(4) Truncate words that contains: (DONE)
    (4.1) > 1 consecutive character at start
    (4.2) > 2 consecutive chars at end

(5) Pass Google Translate API (READ OPS TIMEOUT ERROR)

(6) Remove: (DONE)
        - stopwords
"""
processed_data = raw_data.copy()

#processed_train = samples.copy()

# (1) To lowercase
processed_data['review'] = processed_data['review'].apply(lambda x: x.lower())

# (2.1) Remove non-ASCII printable chars (Emojis inclusive)
def remove_non_printable(text):
     clean_txt = ''.join(c for c in text if c in string.printable)
     return clean_txt
     
processed_data['review'] = processed_data['review'].apply(lambda x: remove_non_printable(x))

# (2.2) Remove punctuations and numerics
def remove_punc_and_digit(text):
    removable = [p for p in string.printable[63:]]
    del removable[31]
    removable = removable + [c for c in string.printable[0:10]]
    clean_txt = ''.join(c for c in text if c not in removable)
    return clean_txt

processed_data['review'] = processed_data['review'].apply(lambda x: remove_punc_and_digit(x))

# (3) Make single spacing
def single_spacing(text):
    clean_txt = re.sub(r'\s{2,}', ' ', text)
    return clean_txt

processed_data['review'] = processed_data['review'].apply(lambda x: single_spacing(x))

# (4) Truncation
def truncate(text):
    re_pattern_1 = re.compile(r'^(\w)\1*')
    re_pattern_2 = re.compile(r'(\w)\1*$')
    match_sub_1 = r'\1'
    match_sub_2 = r'\1\1'
    t = re_pattern_1.sub(match_sub_1,text)
    clean_txt = re_pattern_2.sub(match_sub_2,t)
    return clean_txt

processed_data['review'] = processed_data['review'].apply(lambda x: truncate(x))

STOPWORDS = set(stopwords.words('english'))
def remove_stopwords(text, STOPWORDS):
    clean_txt = ' '.join(word for word in text.split() if word not in STOPWORDS)
    return clean_txt

processed_data['review'] = processed_data['review'].apply(lambda x: remove_stopwords(x,STOPWORDS))

# check for current longest string
processed_data['review'].apply(lambda x: len(x)).argmax()

# Special cases
processed_data['review'][54149] = processed_data['review'][54149][:97]
processed_data['review'][78285] = processed_data['review'][78285][:182]
processed_data['review'][78627] = processed_data['review'][78627][:182]
processed_data['review'][79780] = processed_data['review'][79780][:182]
processed_data['review'][1613] = processed_data['review'][1613][204:]
processed_data['review'][5647] = processed_data['review'][5647][18:674]
processed_data['review'][2506] = processed_data['review'][2506][:688]

processed_data['review'].apply(lambda x: len(x)).argmax()
len(processed_data['review'][2506])

"""
# =========== googletrans is useful if reviews are generally short =========== #

# (5) Translate each malay ('ms') word to english ('en')
# experiment with 'ms', 'id', 'translator.detect(text).lang'
def translated(text):
    translator = Translator() 
    translation = translator.translate(text,src='id',dest='en')
    return translation.text

processed_train['review'] = processed_train['review'].apply(lambda x: translated(x))
"""
# The maximum number of words to be used. (most frequent)
MAX_NB_WORDS = 60000
# Max number of words in each review
MAX_SEQUENCE_LENGTH = 688
# This is fixed
EMBEDDING_DIM = 200

tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(processed_data['review'].values)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

X = tokenizer.texts_to_sequences(processed_data['review'].values)
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', X.shape)

y = pd.get_dummies(processed_data['rating'].values)
print('Shape of label tensor:', y.shape)

""" ----------------- Model Building - Bidirectional LSTM ----------------- """
# Hyperparameters
epochs = 30 #10
learning_rate = 0.0005 #0.001
batch_size = 128 #64
dropout_prob = 0.2
train_size = 0.8

# train - val split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = (1-train_size), random_state = 42)

model = Sequential()
model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
model.add(SpatialDropout1D(0.1))
model.add(Bidirectional(LSTM(EMBEDDING_DIM, dropout=dropout_prob, recurrent_dropout=0))) # 100 0.2 0.2 
model.add(Dense(5, activation='softmax'))
model.compile(loss='categorical_crossentropy', 
              optimizer=Adam(learning_rate=learning_rate), 
              metrics=['accuracy'])
print(model.summary())

""" ------------------------------ Training ------------------------------"""
model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,validation_split=0.20)

accr = model.evaluate(X_test,y_test)
print("Test Loss = ", accr[1])

""" ------------------------------ Testing ------------------------------"""
test_data = pd.read_csv('test.csv', encoding='latin-1')
test_data['review'] = test_data['review'].apply(lambda x: x.lower())
test_data['review'] = test_data['review'].apply(lambda x: remove_non_printable(x))
test_data['review'] = test_data['review'].apply(lambda x: remove_punc_and_digit(x))
test_data['review'] = test_data['review'].apply(lambda x: single_spacing(x))
test_data['review'] = test_data['review'].apply(lambda x: truncate(x))

seq = tokenizer.texts_to_sequences(test_data['review'].values)
padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
pred = model.predict(padded)

# 5 classes
labels = [1,2,3,4,5]

    
rating = [labels[np.argmax(x)] for x in pred]

results = pd.DataFrame({"review_id":test_data['review_id'],
                      "rating":rating})
results.to_csv("prelim_result_2.csv",index=False)



