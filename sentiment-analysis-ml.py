""" Sentiment Analysis (multi-class rating) on E-commerce product reviews """
import pandas as pd
import numpy as np 
import re
import string
import emoji
from nltk.corpus import stopwords
from googletrans import Translator

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score


raw_data = pd.read_csv('train.csv', encoding='utf-8')
extension = pd.read_csv('extension.csv', encoding='utf-8')

raw_data = pd.concat([raw_data, extension], axis=0)

clean = raw_data.copy()
clean.isnull().sum() # Check null 
""" ============================ Text Cleaning ============================"""
removables = [p for p in string.printable[63:]]
del removables[31]
removables = removables + [c for c in string.printable[0:10]]

STOPWORDS = set(stopwords.words('english'))

# (1) To lowercase
clean['review'] = clean['review'].apply(lambda x: x.lower())

def clean_text(text, removables, STOPWORDS):
    # (2) Remove non-ASCII printable chars (emojis inclusive)
    txt = ''.join(c for c in text if c in string.printable)
    
    # (3) Remove punctuations and numerics
    txt = ''.join(c for c in txt if c not in removables)
    
    # (4) Single spacing
    txt = re.sub(r'\s{2,}', ' ', txt)
    
    # (5) Truncation for misspelled words
    re_pattern_1 = re.compile(r'^(\w)\1*')
    re_pattern_2 = re.compile(r'(\w)\1*$')
    match_sub_1 = r'\1'
    match_sub_2 = r'\1\1'
    t = re_pattern_1.sub(match_sub_1,txt)
    txt = re_pattern_2.sub(match_sub_2,t)
    
    # (6) Remove Stopwords
    txt = ' '.join(word for word in txt.split() if word not in STOPWORDS)
    
    return txt

clean['review'] = clean['review'].apply(lambda x: clean_text(x, removables, STOPWORDS))

# check for current longest string
clean['review'].apply(lambda x: len(x)).argmax()

# Special cases
clean['review'][54149] = clean['review'][54149][:97]
clean['review'][78285] = clean['review'][78285][:182]
clean['review'][78627] = clean['review'][78627][:182]
clean['review'][79780] = clean['review'][79780][:182]
clean['review'][1613] = clean['review'][1613][204:]
clean['review'][5647] = clean['review'][5647][18:674]
clean['review'][2506] = clean['review'][2506][:688]

# Remove empty strings/nan due to emoji removal
empty = [idx for idx in clean[clean['review']==''].index]
clean.drop(clean.index[empty], inplace = True)
clean.index = range(len(clean))

clean.isnull().sum()
clean.dropna(inplace=True)

def translated(df):
    translator = Translator()
    for idx in range(len(df)):
        df['review'][idx] = (translator.translate(df['review'][idx],src='ms',dest='en')).text
        print("Translation complete for idx: ", idx)
    return df

translated_clean = translated(clean)

""" ============================ Model Building ============================ """
X_train, X_test, y_train, y_test = train_test_split(clean['review'],clean['rating'], test_size = 0.25, random_state = 42)

from sklearn import metrics

# Linear SVC:
text_clf_lsvc = Pipeline([('tfidf', TfidfVectorizer()),
                     ('clf', LinearSVC()),
])
text_clf_lsvc.fit(X_train, y_train)
y_pred = text_clf_lsvc.predict(X_test)
print(metrics.classification_report(y_test, y_pred))

# NB:
text_clf_nb = Pipeline([('tfidf', TfidfVectorizer()),
                     ('clf', MultinomialNB()),
])
text_clf_nb.fit(X_train, y_train)
y_pred = text_clf_nb.predict(X_test)
print(metrics.classification_report(y_test, y_pred))

# Logistic Regression:
text_clf_lr = Pipeline([('tfidf', TfidfVectorizer()),
                     ('clf', LogisticRegression()),
])
text_clf_lr.fit(X_train, y_train)
y_pred = text_clf_lr.predict(X_test)
print(metrics.classification_report(y_test, y_pred))

""" ============================ Submission ============================ """
test_data = pd.read_csv('test.csv',encoding='utf-8')
test_data['review'] = test_data['review'].apply(lambda x: x.lower())
test_data['review'] = test_data['review'].apply(lambda x: clean_text(x, removables, STOPWORDS))

ratings = text_clf_lr.predict(test_data['review'])

submission = pd.DataFrame({"review_id": test_data['review_id'],
                           "rating": ratings})

submission.to_csv("submission_5_lr.csv", index=False)
