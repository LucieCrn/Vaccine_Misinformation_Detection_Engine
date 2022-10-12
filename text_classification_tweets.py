# -*- coding: utf-8 -*-
"""Text-classification-tweets.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1STB2RbM9xrzIx2vJabnKl1uSDp_L_8-k
"""

# 1st step: Merging the datasets

import pandas as pd
import os
import re
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn import metrics
import pickle

##### PATHS #####
file = os.path.dirname(__file__)
data02_path = os.path.join(file, "dataset_lables.xlsx")
data_other_path = os.path.join(file, "dataset_group2.xlsx")


data02 = pd.read_excel(data02_path)
data02.head(200)

data02.dtypes

data02.dropna()

data02["Label"]=data02["Label"].astype(int)

data_other = pd.read_excel(data_other_path)
data_other.head()

data_other.dtypes

data_other= pd.DataFrame(data_other)

data_other["ID"]=data_other["ID"].astype(int)

data_other.dropna()

## Concat the 2 drones

df_tweets = pd.concat([data02, data_other])

df_tweets


## DATA PREPROCESSING

#Remove punctuation

#from nltk.tokenize import RegexpTokenizer

import re
def  clean_text(df_tweets, text_field, new_text_field_name):
    df_tweets[new_text_field_name] = df_tweets[text_field].str.lower()
    df_tweets[new_text_field_name] = df_tweets[new_text_field_name].apply(lambda elem: re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", " ", elem))  
    # remove numbers
    df_tweets[new_text_field_name] = df_tweets[new_text_field_name].apply(lambda elem: re.sub(r"\d+", " ", elem))
    
    return df_tweets

df_tweets = clean_text(df_tweets, 'Text', 'clean_tweets')

df_tweets

# Extract Hashtags 

df_tweets['Hashtags'] = df_tweets['Text'].str.findall(r'#.*?(?=\s|$)')

df_tweets

# STOPWORDS
import nltk 
nltk.download('stopwords')
from nltk.corpus import stopwords

stop_words = set(stopwords.words("english"))

df_tweets['clean_tweets'] = df_tweets['clean_tweets'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))

# TOKENIZATION
#Making tokens

import nltk 
nltk.download('punkt')
from nltk.tokenize import sent_tokenize, word_tokenize
df_tweets['tokens'] = df_tweets['clean_tweets'].apply(lambda x: word_tokenize(x))

df_tweets.head(100)

df_tweets.to_csv("df_tweets.csv", sep=";")



def model_predict(my_input, df_tweets) : 
    ## VECTORIZATION TF-IDF
    
    my_input = my_input.to_numpy()
    my_input = my_input.flatten()
    
    X_train, X_test , y_train, y_test = train_test_split(df_tweets['clean_tweets'].values,df_tweets['Label'].values,test_size=0.2,random_state=123,stratify=df_tweets['Label'].values)
    
    X_test = np.concatenate((X_test,my_input))
    y_test = np.concatenate((y_test,np.array([])))
    
    tfidf_vectorizer = TfidfVectorizer() 
    
    tfidf_train_vectors = tfidf_vectorizer.fit_transform(X_train)
    
    tfidf_test_vectors = tfidf_vectorizer.transform(X_test)
    
    
    ## SVM
    
    from sklearn import svm
    
    classifier_svm = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
    classifier_svm.fit(tfidf_train_vectors, y_train)
    model_svm = classifier_svm.fit(tfidf_train_vectors, y_train)
    
    y_pred_SVM = model_svm.predict(tfidf_test_vectors)
    result = y_pred_SVM[-1]
    return result

# Use accuracy_score function to get the accuracy
# print("SVM Accuracy Score -> ",accuracy_score(y_pred_SVM, y_test))

# print(classification_report(y_test,y_pred_SVM))

# d_SVM = {
#      'Model': 'SVM',
#      'Test Set Accuracy': metrics.accuracy_score(y_pred_SVM, y_test)
# }

# df_models_svm = pd.DataFrame(d_SVM, index=[0])


# Save the model and the dataset 

# with open(os.path.join(file, "model_svm.pickle"), 'wb') as output:
#     pickle.dump(model_svm, output)
    
# with open(os.path.join(file, "df_models_svm.pickle.pickle"), 'wb') as output:
#     pickle.dump(df_models_svm, output)
