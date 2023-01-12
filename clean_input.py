#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 10:58:51 2022

@author: Lucie
"""
import pip
pip.main(["install","nltk"])
pip.main(["install","sklearn"])
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import os 
import pickle
import re 
import numpy as np
import pandas as pd

file = os.path.dirname(__file__)
path_tfidf = os.path.join(file, "tfidf.pickle")


# def clean(text):
#     punctuation_signs = list("?:!.,;")
#     stop_words = list(stopwords.words('english'))
    
#     text = str(text)
#     text= text.replace("\r", " ")
#     text= text.replace("\n", " ")
#     text= text.replace("    ", " ")
#     text= text.replace('"', '')
#     text= text.lower()
#     text= text.replace("", '')
#     text = re.sub(r'[^\w\s]', '', text)
    
#     text = [text]
#     vectorizer = TfidfVectorizer()
#     vectors = vectorizer.fit_transform(text)

#     return vectors

def processor(text):
    stop_words = set(stopwords.words("english"))
    text = ' '.join([word for word in text.split() if word not in (stop_words)])
    text = re.sub('<[^>]*>', '', text) 
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
    return text

def table_tweet(tweet) : 
    clean_tweet = processor(tweet)
    table = pd.DataFrame({"ID": [""], "Text": [tweet], "Label": [""], "clean_tweets": [clean_tweet], "Hashtags":["Nan"], "tokens":["Nan"]}, index=None)
    table_for_concat = pd.DataFrame({"clean_tweets": [clean_tweet]})
    return table_for_concat
