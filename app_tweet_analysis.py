#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 13:23:21 2022

@author: Lucie
"""

import streamlit as st
import pandas as pd
import os
import re
import pickle
from clean_input import table_tweet
from text_classification_tweets import model_predict
import numpy as np


##### PATHS #####
file = os.path.dirname(__file__)
folder_images = os.path.join(file, "Médias")
dataset = os.path.join(file, "dataset_lables.xlsx")
model_svm = os.path.join(file, "model_svm.pickle")
df_tweets = os.path.join(file, "df_tweets.csv")


##### DATASET #####
dataset = pd.read_excel(dataset)
df_tweets = pd.read_csv(df_tweets, sep=";")


##### PROJECT DESCRIPTION PAGE #####

def page_description() : 
    st.write('<h1 style="text-align:center;color:#6A0888;font-weight:bolder;font-size:40px;">WELCOME TO OUR STREAMLIT APP 🤗</h1>',unsafe_allow_html=True)
    st.write('<h1 style="text-align:center;color:#6A0888;font-weight:bolder;font-size:20px;">PROJECT DESCRIPTION 💉</h1>',unsafe_allow_html=True) 
    st.text("")
    st.write("At Numerix, we are fake news hunters. Our speciality is to make truth great again ! 🦸")
    st.write("The aim of this Streamlit app is to analyze if a tweet about Covid-19 vaccine is considered misinformation or not, in order to detect misinformation messages about this topic on Twitter.")
    st.write("We have developped an algorithm using the [Support Vector Machines (SVM) algorithm](https://monkeylearn.com/blog/introduction-to-support-vector-machines-svm/). The algorithm has been trained with a dataset containing 6000 tweets extracted from Twitter about the topic “Covid-19 vaccine”.")
    st.write("On the “Tweet analysis” page of this app, you will be able to give a tweet and it will be classified into “Misinformation” and “No misinformation”.")




##### TWEET ANALYSIS PAGE #####

def page_analysis() :        
        
    st.write('<h1 style="text-align:center;color:#00BFFF;font-weight:bolder;font-size:40px;">TWEET ANALYSIS 🧐</h1>',unsafe_allow_html=True)
    st.write('<h1 style="text-align:left;color:#00BFFF;font-weight:bolder;font-size:20px;">Paste your tweet below 💬</h1>',unsafe_allow_html=True)
    with st.form("My Form") : 
        tweet = st.text_area("")
        nbr_words = len(re.findall(r"\w+", tweet))
        st.write(f"Your tweet contains {nbr_words} words.")
        submit_button = st.form_submit_button("Analyse my tweet 🔍")
    
    if submit_button : 
        with open(model_svm, 'rb') as data:
                model = pickle.load(data)
        
        my_input = table_tweet(tweet)
        label = model_predict(my_input, df_tweets)
        #st.write(label)

        if label == 0 : 
            st.success('No Misinformation ✅')
        elif label == 1 : 
            st.error('Misinformation ❌')
        else : 
            st.warning ("There is an error 🚨")
    
    

##### DATASET ANALYSIS PAGE #####

def page_dataset() : 
    st.write('<h1 style="text-align:center;color:#0B610B;font-weight:bolder;font-size:40px;">DATASET ANALYSIS 📊</h1>',unsafe_allow_html=True)
    st.markdown("---")
    
    # Show dataframe
    col1, col2 = st.columns(2)
    
    with col1: 
        st.dataframe(dataset)
    
    with col2: 
        st.write("This dataset contains 6000 tweets extracted from Twitter and talk about Covid-19 vaccine.")
        st.write('<h1 style="text-align:left;color:#BDBDBD;font-weight:normal;font-size:15px;">💡 Move the mouse over the column to display the text.</h1>',unsafe_allow_html=True)
                 
    
    # Add space btw DF and graphs
    st.text("")
    st.text("")
    st.text("")
    
    # Word clouds for each label
    col1, col2 = st.columns(2)
    
    with col1: 
        st.image(os.path.join(folder_images, "label0_cloud.png"), caption="No missinformed tweets word cloud")
        
    with col2:
        st.image(os.path.join(folder_images, "label1_cloud.png"), caption="Missinformed tweets word cloud")




#############
       
def main() : 
    st.set_page_config(layout="wide",
    page_title="Tweet Analysis",
    page_icon="🔍")
    
# COLONNE
    #st.sidebar.image(os.path.join(folder_images, "Numerix.png"))
    #st.sidebar.markdown("---")
    
    # Select a page 
    menu = st.sidebar.selectbox("Choose a page", ("Project Description", "Tweet Analysis", "Dataset analysis"))
    if menu == 'Project Description':
        page_description()
    elif menu == 'Tweet Analysis':  
        page_analysis()
    elif menu == 'Dataset analysis':  
        page_dataset()
        
    # Link Twitter
    st.sidebar.markdown("---")
    st.sidebar.image(os.path.join(folder_images, "twitter_logo.png"), width=50)
    st.sidebar.markdown('<a href="http://twitter.com">Twitter</a>', unsafe_allow_html=True)
    
    # Our names
    st.sidebar.text("")
    st.sidebar.write('<h1 style="text-align:center;color:#BDBDBD;font-weight:normal;font-size:13px;">Created by Léna, Aurélien, Florian, Mélina, and Lucie</h1>',unsafe_allow_html=True)

    
    
    
if __name__ == '__main__':
    main()
