#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 00:43:43 2020

@author: vaigupta
"""

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,recall_score,precision_score,accuracy_score


hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.title("Classfication Algorthims Handy")

data_load=st.sidebar.file_uploader("Upload your CSV",type='csv')
row_display=st.sidebar.slider("Show Rows",0,100,10,10)

if data_load is not None:
    model_input=pd.read_csv(data_load)
    
    if st.sidebar.button("Show Data"):
        st.markdown("## *Data Frame Shape* ##")
        st.write(model_input.shape)
        st.markdown("## *Data Frame Data Types* ##")
        st.write(model_input.dtypes) 
        
        st.markdown("## *Top {} rows displayed:* ##".format(row_display))
        st.dataframe(model_input, height=200)
        #pair=sns.pairplot(data=model_input)
        #st.pyplot(pair)

    y_var=st.sidebar.selectbox("Select your Y Variable",model_input.columns)        
    
    X=model_input.drop(columns=y_var)
    Y=model_input[y_var]
    


    model_type=st.sidebar.selectbox("Select your Classification Algorithm",['Logistic Regression','Decision Tree','Random Forrest','Neural Network'])

    train_size=st.sidebar.slider("Select your Train Size",0,100,80,10)
    outlier = st.sidebar.checkbox("Remove Outlier")
   
    if st.sidebar.button("Run Model"):
        st.markdown("# Y variable #")
        st.write(Y)
        X_train ,X_test, Y_train , Y_test = train_test_split(X,Y,train_size=train_size)
        LR = LogisticRegression()
        Model_O=LR.fit(X_train,Y_train)
        
        Y_pred = Model_O.predict(X_test)
        
        conf = confusion_matrix(Y_test,Y_pred)
        st.write(conf)
        
        precision = precision_score(Y_test,Y_pred)
        st.write("Precision Score:{}".format(precision))
        
        recall_score = recall_score(Y_test,Y_pred)
        st.write("Recall Score:{}".format(recall_score))        
        
else:
    st.write("Please upload a data source")    





