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
import streamlit.components.v1 as components



st.set_page_config(page_title="DataZilla",page_icon='zilla.jpeg',  initial_sidebar_state = 'auto')
#st.image('/Users/vaigupta/Documents/DS/StreamLit/zilla.jpeg',width=500,use_column_width=True)



hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.title("Classfication Algorthims Handy")

data_load=st.sidebar.file_uploader("Upload your CSV",type='csv')


if data_load is not None:
    row_display=st.sidebar.slider("Show Rows",0,100,10,10)
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
    st.markdown("""
                    
                                        <!DOCTYPE html>
                    <html lang="en">
                    <head>
                      <title>Bootstrap Example</title>
                      <meta charset="utf-8">
                      <meta name="viewport" content="width=device-width, initial-scale=1">
                      <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.4.1/css/bootstrap.min.css">
                      <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.5.1/jquery.min.js"></script>
                      <script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.4.1/js/bootstrap.min.js"></script>
                    </head>
                    <body>
                    
                    
                      <h2>Modal Example</h2>
                      <!-- Trigger the modal with a button -->
                      <button type="button" class="btn btn-info btn-lg" data-toggle="modal" data-target="#myModal">Open Modal</button>
                    
                      <!-- Modal -->
                      <div class="modal fade" id="myModal" role="dialog">
                        <div class="modal-dialog">
                        
                          <!-- Modal content-->
                          <div class="modal-content">
                            <div class="modal-header">
                              <button type="button" class="close" data-dismiss="modal">&times;</button>
                              <h4 class="modal-title">Modal Header</h4>
                            </div>
                            <div class="modal-body">
                              <p>Some text in the modal.</p>
                            </div>
                            <div class="modal-footer">
                              <button type="button" class="btn btn-default" data-dismiss="modal">Close</button>
                            </div>
                          </div>
                          
                        </div>
                      </div>
                    
                    </body>
                    </html>
                    
                    """)






