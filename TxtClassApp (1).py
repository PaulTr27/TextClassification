#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# import streamlit and transformers

import streamlit as st  
from transformers import pipeline
with st.spinner('Loading...'):
    classifier = pipeline("zero-shot-classification", device=0)


default_labels = ['Positive','Negative','Neutral']




## create streamlit framework

st.title('Text Classification Streamlit App')

st.header('Zero-shot training model demonstration')
st.write('Default labels: Positive, Negative, Neutral')


st.subheader("Input your sentence")
user_input = st.text_input("Type your sentence here","This is an amazing day!")
pred_class = classifier(user_input,default_labels)['labels'][0]
st.write('Your sentence belongs in class: ',pred_class)

        
    


# In[ ]:




