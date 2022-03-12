#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# import streamlit and transformers
import transformers
import streamlit as st  



# In[ ]:



# In[ ]:


## create streamlit framework
def main():
    st.title('Text Classification Streamlit App')

    if 'count' not in st.session_state:
        st.session_state.count = 1
        st.write('Press Load GUI to continue')
        st.write(st.session_state.count)
        from transformers import pipeline
        with st.spinner('Loading...'):
            classifier = pipeline("zero-shot-classification", device=0)
            default_labels = ['Positive','Negative','Neutral']




    load = st.button('Load GUI')


    if load or st.session_state.count == 1:

        st.header('Zero-shot training model demonstration')
        st.write('Default labels: Positive, Negative, Neutral')
        st.subheader('Do you want to use custom labels?')
        user_labels = [] #labels for classification
        choice = st.checkbox(label='Pick yes to use custom labels')
        if choice:
            label_no = st.slider('Number of labels',3,5,1,1)
            for i in range(label_no):
                label = st.text_input("Label number {} here".format(i+1))
                user_labels.append(label)
        else: 
            user_labels = default_labels
        st.subheader("Input your sentence")
        user_input = st.text_input("Type your sentence here","This is an amazing day!")
        pred_class = classifier(user_input,user_labels)['labels'][0]
        st.write('Your sentence belongs in class: ',pred_class)
        st.write(st.session_state.count)
        
    
if __name__ == 'main' :
    main()
            


# In[ ]:




