import streamlit as st
import pandas as pd
import numpy as np
import pickle  # to load a saved model

st.title('IRIS CLASSIFICATION')
st.image("iris.png")
st.markdown('Dataset :')
data = pd.read_csv('iris.csv')
st.write(data.head())
sepalLengthCM = st.slider('Sepal Length', 0.0,10.0, 1.0)
sepalWidthCM = st.slider('Sepal Width', 0.0,10.0, 1.0)
petalLengthCM = st.slider('Petal Length', 0.0,10.0, 1.0)
petalWidthCM = st.slider('Petal Width', 0.0,10.0, 1.0)

if st.button("Click to Predict"):    
    feature_list = [sepalLengthCM,sepalWidthCM,petalLengthCM,petalWidthCM,]            
    single_sample = np.array(feature_list).reshape(1, -1)
    loaded_model = pickle.load(open('model.pkl', "rb"))
    prediction = loaded_model.predict(single_sample)
    if (prediction[0]=="Iris-virginica"):
        st.image("virginica.png")
    elif (prediction[0]=="Iris-versicolor"):
        st.image("versicolor.png")
    else :
        st.image("setosa.png")
