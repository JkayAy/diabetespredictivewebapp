# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 00:18:06 2023

@author: NOCAY
"""
import pandas as pd
import numpy as np
import pickle
import streamlit as st

#loading the saved model
loaded_model = pickle.load(open("C:/Users/NOCAY/Desktop/DATA SCIENCE/trained_model.sav", 'rb'))

#creating a function for prediction

def diabetes_prediction(input_data):
    
    #Changing the inputn data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)
    #reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    #standardize the input data
    #std_data = scaler.transform(input_data_reshaped)
    #print(std_data)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0]==0):
        return 'The person is not diabetic'
    else:
        return 'The person is diabetic'
    
    
def main():
    
    #giving the title
    st.title('Diabetes Prediction Web App')
    
    #getting the input data from the user
    #Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age
    
    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Glucose Level')
    BloodPressure = st.text_input('Blood Pressure Value')
    SkinThickness = st.text_input('Skinthickness Value')
    Insulin = st.text_input('Insulin Level')
    BMI = st.text_input('BMI Value')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function Value')
    Age = st.text_input('Age of the Person')
    
    #code for Prediction
    
    diagnosis = ''
    
    #creating a button for Prediction
    
    if st.button('Diabetes Test Result'):
        diagnosis = diabetes_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
    
    st.success(diagnosis)
    
    
    
if __name__=='__main__':
    main()
    
    