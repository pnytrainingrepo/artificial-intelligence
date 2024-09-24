# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 00:27:47 2022

@author: lipim
"""

import pickle
import streamlit as st
from streamlit_option_menu import option_menu

load_diabetes = pickle.load(open("diabetes.pkl", 'rb'))
load_heart = pickle.load(open("heart.pkl", 'rb'))
load_breast = pickle.load(open("breast.pkl", 'rb'))

with st.sidebar:
    selected = option_menu('Multiple Disease Prediction',
                           ['Diabetes Prediction', 'Heart Disease Prediction', 'Breast Cancer Prediction'],
                           icons=['activity', 'heart', 'cast'], default_index=0)

if (selected == "Diabetes Prediction"):
    st.title("Diabetes Prediction")
    c1, c2, c3 = st.columns(3)
    with c1:
        Pregnancies = st.number_input('Pregnancies')
    with c2:
        Glucose = st.number_input('Glucose')
    with c3:
        BloodPressure = st.number_input('BloodPressure')
    with c1:
        SkinThickness = st.number_input('SkinThickness')
    with c2:
        Insulin = st.number_input('Insulin')
    with c3:
        BMI = st.number_input('BMI')
    with c1:
        DiabetesPedigreeFunction = st.number_input('DiabetesPedigreeFunction')
    with c2:
        Age = st.number_input('Age')

    diab_diagnosis = ''

    if st.button('Diagnosis'):
        diab_diagnosis = load_diabetes.predict(
            [[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
        if (diab_diagnosis[0] == 0):
            diab_diagnosis = 'The person is not diabetic.'
        else:
            diab_diagnosis = 'The person is diabetic.'
    st.success(diab_diagnosis)

if (selected == "Heart Disease Prediction"):
    st.title("Heart Disease Prediction")

    c1, c2, c3 = st.columns(3)

    with c1:
        age = st.number_input('Age')
    with c2:
        sex = st.number_input('Sex')
    with c3:
        cp = st.number_input('Chest Pain')
    with c1:
        trestbps = st.number_input('Blood Pressure')
    with c2:
        chol = st.number_input('Cholestoral')
    with c3:
        fbs = st.number_input('Fasting Blood Sugar')
    with c1:
        restecg = st.number_input('Electrocardiographic Result')
    with c2:
        thalach = st.number_input('Maximum Heartbeat Rate')
    with c3:
        exang = st.number_input('Exercise induced angina')
    with c1:
        oldpeak = st.number_input('ST depression')
    with c2:
        slope = st.number_input('Slope of peak Exercise ST segment')
    with c3:
        ca = st.number_input('Number of Major vessels colored by Flourosopy')
    with c1:
        thal = st.number_input('Thal')

    heart_diagnosis = ''

    if st.button('Diagnosis'):
        heart_diagnosis = load_heart.predict(
            [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
        if (heart_diagnosis[0] == 0):
            heart_diagnosis = 'The person is suffer from Heart Disease.'
        else:
            heart_diagnosis = 'The person is not suffer from Heart Disease.'

    st.success(heart_diagnosis)

if (selected == "Breast Cancer Prediction"):
    st.title("Breast Cancer Prediction")

    c1, c2, c3 = st.columns(3)

    with c1:
        radius_mean = st.number_input('Radius Mean')
    with c2:
        texture_mean = st.number_input('Texture Mean')
    with c3:
        perimeter_mean = st.number_input('Perimeter Mean')
    with c1:
        area_mean = st.number_input('Area Mean')
    with c2:
        smoothness_mean = st.number_input('Smoothness Mean')
    with c3:
        compactness_mean = st.number_input('Compactness Mean')
    with c1:
        concavity_mean = st.number_input('Concavity Mean')
    with c2:
        concave_points_mean = st.number_input('Concave Points Mean')
    with c3:
        symmetry_mean = st.number_input('Symmetry Mean')
    with c1:
        fractal_dimension_mean = st.number_input('Fractal Dimension Mean')
    with c2:
        radius_se = st.number_input('Radius se')
    with c3:
        texture_se = st.number_input('Texture se')
    with c1:
        perimeter_se = st.number_input('Perimeter se')
    with c2:
        area_se = st.number_input('Area se')
    with c3:
        smoothness_se = st.number_input('Smoothness se')
    with c1:
        compactness_se = st.number_input('Compactness se')
    with c2:
        concavity_se = st.number_input('Concavity se')
    with c3:
        concave_points_se = st.number_input('Concave Points se')
    with c1:
        symmetry_se = st.number_input('Symmetry se')
    with c2:
        fractal_dimension_se = st.number_input('Fractal Dimension se')
    with c3:
        radius_worst = st.number_input('Radius Worst')
    with c1:
        texture_worst = st.number_input('Texture Worst')
    with c2:
        perimeter_worst = st.number_input('Perimeter Worst')
    with c3:
        area_worst = st.number_input('Area Worst')
    with c1:
        smoothness_worst = st.number_input('Smoothness Worst')
    with c2:
        compactness_worst = st.number_input('Compactness Worst')
    with c3:
        concavity_worst = st.number_input('Concavity Worst')
    with c1:
        concave_points_worst = st.number_input('Concave Points Worst')
    with c2:
        symmetry_worst = st.number_input('Symmetry Worst')
    with c3:
        fractal_dimension_worst = st.number_input('Fractal Dimension Worst')

    breast_diagnosis = ''

    if st.button('Diagnosis'):
        breast_diagnosis = load_breast.predict([[radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean,
                                                 compactness_mean, concavity_mean, concave_points_mean, symmetry_mean,
                                                 fractal_dimension_mean, radius_se, texture_se, perimeter_se, area_se,
                                                 smoothness_se, compactness_se, concavity_se, concave_points_se,
                                                 symmetry_se, fractal_dimension_se, radius_worst, texture_worst,
                                                 perimeter_worst, area_worst, smoothness_worst, compactness_worst,
                                                 concavity_worst, concave_points_worst, symmetry_worst,
                                                 fractal_dimension_worst]])
        if (breast_diagnosis[0] == 1):
            breast_diagnosis = 'Malignant'
        else:
            breast_diagnosis = 'Brillient'

    st.success(breast_diagnosis)