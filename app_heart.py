# źródło danych [https://www.kaggle.com/c/titanic/](https://www.kaggle.com/c/titanic)

import string
import streamlit as st
import pickle
from datetime import datetime
import pandas as pd

startTime = datetime.now()

import pathlib
from pathlib import Path

temp = pathlib.PosixPath
pathlib.WindowsPath = pathlib.PosixPath

# nie bardzo wiem jak się dobrać do modelu, żeby dostać min/max/mean, zatem robie jak w poprzednim zadaniu
csv_df = pd.read_csv('./2/DSP_8.csv')
model = pickle.load(open("./2/model.sv",'rb'))

# mapowania LabelEncodera:
# Sex Mappings:
#  F -> 0
#  M -> 1
# ChestPainType Mappings:
#  ASY -> 0
#  ATA -> 1
#  NAP -> 2
#  TA -> 3
# ST_Slope Mappings:
#  Down -> 0
#  Flat -> 1
#  Up -> 2
# ExerciseAngina Mappings:
#  N -> 0
#  Y -> 1
# RestingECG Mappings:
#  LVH -> 0
#  Normal -> 1
#  ST -> 2
# Random Forest: 0.997275204359673

sex_d = {0: 'Female', 1: 'Male'}
ChestPainType_d = {0:"ASY",1:"ATA", 2:"NAP", 3: "TA"}
ST_Slope_d = {0:"Down", 1:"Flat", 2:"Up"}
ExerciseAngina_d = {0:"No", 1:"Yes"}
RestingECG_d = {0:"LVH", 1:"Normal", 2:"ST"}
Fasting_d = {0:"No", 1:"Yes"}

def createSlider(title: string, key: string):
	median=round(csv_df[f"{key}"].median())
	min=round(csv_df[f"{key}"].min())
	max=round(csv_df[f"{key}"].max())	
	return st.slider(title, value=median, min_value=min, max_value=max)

def main():
	st.set_page_config(page_title="ML MD")
	overview = st.container()
	left, middle, right = st.columns(3)
	prediction = st.container()

	st.image("https://i.imgur.com/rs92dAd.jpeg", "https://i.imgur.com/rs92dAd.jpeg")

	with overview:
		st.title("Machine Learning MD")

	with left:
		sex_radio = st.radio( "Gender", list(sex_d.keys()), format_func=lambda x : sex_d[x] )
		chestpain_radio = st.radio("Chest Pain type", list(ChestPainType_d.keys()), format_func=lambda x : ChestPainType_d[x])
		ST_Slope_radio = st.radio( "ST Slope", list(ST_Slope_d.keys()), index=1, format_func= lambda x: ST_Slope_d[x] )
		
	with middle:
		ExerciseAngina_radio = st.radio( "Exercise induced angina", list(ExerciseAngina_d.keys()), index=1, format_func= lambda x: ExerciseAngina_d[x] )
		RestingECG_radio = st.radio( "Resting electrocardiographic results", list(RestingECG_d.keys()), index=1, format_func= lambda x: RestingECG_d[x] )
		Fasting_radio = st.radio( "Fasting blood sugar", list(Fasting_d.keys()), index=1, format_func= lambda x: Fasting_d[x] )

	with right:
		age_slider = createSlider("Age of the person", "Age")
		restingbp_slider = createSlider("Resting blood pressure", "RestingBP")
		chole_slider = createSlider("Cholesterol", "Cholesterol")
		maxHR_slider = createSlider("Maximum heart rate", "MaxHR")
		oldpeak_slider = createSlider("Previous peak", "Oldpeak")

	data = [[
			age_slider, sex_radio, chestpain_radio, restingbp_slider, 
		  	chole_slider, Fasting_radio, RestingECG_radio, maxHR_slider,
		  	ExerciseAngina_radio, oldpeak_slider, ST_Slope_radio]]

	survival = model.predict(data)
	
	s_confidence = model.predict_proba(data)
	with prediction:
		st.subheader("Am i prone to a heart disease?")
		st.subheader(("YES" if survival[0] == 1 else "NO"))
		st.write("Confidence {0:.2f} %".format(s_confidence[0][survival][0] * 100))

if __name__ == "__main__":
    main()
