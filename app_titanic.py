# źródło danych [https://www.kaggle.com/c/titanic/](https://www.kaggle.com/c/titanic)

import streamlit as st
import pickle
from datetime import datetime
import pandas as pd

startTime = datetime.now()

import pathlib
from pathlib import Path

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# nie bardzo wiem jak się dobrać do modelu, żeby dostać min/max/mean, zatem robie jak w poprzednim zadaniu
csv_df = pd.read_csv('1/DSP_1.csv')

filename = "1/model.sv"
model = pickle.load(open(filename,'rb'))

pclass_d = {0:"Pierwsza",1:"Druga", 2:"Trzecia"}
embarked_d = {0:"Cherbourg", 1:"Queenstown", 2:"Southampton"}
sex_d = {0: 'Kobieta', 1: 'Mężczyzna'}

def main():

	st.set_page_config(page_title="Titanic")
	overview = st.container()
	left, right = st.columns(2)
	prediction = st.container()

	st.image("https://i.imgur.com/NcCz6we.jpeg", "https://i.imgur.com/NcCz6we.jpeg")

	with overview:
		st.title("Uczymy się ML z Titaniciem")

	with left:
		pclass_radio = st.radio("Klasa", list(pclass_d.keys()), format_func=lambda x : pclass_d[x])
		sex_radio = st.radio( "Płeć", list(sex_d.keys()), format_func=lambda x : sex_d[x] )
		embarked_radio = st.radio( "Port zaokrętowania", list(embarked_d.keys()), index=2, format_func= lambda x: embarked_d[x] )

	with right:
		age_median=round(csv_df['Age'].median())
		age_min=round(csv_df['Age'].min())
		age_max=round(csv_df['Age'].max())
		age_slider = st.slider("Wiek", value=age_median, min_value=age_min, max_value=age_max)

		SibSp_median=round(csv_df['SibSp'].median())
		SibSp_min=round(csv_df['SibSp'].min())
		SibSp_max=round(csv_df['SibSp'].max())
		sibsp_slider = st.slider("Liczba rodzeństwa i/lub partnera", value=SibSp_median, min_value=SibSp_min, max_value=SibSp_max)

		Parch_median=round(csv_df['Parch'].median())
		Parch_min=round(csv_df['Parch'].min())
		Parch_max=round(csv_df['Parch'].max())
		parch_slider = st.slider("Liczba rodziców i/lub dzieci", value=Parch_median, min_value=Parch_min, max_value=Parch_max)

		fare_median=round(csv_df['Fare'].median())
		fare_min=round(csv_df['Fare'].min())
		fare_max=round(csv_df['Fare'].max())	
		fare_slider = st.slider("Cena biletu", value=fare_median, min_value=fare_min, max_value=fare_max, step=1)

	data = [[pclass_radio, sex_radio,  age_slider, sibsp_slider, parch_slider, fare_slider, embarked_radio]]
	survival = model.predict(data)
	s_confidence = model.predict_proba(data)

	with prediction:
		st.subheader("Czy taka osoba przeżyłaby katastrofę?")
		st.subheader(("Tak" if survival[0] == 1 else "Nie"))
		st.write("Pewność predykcji {0:.2f} %".format(s_confidence[0][survival][0] * 100))

if __name__ == "__main__":
    main()
