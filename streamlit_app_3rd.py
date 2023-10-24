import streamlit as st
import pandas as pd
import joblib

import sklearn
from xgboost import XGBRegressor


Inputs = joblib.load("Inputs.pkl")
Model = joblib.load("Model.pkl")

def prediction(Condition,Fuel_type,Transmission,Type,Production_year,Mileage_km,Power_HP):
    test_df = pd.DataFrame(columns=Inputs)
    test_df.at[0,"Condition"] = Condition
    test_df.at[0,"Fuel_type"] = Fuel_type
    test_df.at[0,"Transmission"] = Transmission
    test_df.at[0,"Type"] = Type
    test_df.at[0,"Production_year"] = Production_year
    test_df.at[0,"Mileage_km"] = Mileage_km
    test_df.at[0,"Power_HP"] = Power_HP
    st.dataframe(test_df)
    result = Model.predict(test_df)[0]
    return result


def main():
    st.title("Used Car prediction")
    Condition = st.selectbox("Condition" , ['New', 'Used'])
    Fuel_type = st.selectbox("Fuel_type" , ['Gasoline', 'Diesel', 'Electric', 'Hybrid', 'other'])
    Transmission = st.selectbox("Transmission" , ['Manual', 'Automatic'])
    Type = st.selectbox("Type" , ['small_cars', 'city_cars', 'convertible', 'compact', 'SUV',
       'sedan', 'coupe', 'station_wagon', 'minivan'])
    Production_year = st.slider("Production_year" , min_value= 2000 , max_value=2021 , value=0,step=1)
    Mileage_km = st.slider("Mileage_km" , min_value= 0 , max_value=500000000 , value=0,step=100)
    Power_HP = st.slider("Power_HP" , min_value= 69 , max_value=1398.0 , value=0,step=100)



    if st.button("predict"):

        result = prediction(Condition,Fuel_type,Transmission,Type,Production_year,Mileage_km,Power_HP)
        st.success(f'Fair Price will be around . {result}')

if __name__ == '__main__':
        main()
