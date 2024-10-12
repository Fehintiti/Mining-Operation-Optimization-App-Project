# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 23:49:38 2024

@author: USER
"""

import pickle
import streamlit as st
import pandas as pd
import joblib
from streamlit_option_menu import option_menu
import os

# Load models and preprocessors
def load_models():
    maintenance_model = joblib.load('best_rf_maintenance_model.pkl')
    fuel_model = joblib.load('best_rf_fuel_model.pkl')
    scaler = joblib.load('scaler.pkl')
    label_encoders = joblib.load('label_encoders.pkl')
    return maintenance_model, fuel_model, scaler, label_encoders

# Feature engineering for 'stress_index' only (no 'fuel_per_km')
def feature_engineering(df):
    if 'temperature_celsius' in df.columns and 'humidity_percent' in df.columns:
        df['stress_index'] = df['temperature_celsius'] * df['humidity_percent'] / 100
    else:
        df['stress_index'] = 0  # Placeholder or default value if data is missing
    
    return df

# Preprocessing function without 'fuel_per_km'
def preprocess_data(df, scaler, label_encoders, task, trained_columns):
    df = feature_engineering(df)
    
    # Encode categorical variables
    for column in ['operation_mode', 'terrain_type']:
        if column in df.columns:
            df[column] = label_encoders[column].transform(df[column])
    
    # Drop task-specific columns before scaling
    if task == 'Maintenance':
        df = df.drop(columns=['maintenance_flag', 'fuel_consumption_liters'], errors='ignore')
    elif task == 'Fuel':
        df = df.drop(columns=['fuel_consumption_liters', 'maintenance_flag', 'downtime_hours'], errors='ignore')
    
    # Scale numerical features
    scaled_columns = ['load_weight_tonnes', 'distance_traveled_km', 'temperature_celsius', 
                      'humidity_percent', 'stress_index']
    
    df[scaled_columns] = scaler.transform(df[scaled_columns])
    
    # Ensure the column order matches the trained model
    df = df[trained_columns]
    
    return df

# Main function for the app
def main():
    # App title
    st.title("Mining Company's: Maintenance & Fuel Efficiency Prediction")

    # **Project Overview**
    st.markdown("""
    ### Overview:
    This app is developed to support **Mining Company's sustainability** goals by providing predictive insights on **equipment maintenance** and **fuel efficiency**. 
    Mining operations involve heavy machinery that requires careful monitoring to avoid breakdowns and minimize downtime. At the same time, fuel consumption is a significant part of operational costs and environmental impact.

    - **Maintenance Prediction** allows operational teams to predict when equipment might need maintenance based on performance and environmental factors, reducing unscheduled downtime and optimizing maintenance schedules.
    
    - **Fuel Efficiency Prediction** helps forecast fuel usage based on real-time data, allowing for more efficient operations, fuel savings, and a reduction in the carbon footprint, contributing to Rio Tinto’s commitment to **sustainable mining** practices.
    """)

    # Sidebar Navigation
    with st.sidebar:
        selected = option_menu(
            'Prediction Menu',
            ['Maintenance Prediction', 'Fuel Efficiency Prediction'],
            icons=['tools', 'fuel-pump'],
            default_index=0
        )

    # Load models and preprocessing objects
    maintenance_model, fuel_model, scaler, label_encoders = load_models()

    # Define the column order based on training (no 'fuel_per_km')
    trained_columns_maintenance = ['operation_mode', 'terrain_type', 'load_weight_tonnes',
                                   'distance_traveled_km', 'downtime_hours', 'temperature_celsius',  
                                   'humidity_percent', 'sensor_failure', 'stress_index']
    
    trained_columns_fuel = ['operation_mode', 'terrain_type', 'load_weight_tonnes',
                            'distance_traveled_km', 'temperature_celsius', 'humidity_percent', 
                            'sensor_failure', 'stress_index']  # 'fuel_per_km' removed



    # Maintenance Prediction Page
    if selected == 'Maintenance Prediction':
        st.subheader("Predict Equipment Maintenance")

        # Input fields for Maintenance Prediction
        col1, col2, col3 = st.columns(3)
        with col1:
            operation_mode = st.selectbox("Operation Mode", ["manual", "automatic"])
            load_weight = st.number_input("Load Weight (tonnes)", min_value=0.0, step=0.01)
            sensor_failure = st.selectbox("Sensor Failure (1 = Yes, 0 = No)", [1, 0])

        with col2:
            terrain_type = st.selectbox("Terrain Type", ["rocky", "sandy", "mixed", "clay"])
            distance_traveled = st.number_input("Distance Traveled (km)", min_value=0.0, step=0.01)
            temperature = st.number_input("Temperature (°C)", min_value=-50.0, max_value=50.0, step=0.1)

        with col3:
            humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, step=0.1)
            downtime_hours = st.number_input("Downtime Hours", min_value=0.0, step=0.1)

        # Prepare data for Maintenance Prediction
        maintenance_input_data = {
            'operation_mode': [operation_mode],
            'terrain_type': [terrain_type],
            'load_weight_tonnes': [load_weight],
            'distance_traveled_km': [distance_traveled],
            'temperature_celsius': [temperature],
            'humidity_percent': [humidity],
            'downtime_hours': [downtime_hours],
            'sensor_failure': [sensor_failure],
            'stress_index': [temperature * humidity / 100]
        }

        df_input_maintenance = pd.DataFrame(maintenance_input_data)
        df_processed_maintenance = preprocess_data(df_input_maintenance, scaler, label_encoders, task='Maintenance', trained_columns=trained_columns_maintenance)

        # Maintenance Prediction
        if st.button('Predict Maintenance'):
            maintenance_predictions = maintenance_model.predict(df_processed_maintenance)
            if maintenance_predictions[0] == 1:
                st.success("Maintenance Needed")
            else:
                st.success("No Maintenance Needed")

    # Fuel Efficiency Prediction Page
    if selected == 'Fuel Efficiency Prediction':
        st.subheader("Predict Fuel Efficiency")
        

        # Input fields for Fuel Efficiency Prediction
        col1, col2, col3 = st.columns(3)
        with col1:
            operation_mode = st.selectbox("Operation Mode", ["manual", "automatic"])
            load_weight = st.number_input("Load Weight (tonnes)", min_value=0.0, step=0.01)
            sensor_failure = st.selectbox("Sensor Failure (1 = Yes, 0 = No)", [1, 0])

        with col2:
            terrain_type = st.selectbox("Terrain Type", ["rocky", "sandy", "mixed", "clay"])
            distance_traveled = st.number_input("Distance Traveled (km)", min_value=0.0, step=0.01)
            temperature = st.number_input("Temperature (°C)", min_value=-50.0, max_value=50.0, step=0.1)

        with col3:
            humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, step=0.1)
            

        # Prepare data for Fuel Efficiency Prediction
        fuel_input_data = {
            'operation_mode': [operation_mode],
            'terrain_type': [terrain_type],
            'load_weight_tonnes': [load_weight],
            'distance_traveled_km': [distance_traveled],
            'temperature_celsius': [temperature],
            'humidity_percent': [humidity],
            'sensor_failure': [sensor_failure],
         
            'stress_index': [temperature * humidity / 100]
        }

        df_input_fuel = pd.DataFrame(fuel_input_data)
        df_processed_fuel = preprocess_data(df_input_fuel, scaler, label_encoders, task='Fuel', trained_columns=trained_columns_fuel)

        # Fuel Efficiency Prediction
        if st.button('Predict Fuel Efficiency'):
            fuel_predictions = fuel_model.predict(df_processed_fuel)
            st.success(f"Predicted Fuel Consumption: {fuel_predictions[0]:.2f} liters")

if __name__ == "__main__":
    main()
