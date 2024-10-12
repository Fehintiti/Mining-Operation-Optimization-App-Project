# Predictive Maintenance and Fuel Efficiency Analysis for Rio Tinto

## Project Overview
This project leverages machine learning models to predict **equipment maintenance needs** and **fuel efficiency** in a mining environment. The project aligns with **Rio Tinto’s sustainability goals** by helping optimize mining operations through predictive insights, reducing downtime, and enhancing fuel efficiency for heavy machinery.

## Aims and Objectives
- **Predictive Maintenance:** To forecast when equipment might require maintenance using operational data, thus preventing unexpected downtimes.
- **Fuel Efficiency Prediction:** To develop a model that predicts fuel consumption, contributing to more cost-effective and environmentally-friendly mining operations.
- **Support Sustainable Operations:** By providing actionable predictions, this project helps improve efficiency, reduce fuel waste, and contribute to sustainable mining practices.

## Models Used

### 1. **Random Forest Classifier (Maintenance Prediction)**
   - **Why Random Forest?**  
     Random Forest is robust for handling imbalanced datasets, making it an ideal choice for the maintenance prediction problem. It offers feature importance insights, which helps in understanding the key factors driving maintenance needs. Additionally, **class_weight='balanced'** ensures it manages the infrequency of maintenance issues effectively.
     
### 2. **Random Forest Regressor (Fuel Efficiency Prediction)**
   - **Why Random Forest Regressor?**  
     This model is effective at handling non-linear relationships between operational variables and fuel consumption. It’s ideal for predicting continuous values such as fuel usage, offering high accuracy and flexibility in modeling the fuel efficiency based on various operational parameters.

## Analysis of Results

### Maintenance Prediction (Classification)
- The maintenance prediction model was evaluated using standard classification metrics:
  - **Accuracy:** 98%
  - **Precision, Recall, F1-Score:** High across all metrics, demonstrating the model’s reliability in predicting maintenance requirements.
- **Feature Importance:** The most critical features for predicting maintenance were:
  - **Downtime hours**
  - **Operational mode**
  - **Load weight**
  
### Fuel Efficiency Prediction (Regression)
- The fuel efficiency model was evaluated using **Root Mean Squared Error (RMSE)**, which measured how well the model predicted continuous fuel consumption values:
  - **RMSE:** 12.45 liters
- The model demonstrated strong predictive power and identified important factors impacting fuel efficiency, such as **load weight** and **distance traveled**.

## Summary of Results
- **Maintenance Prediction:** The model successfully predicted equipment maintenance needs, helping prevent costly downtime.
- **Fuel Efficiency Prediction:** The model offered accurate predictions of fuel consumption, providing insights for optimizing fuel usage and contributing to cost savings.
- Both models aligned with Rio Tinto’s goals by reducing operational inefficiencies and supporting sustainability in mining operations.

## Streamlit App Overview

### App Features:
- **Maintenance Prediction:**  
  Users can input operational data like **load weight**, **temperature**, **humidity**, and **downtime hours** to predict whether the equipment will need maintenance.
  
- **Fuel Efficiency Prediction:**  
  Users can input parameters such as **distance traveled**, **operational mode**, and **environmental factors** to predict fuel consumption.

The app enables real-time decision-making by allowing operational teams to interact with the machine learning models, facilitating more efficient maintenance planning and fuel optimization.

## How to Run the App
1. Clone the repository from GitHub.
2. Install the required dependencies using:
   ```bash
   pip install -r requirements.txt
