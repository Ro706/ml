# Crime Against Women in India: Analysis and Forecasting

## Introduction

This project aims to analyze the trends of crimes against women in India using a dataset of reported cases. The project involves exploratory data analysis (EDA) to understand the distribution and trends of different crimes, and time series forecasting to predict future crime rates.

## Dataset

The dataset used in this project is `crime_aginest_women.csv`, which contains data on various crimes against women reported in different states and districts of India from 2001 to 2014.

The dataset includes the following columns:
- `year`: The year the crimes were reported.
- `state_name`: The name of the state.
- `district_name`: The name of the district.
- Various columns for different types of crimes, such as `murder_with_rape_gang_rape`, `dowry_deaths`, `acid_attack`, etc.

## Exploratory Data Analysis

The EDA is performed in the `eda/eda.py` script. The script performs the following steps:
1.  Loads the dataset and displays summary statistics.
2.  Checks for missing values.
3.  Generates and saves the following plots in the `eda/plots` directory:
    *   `crime_distribution.png`: A bar plot showing the distribution of different crimes.
    *   `total_crimes_over_years.png`: A line plot showing the trend of total crimes over the years.
    *   `severe_crimes_over_years.png`: A line plot showing the trend of some of the most severe crimes over the years.

## Time Series Forecasting

The `eda/model.py` script performs time series forecasting to predict the total number of crimes for the next 5 years. The script uses Holt's linear trend model for forecasting. The forecast plot is saved as `eda/plots/forecast.png`.

## Regression Analysis

The `eda/eda-2.py` script builds a linear regression model to predict the total number of crimes based on a set of selected crime features. The script also generates a scatter plot of actual vs. predicted values, which is saved as `eda/plots/actual_vs_predicted.png`.

## How to Run

1.  **Install the required libraries:**
    ```
    pip install -r requirements.txt
    ```
2.  **Run the EDA script:**
    ```
    python eda/eda.py
    ```
3.  **Run the time series forecasting script:**
    ```
    python eda/model.py
    ```
4.  **Run the regression analysis script:**
    ```
    python eda/eda-2.py
    ```

## Other Files

The `sklearn` directory and the files `MLtask.md`, `mlREADME.md`, and the original `README.md` contain generic machine learning examples and are not part of this project.