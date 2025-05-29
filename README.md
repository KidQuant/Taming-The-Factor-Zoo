# Taming The Factor Zoo

## Overview

Taming the Factor Zoo is a comprehensive research project aimed at predicting stock-live risk premiums using advanced statistical learning techniques. Inspired by the proliferation of risk factors in asset pricing literature-often referred to as the "factor zoo"- this project seeks to identify and utilize the most predictive firm-specific characters and macroeconomic variables to forecast excess stock returns.

## Project Objectives

* **Predictive Modeling**: Identify which among the numerous firm-specific charactersitics contribute significantly to the prediction of risk premiums.

* **Factor Selection**: Identify which among the numerous firm-specific characteristics contribute significantly to the prediction of risk premiums.

* **Portfolio Construction**: Utilize model predictions to construct portfolios and assess their performance metrics, such as average risk premium, volatility, and Sharpe ratio.

## Data Description

The dataset encompassess:

* **Time Span**: 60 years of monthly data.

* **Stocks**: Approximately 30,000 unique stocks.

* **Firm-Specific Characteristics**: 94 variables per stock, capturing various financial and operational metrics.

* **Macroeconomic Variables**: 8 aggregate time-series variables representing broader economic conditions.

The primary target variable is the risk premium, defined as the difference between a stock's return and the risk-free rate (approximated by the three-month U.S. Treasury Bill rate).

## Methodology

The project follows a structured approach:

1. **Data Preprocessing**: Cleaning and preparing the dataset for analysis.
2. **Exploratory Data Analysis (EDA)**: Understanding data distributions, relationships, and identifying potential anomalies.
3. **Model Development**: Implementing various statistical learning models, including:

    * Linear Regression (with varying numbers of factors)
    * Ordinary Least Squares (OLS)
    * Elastic Net Regression
    * Partial Least Squares (PLS)
    * Principal Component Regression (PCR)
    * Neural Networks (NN)
    * Gradient Boosted Regression Trees (GBRT)

4. **Model Evaluation**: Assessing models based on predictive performance metrics like the Out-of-Sample (OOS) $R^2$ and Mean Squared Error (MSE).

5. **Portfolio Analysis**: Constructing equal-weight portfolios of the top 100 stocks based on predicted risk-premiums and evaluating their performance.