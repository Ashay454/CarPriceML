# 🚗 Car Price Prediction ML Project

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange)
![Status](https://img.shields.io/badge/Status-Completed-green)

A Machine Learning pipeline to predict the selling price of cars based on various features like car age, mileage, fuel type, etc. This project utilizes a **Random Forest Regressor** integrated within a robust `Scikit-Learn Pipeline` to handle preprocessing and modeling seamlessly.

## 📊 Project Overview

Buying a used car can be tricky. This tool helps users estimate the fair market value of a car.
The project covers:
1.  **Data Analysis**: Exploratory Data Analysis (EDA) to understand feature correlations.
2.  **Preprocessing**: Automatic handling of categorical (OneHotEncoding) and numerical (StandardScaling) data.
3.  **Modeling**: Training a Random Forest Regressor.
4.  **Evaluation**: Measuring performance using RMSE and R2 Score.

## 🛠️ Tech Stack & Features

* **Core Library**: `scikit-learn`, `pandas`, `numpy`
* **Model**: Random Forest Regressor
* **Pipeline**: Uses `ColumnTransformer` and `Pipeline` to prevent data leakage and simplify deployment.
* **Persistence**: Saves the trained model using `joblib` for future use.

## 📂 Project Structure

```text
CarPriceML/
│
├── data/                   # Raw dataset (car_data.csv)
├── notebooks/              # Jupyter notebooks for EDA and experiments
├── outputs/                # Generated plots and metrics
│   ├── actual_vs_predicted.png
│   ├── correlation_heatmap.png
│   └── ...
├── src/                    # Source code for the project
│   ├── model.py            # Model definition (Pipeline & Random Forest)
│   ├── data_preprocessing.py
│   ├── train.py            # Script to train and save the model
│   └── predict.py          # Script to make predictions
├── .gitignore              # Files to ignore (venv, __pycache__)
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation
