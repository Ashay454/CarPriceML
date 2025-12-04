## 🚗 Car Price Prediction Model (CarPriceML)

A complete **Machine Learning pipeline** designed to predict the selling price of used cars. The project manages data loading, cleaning, feature engineering, model training, and evaluation in a robust, modular structure powered by **Scikit-Learn Pipelines** and the **Random Forest Regressor**.

-----

### **📊 Key Features & Methodology**

| Category | Component | Detail |
| :--- | :--- | :--- |
| **Data Cleaning** | `clean_data()` | Handles unit removal (e.g., 'bhp', 'kmpl', 'CC') and converts relevant columns (`max_power`, `mileage`, `engine`) to numeric types before dropping missing values. |
| **Feature Engineering** | `feature_engineering()` | Creates the highly relevant **`car_age`** feature by subtracting the car's `year` from the current year. |
| **Preprocessing** | `ColumnTransformer` | Applies **`StandardScaler`** to numerical features and **`OneHotEncoder`** to categorical features to prevent data leakage. |
| **Model** | **Random Forest Regressor** | Used for training with `n_estimators=100` and `random_state=42` for reproducible results. |
| **Persistence** | `joblib` | Saves the entire preprocessor and model pipeline to the file **`car_price_model.pkl`**. |

-----

### **🛠️ Technologies Used**

  * **Core Languages:** Python 3+
  * **Data Handling:** `pandas`, `numpy`, `datetime`
  * **Machine Learning:** `scikit-learn` (Pipeline, ColumnTransformer, RandomForestRegressor)
  * **Visualization:** `matplotlib`, `seaborn`
  * **Serialization:** `joblib`
  * **Command Line:** `argparse`, `logging`

-----

### **🚀 Installation & Setup**

To set up the environment and dependencies:

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/Ashay454/CarPriceML.git
    cd CarPriceML
    ```

2.  **Install dependencies:**
    *(Ensure you have a `requirements.txt` file listing all necessary libraries, e.g., `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `joblib`)*

    ```bash
    pip install -r requirements.txt
    ```

3.  **Data Placement:** Ensure your dataset, named **`car_data.csv`**, is placed inside the `data/` folder.

-----

### **🏃 Usage**

The project is run using the command-line script **`run_model.py`**.

#### **1. Train the Model and Generate Reports**

This script executes the entire ML pipeline: data loading, cleaning, feature engineering, model training, saving the model, and generating evaluation plots in the `outputs/` directory.

```bash
python run_model.py
```

  * **Output Artifacts:**
      * **`car_price_model.pkl`** (The saved pipeline)
      * **`outputs/actual_vs_predicted.png`** (Model performance plot)
      * **`outputs/residuals_dist.png`** (Error distribution plot)
      * **`outputs/correlation_heatmap.png`** (EDA plot)
      * **`outputs/car_age_vs_price.png`** (EDA plot)

#### **2. Skipping EDA**

To save time during iterative training, you can skip the time-consuming plot generation using the `--skip-eda` flag:

```bash
python run_model.py --skip-eda
```

#### **3. Making New Predictions**

Once the `car_price_model.pkl` file is generated, you can use the `predict.py` script to get a price estimate for a new car.

```bash
# Runs a prediction on a hardcoded sample car (year=2018, km_driven=40000, fuel='Diesel', etc.)
python predict.py
```

-----

### **📂 Project Structure**

```
CarPriceML/
│
├── data/                       # Contains the raw dataset (car_data.csv)
├── notebooks/                  # Jupyter notebooks for initial EDA and experiments
├── outputs/                    # Generated plots and evaluation metrics
├── src/                        # Source code for the ML pipeline
│   ├── data_preprocessing.py   # Data loading, cleaning, and feature engineering functions
│   ├── model.py                # Pipeline construction, training, and evaluation
│   ├── plot.py                 # Functions for generating visualizations (EDA and Results)
│   └── __init__.py             # Makes 'src' a package
├── car_price_model.pkl         # The final saved model pipeline
├── run_model.py                # Main execution script for the training pipeline
├── predict.py                  # Script to load the model and predict for new input
├── requirements.txt            
└── README.md                   
```

-----

### **🤝 Contributing**

If you have suggestions for improving the data cleaning, feature engineering, model selection, or the deployment process, please feel free to open an issue or submit a pull request\!

-----
