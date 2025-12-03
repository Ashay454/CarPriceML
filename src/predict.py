import joblib
import pandas as pd
import datetime

def load_model(filename='car_price_model.pkl'):
    return joblib.load(filename)

def predict_new_car(model, input_data):

    input_df = pd.DataFrame([input_data])

    if 'year' in input_df.columns:
        current_year = datetime.datetime.now().year
        input_df['car_age'] = current_year - input_df['year']
        input_df.drop('year', axis=1, inplace=True)
        
    prediction = model.predict(input_df)
    return prediction[0]

if __name__ == "__main__":
    sample_car = {
        'year': 2018,
        'km_driven': 40000,
        'fuel': 'Diesel',
        'seller_type': 'Individual',
        'transmission': 'Manual',
        'owner': 'First Owner',
        'mileage': 22.0,
        'engine': 1248.0,
        'max_power': 74.0,
        'seats': 5.0
    }
    
    try:
        loaded_model = load_model()
        price = predict_new_car(loaded_model, sample_car)
        print(f"Predicted Selling Price: {price:.2f}")
    except FileNotFoundError:
        print("Model file not found. Run 'run_model.py' first.")