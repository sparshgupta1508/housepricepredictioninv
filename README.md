# House Price Prediction Web App

A web application that predicts house prices based on user-provided features using a neural network model trained on real estate data. The app is built with **Flask**, **TensorFlow**, and **scikit-learn**, and features a responsive, modern frontend.

---

## Features

- Predict house prices using 15 property features:
  - Bedrooms, Bathrooms, Living space, Lot space, Floors
  - Waterfront, View, Condition, Grade
  - Above ground area, Basement, Year built, Year renovated
  - Latitude, Longitude
- Responsive and user-friendly interface with animated loading spinner.
- Displays predicted price instantly without page reload.
- Handles invalid input gracefully.

---

## Technologies Used

- **Frontend**: HTML, CSS, JavaScript (Fetch API)
- **Backend**: Flask
- **Machine Learning**: TensorFlow (Keras), scikit-learn
- **Data Preprocessing**: MinMaxScaler (scikit-learn)

---
## Project Structure
housepricepredictioninv/

- app.py # Flask application
- house_price_prediction_model.h5 # Trained neural network model
- scaler.pkl # Saved scaler object for feature scaling
- templates/
- index.html # Frontend HTML page
- static/
- (optional: CSS/JS/images)

