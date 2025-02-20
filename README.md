# 🏆 Gold Price Prediction

## 📌 Project Overview
This project focuses on predicting gold prices based on historical financial data using machine learning techniques. The dataset contains multiple economic indicators that influence gold prices, which are used to train a regression model.

## 📊 Dataset
The dataset contains multiple features, including:
- **📅 Date**
- **📈 S&P 500 Index (SPX)**
- **💰 Gold ETF (GLD)**
- **🛢 Crude Oil ETF (USO)**
- **🥈 Silver ETF (SLV)**
- **💱 EUR/USD Exchange Rate**

## 🛠 Dependencies
The following Python libraries are required to run the notebook:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
```

## 🚀 Installation
Ensure you have Python installed along with the required libraries. You can install them using:
```sh
pip install numpy pandas scikit-learn matplotlib seaborn
```

## 🔄 Data Processing
1️⃣ Load the dataset using Pandas.  
2️⃣ Perform data cleaning by handling missing values.  
3️⃣ Normalize or scale features if needed.  
4️⃣ Split the dataset into training and testing sets.  

## 💡 Model Training
- The **Random Forest Regressor** model is used for price prediction.
- The dataset is split into **training (80%) and testing (20%)** sets.
- The model is trained on the training dataset.

## 📈 Model Evaluation
- The performance of the trained model is evaluated using:
  - **📉 Mean Absolute Error (MAE)**
  - **📊 Mean Squared Error (MSE)**
  - **📈 R-squared Score (R²)**
- The evaluation metrics are printed as output.

## ▶️ How to Run
1️⃣ Open the `Gold_Price_Prediction.ipynb` file in **Jupyter Notebook** or **Google Colab**.  
2️⃣ Execute the cells sequentially to process the data and train the model.  
3️⃣ Observe the performance metrics of the model.  

## 🔥 Future Enhancements
- Implement **additional machine learning models** (e.g., **Linear Regression, XGBoost**) for comparison.  
- Improve **feature engineering** for better predictions.  
- Tune **hyperparameters** for better accuracy.  
- Deploy the model as a **web application** for user interaction.  

## 🤝 Contribution
Feel free to contribute by **opening an issue** or **submitting a pull request**.  

## 📜 License
This project is licensed under the **MIT License**.
