# Volatility Forecasting in Indian Financial Markets

## Problem Statement
Financial market volatility is a key indicator for risk assessment and investment decision-making. Traditional statistical methods often fail to capture complex market patterns, leading to inaccurate volatility predictions. This project aims to develop a machine learning-based forecasting model that leverages historical Nifty 50 stock data to predict stock price volatility more accurately.

## Objective
- To build a machine learning model that forecasts financial market volatility using historical data.
- To compare the performance of linear regression and LSTM-based deep learning models in predicting stock price trends.
- To improve prediction accuracy, aiding financial analysts in better decision-making.

## Technologies Used
- **Programming Language:** Python
- **Libraries:** Pandas, NumPy, Scikit-Learn, TensorFlow, Matplotlib, Yahoo Finance API
- **Models Implemented:** Linear Regression, LSTM

## Methodology
1. **Data Collection:**
   - Historical stock data of Nifty 50 retrieved using Yahoo Finance API.
   - Features include Close price, Moving Averages (MA20, MA50), and stock Returns.

2. **Preprocessing:**
   - Handling missing values using forward fill method.
   - Normalization of stock prices and volume data.
   - Feature engineering (moving averages, return percentages).

3. **Model Building:**
   - Implemented Linear Regression as a baseline model.
   - Developed an LSTM-based deep learning model for sequence forecasting.
   - Trained models using an 80-20 train-test split.

4. **Evaluation Metrics:**
   - Mean Absolute Error (MAE)
   - Root Mean Squared Error (RMSE)
   - Comparison of actual vs. predicted prices

5. **Results & Observations:**
   - The LSTM model demonstrated improved forecasting accuracy over Linear Regression.
   - Achieved an overall **15% improvement** in prediction accuracy compared to baseline methods.
   - The model effectively captures market trends, assisting financial analysts in risk assessment.

## Conclusion
This project successfully developed a machine learning-based volatility forecasting system. The results show that deep learning methods like LSTMs can outperform traditional regression models in predicting stock market movements. Future enhancements could include additional macroeconomic indicators and sentiment analysis from financial news to improve accuracy further.

## Future Scope
- Incorporation of external factors such as economic indicators and social sentiment.
- Implementation of Reinforcement Learning for optimized stock predictions.
- Deployment of the model as a web application for real-time predictions.

