# FUTURE_ML_01
 Superstore Sales Forecasting — Future Interns ML Internship
A machine learning project that forecasts daily sales using the Sample Superstore dataset and a Random Forest model.

📌 Project Overview
This project predicts future sales based on historical order data. It uses date-based feature engineering and a Random Forest Regressor to generate forecasts, then visualizes actual vs predicted sales on a chart.

🧰 Tech Stack

Python
Pandas, NumPy
Scikit-learn (RandomForestRegressor)
Matplotlib


📁 Dataset
Sample - Superstore.csv — a retail dataset containing order dates, sales, categories, and more.

⚙️ How It Works

Loads and parses the Superstore CSV
Groups sales by date
Extracts features: Year, Month, DayOfWeek, Quarter
Trains Random Forest on 80% of data
Predicts on remaining 20%
Evaluates using Mean Absolute Error (MAE)
Plots Actual vs Forecast chart


▶️ Run
bashpip install pandas numpy scikit-learn matplotlib
python task1.py

📊 Output

MAE score printed in terminal
Line chart: Actual Sales vs AI Forecas
