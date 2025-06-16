#  Housing Price Prediction Using Machine Learning and Visualization

This project explores how machine learning and data visualization can be used to predict house prices and support real estate investment decisions. The analysis is based on the **Ames Housing Dataset**, which contains detailed information on residential properties in Ames, Iowa.

---

##  Project Overview

We trained two machine learning models — **Random Forest Regressor** and **Gradient Boosting Regressor** — to predict house prices based on features like living area size, year built, overall quality, garage area, and neighborhood.

The results of the models were evaluated using standard metrics and visualized through an interactive **Tableau dashboard** to uncover market trends and provide actionable insights.

---

##  Files Included

- `model_training.py` – Main script to train and evaluate models  
- `ames_housing_sorted.csv` – Cleaned dataset used for training  
- `ames_for_tableau.xlsx` – Final dataset with predictions, used in Tableau  
- `housing_dashboard.twbx` – Tableau packaged dashboard file

---

##  Machine Learning Models Used

- Random Forest Regressor
- Gradient Boosting Regressor

Evaluation Metrics:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

---

##  Tableau Dashboard

Explore the interactive dashboard here:  
[Housing Market Insights and Price Prediction – Ames Dataset Visualization](https://public.tableau.com/app/profile/gift.feateide/viz/HousingMarketInsightsandPricePredictionAmesDatasetVisualization/Dashboard1)

This dashboard includes:
- Sale price distribution
- Sale price vs living area
- Average sale price by neighborhood
- Sale price by overall quality
- Actual vs predicted prices
- **Interactive filters** (Neighborhood, Overall Quality, Full Bathroom Count)
- **Drill-down functionality** for focused insights

---

## ⚙️ How to Run the Model Script

### 1. Install required libraries:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn

##  Author

Gift Feateide Adipere 
MSc Information Technology Management  
Berlin, Germany  
