import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load and prepare the Ames Housing data
file_path = r"C:\Users\ADIPERE\Desktop\MLVD practical\MLVD2\ames_housing_sorted.csv"
df = pd.read_csv(file_path)
df = df[df['SalePrice'].notna()]  # drop rows where SalePrice is missing

# 2. Select the features and target
features = ['GrLivArea', 'OverallQual', 'YearBuilt', 'GarageArea', 'Neighborhood']
X = df[features].copy()
y = df['SalePrice']

# 3. Encode the categorical 'Neighborhood' feature
le = LabelEncoder()
X['Neighborhood'] = le.fit_transform(X['Neighborhood'])

# 4. Split into training and test sets (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Initialize the models
rf = RandomForestRegressor(n_estimators=100, random_state=42)
gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)

# 6. Train the models
rf.fit(X_train, y_train)
gb.fit(X_train, y_train)

# 7. Predict on the test set
rf_preds = rf.predict(X_test)
gb_preds = gb.predict(X_test)

# 8. Define an evaluation function
def evaluate_model(name, true, preds):
    mae = mean_absolute_error(true, preds)
    rmse = np.sqrt(mean_squared_error(true, preds))
    r2 = r2_score(true, preds)
    print(f"{name} Performance:")
    print(f"  MAE:   {mae:.2f}")
    print(f"  RMSE:  {rmse:.2f}")
    print(f"  R²:    {r2:.4f}\n")

# 9. Print evaluation results
evaluate_model("Random Forest", y_test, rf_preds)
evaluate_model("Gradient Boosting", y_test, gb_preds)

# 10. Plot feature importances from the Gradient Boosting model
importances = gb.feature_importances_
# sort features by importance
sorted_idx = np.argsort(importances)
sorted_importances = importances[sorted_idx]
sorted_features = np.array(features)[sorted_idx]

plt.figure(figsize=(8, 5))
sns.barplot(x=sorted_importances, y=sorted_features, palette='Blues_r')
plt.title("Feature Importance (Gradient Boosting)")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()
