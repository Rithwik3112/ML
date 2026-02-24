import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor

# Load data
train = pd.read_csv("train_used_car.csv")
test = pd.read_csv("test_user_car.csv")

# Separate target and features
X = train.drop(columns=["price"])
y = train["price"]

# Identify categorical columns
cat_cols = X.select_dtypes(include="object").columns.tolist()

# Initialize encoder that handles unseen categories
encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

# Fit on training data, transform both
X[cat_cols] = encoder.fit_transform(X[cat_cols])
test[cat_cols] = encoder.transform(test[cat_cols])

# Fill any missing values (important for numeric columns)
X.fillna(-1, inplace=True)
test.fillna(-1, inplace=True)

# Train a Random Forest Regressor
model = RandomForestRegressor(random_state=42, n_estimators=200)
model.fit(X, y)

# Predict on test set
preds = model.predict(test)

# Create submission file
submission = pd.DataFrame({
    "id": test["id"],
    "price": preds
})

submission.to_csv("submission_car.csv", index=False)
print("✅ submission.csv created successfully!")
