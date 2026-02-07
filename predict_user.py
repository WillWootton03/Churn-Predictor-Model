import pandas as pd
import numpy as np
import pickle
from keras.models import load_model

predict_entry = pd.DataFrame([{
    "Gender": "Male",
    "Age": 75,
    "Tenure": 1,
    "Usage Frequency": 0,
    "Support Calls": 25,
    "Payment Delay": 15,
    "Subscription Type": "Basic",
    "Contract Length": "Monthly",
    "Total Spend": 20,
    "Last Interaction": 40
}])

# loads our saved model data
model = load_model("churn_model.keras")

# loads the same scalar from the model
with open("scalar.pkl", "rb") as f:
    scaler = pickle.load(f)
# loads the columns we used for our original model to make sure we keep those columns for our predictor
with open("columns.pkl", "rb") as f:
    original_columns = pickle.load(f)

# creates a new pd df for our one new entry
X_new = pd.get_dummies(predict_entry)
# sets columns to be the og columns and makes sure those are np floats
X_new = X_new.reindex(columns=original_columns, fill_value=0).to_numpy(dtype=np.float32)
# scales according to our models scale value
X_new = scaler.transform(X_new)

# gets the probability of churn and the models prediciton
probability = model.predict(X_new)
prediction = (probability > 0.5).astype(int)

print("Probability of churn: ", probability)
print("Will the user churn: ", "Yes" if prediction == 1 else "No")
