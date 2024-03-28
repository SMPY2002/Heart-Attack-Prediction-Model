# Heart-Attack-Prediction-Model
## Introduction
Welcome to the Heart Attack Prediction repository! Here, you'll find an innovative model designed to predict the risk of heart attacks based on basic health parameters. Utilizing advanced machine learning techniques and a user-friendly Flask web application, this project aims to provide valuable insights into cardiovascular health.

## About the Dataset
Our model's predictive prowess is fueled by a comprehensive dataset comprising health records from over 10,000 patients. Sourced from Kaggle, this rich repository of data ensures the robustness and reliability of our predictive analytics.
# Model Building Phase:-

## Model Overview
Let's dive into the heart of the matter! Our prediction model employs the powerful XGBoost algorithm, renowned for its accuracy and versatility in handling complex data patterns. By analyzing key health metrics, such as blood pressure, cholesterol levels, and more, the model delivers reliable predictions regarding potential heart attack risks.
## Main Code snippet for model prediction (by selecting the best hyperparameters using Optuna)
```python
# Making the XGboost model for our heart attack risk prediction [Train with selected features]
def objective(trial):
    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.5),
    }

    model = xgb.XGBClassifier(**params)
    model.fit(X_new_train, y_new_train)
    y_pred = model.predict(X_new_test)
    accuracy = accuracy_score(y_new_test, y_pred)
    return accuracy


# Run Optuna to find the best hyperparameters
study = Optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)

# Get the best hyperparameters
best_params = study.best_params

# Train the model with the best hyperparameters
model = xgb.XGBClassifier(**best_params)
model.fit(X_new_train, y_new_train)
```
## Model Training
Training the model involves harnessing the wealth of information within the dataset to fine-tune our predictive algorithms. Through rigorous experimentation and optimization, our model achieves an impressive accuracy rate of approximately 70% on unseen data.
## Model Evaluation 
To gauge the model's performance, we employ standard evaluation metrics, validating its accuracy and efficacy in real-world scenarios. With a success rate of around 70%, our model proves its mettle in predicting heart attack risks.
## Save the model for its use in the application (with the help of the Python pickle library at the end of the model code)
```python
import pickle
pickle.dump(model, open("model.pkl", "wb"))
```
# Flask Web-App Building Phase:-
Our user-friendly Flask web application serves as a gateway to personalized health insights. By simply inputting basic health parameters, users can receive instant feedback on their heart attack risk status, empowering them to take proactive measures for cardiovascular health.
## Load the model-
```python
# Loading the saved model
model = pickle.load(open("model.pkl", "rb"))
```
## Creating a route for the Main HTML page and rendering it using GET and POST methods 
```python
app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")
```
## Using the GET Method for user input details convert them into a Numpy array and pass it to predict function
```python
@app.route("/predict", methods=["POST"])
def predict():
    data = {
        "age": int(request.form.get("age")),
        "heart_rate": int(request.form.get("heart_rate")),
        "is_diabetic": int(request.form.get("is_diabetic")),
        "family_heart_problem_background": int(
            request.form.get("family_heart_problem_background")
        ),
        "is_smoker": int(request.form.get("smoker")),
        "is_alcohol": int(request.form.get("is_alcohol")),
        "exercise_time": int(request.form.get("exercise")),
        "diet": int(request.form.get("diet")),
    }
    print(data)

    data_array = np.array(
        [
            [
                data["age"],
                data["heart_rate"],
                data["is_diabetic"],
                data["family_heart_problem_background"],
                data["is_smoker"],
                data["is_alcohol"],
                data["exercise_time"],
                data["diet"],
            ]
        ]
    )
    pred = model.predict(data_array)
    if pred == 0:
        return redirect("/success")
    return redirect("/failure")
```
## Render the result of the model output by showing the Success or Failure Page.
```python
@app.route("/success", methods=["GET"])
def success():
    return render_template("success.html")


@app.route("/failure", methods=["GET"])
def failure():
    return render_template("failure.html")
```
## Website Preview 
Landing Page
![Screenshot 2024-03-28 122432](https://github.com/SMPY2002/Heart-Attack-Prediction-Model/assets/118500436/51d83e8c-5ffb-4565-93c4-33607174a77c)

Positive Result 
![Screenshot 2024-03-28 122538](https://github.com/SMPY2002/Heart-Attack-Prediction-Model/assets/118500436/be7ec309-e6f4-40f5-93fd-5fa764663a85)

Negative Result
![Screenshot 2024-03-28 122737](https://github.com/SMPY2002/Heart-Attack-Prediction-Model/assets/118500436/68da59c3-abf4-499a-b020-a1f9518be307)

# Usage
Ready to take the first step towards better heart health? Follow these simple instructions:

1. Clone the repository to your local machine.
2. Install the necessary dependencies using
  ```python
    pip install -r requirements.txt
  ```
3. Run the Flask application using python app.py.
4. Access the web application through your preferred web browser.
5. Input your basic health parameters and await your personalized heart attack risk assessment!
# Contributions
I welcome contributions from all enthusiasts passionate about improving cardiovascular health prediction. Whether it's enhancing model accuracy, optimizing code efficiency, or refining the user interface, your input is invaluable. Feel free to submit pull requests or raise issues for discussion.
# License
This project is licensed under the MIT License, ensuring open collaboration and innovation in the realm of health analytics.
# Acknowledgements
We extend our gratitude to:
* Kaggle for providing the invaluable dataset used in model training.
* XGBoost for its exceptional implementation of tree-based machine learning algorithms.
* Flask for simplifying the development of our user-friendly web application.
Together, we strive to make a positive impact on global health outcomes through the power of predictive analytics. Join us on this journey towards healthier hearts!
