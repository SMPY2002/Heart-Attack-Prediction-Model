from flask import Flask, render_template, request, jsonify, redirect
import numpy as np
import pickle
import time

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))


@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")


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
    time.sleep(5)

    if pred == 0:
        return redirect("/success")
    return redirect("/failure")


@app.route("/success", methods=["GET"])
def success():
    return render_template("success.html")


@app.route("/failure", methods=["GET"])
def failure():
    return render_template("failure.html")


@app.route("/404", methods=["GET"])
def error():
    return render_template("404.html")


@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def catch_all(path):
    return redirect("/404")
