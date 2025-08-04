from flask import Flask, render_template, request
import pickle
import numpy as np

# Load model and label encoders
model = pickle.load(open("model/model.pkl", "rb"))
label_encoders = pickle.load(open("model/label_encoders.pkl", "rb"))

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Get user input
    artist = request.form["artist"]
    genre = request.form["genre"]
    popularity = float(request.form["popularity"])
    capacity = int(request.form["capacity"])
    city = request.form["city"]
    month = int(request.form["month"])
    demand = float(request.form["demand"])
    ticket_type = request.form["ticket_type"]

    # Encode categorical inputs
    artist_encoded = label_encoders["Artist"].transform([artist])[0]
    genre_encoded = label_encoders["Genre"].transform([genre])[0]
    city_encoded = label_encoders["City"].transform([city])[0]
    ticket_type_encoded = label_encoders["Ticket_Type"].transform([ticket_type])[0]

    # Prepare input for prediction
    features = np.array([[artist_encoded, genre_encoded, popularity, capacity,
                          city_encoded, month, demand, ticket_type_encoded]])
    
    # Predict
    prediction = model.predict(features)[0]

    return render_template("result.html", price=round(prediction, 2))

if __name__ == "__main__":
    app.run(debug=True)
