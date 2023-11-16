from flask import Flask, render_template, redirect, url_for, request
import numpy as np
import pickle

# Load the trained model and scalers
model = pickle.load(open('crop_recommendation.pkl', 'rb'))

crop_images = {
    "Rice": "Rice.jpg",
    "Maize": "Maize.jpg",
    "Jute": "Jute.jpg",
    "Cotton": "Cotton.jpg",
    "Coconut": "Coconut.jpg",
    "Papaya": "Papaya.jpg",
    "Orange": "Orange.jpg",
    "Apple": "Apple.jpg",
    "Muskmelon": "Muskmelon.jpg",
    "Watermelon": "Watermelon.jpg",
    "Grapes": "Grapes.jpg",
    "Mango": "Mango.jpg",
    "Banana": "Banana.jpg",
    "Pomegranate": "Pomegranate.jpg",
    "Lentil": "Lentil.jpg",
    "Blackgram": "Blackgram.jpg",
    "Mungbean": "Mungbean.jpg",
    "Mothbeans": "Mothbeans.jpg",
    "Pigeonpeas": "Pigeonpeas.jpg",
    "Kidneybeans": "Kidneybeans.jpg",
    "Chickpea": "Chickpea.jpg",
    "Coffee": "Coffee.jpg"
}

app = Flask(__name__, template_folder="templates")

@app.route('/')
def root():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # Add any login logic here if needed
        # For now, just redirect to the index page
        return redirect(url_for('index'))

    return render_template('login.html')

@app.route('/index')
def index():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    N = float(request.form['Nitrogen'])
    P = float(request.form['Phosphorus'])
    K = float(request.form['Potassium'])
    temp = float(request.form['Temperature'])
    humidity = float(request.form['Humidity'])
    ph = float(request.form['pH'])
    rainfall = float(request.form['Rainfall'])

    feature_list = [N, P, K, temp, humidity, ph, rainfall]

    # Convert the list to a NumPy array and reshape it
    feature_array = np.array(feature_list).reshape(1, -1)

    prediction = model.predict(feature_array)

    crop_dict = {
        1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
        8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
        14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
        19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
    }

    if prediction[0] in crop_dict:
        crop = crop_dict[prediction[0]]
        image_filename = crop_images.get(crop, 'crop.jpg')  # Get the image filename based on the crop
        return render_template('recommendations.html', recommended_crop=crop, image_filename=image_filename)

    else:
        result = "Sorry, we could not determine the best crop to be cultivated with the provided data."
        return render_template('index.html', result=result)

if __name__ == "__main__":
    app.run(debug=True)
