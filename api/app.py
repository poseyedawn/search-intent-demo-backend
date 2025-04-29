from flask import Flask, request, jsonify
import pickle

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)
with open("model.pkl", "rb") as f:
    clf = pickle.load(f)

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    q = data.get("query","")
    label = clf.predict(vectorizer.transform([q]))[0]
    return jsonify({"query":q,"label":label})

if __name__=="__main__":
    app.run(host="0.0.0.0", port=5000)