from flask import Flask, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load course dataset
courses = pd.read_csv("courses.csv")

# Create TF-IDF matrix for course descriptions
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(courses["description"])
similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

@app.route("/")
def home():
    return {"message": "Edu Analytics ML API is running!"}

@app.route("/recommendations", methods=["GET"])
def recommendations():
    try:
        course_id = int(request.args.get("course_id", 1))
        idx = courses[courses["id"] == course_id].index[0]

        # Get similarity scores
        sim_scores = list(enumerate(similarity_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:3]  # top 2 similar courses

        recs = [{
            "id": int(courses.iloc[i[0]]["id"]),
            "title": courses.iloc[i[0]]["title"],
            "description": courses.iloc[i[0]]["description"]
        } for i in sim_scores]

        return jsonify({"course_id": course_id, "recommendations": recs})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
