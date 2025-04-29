import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

# 1) Read the human-labeled sample (must be in the same folder)
df = pd.read_csv("sample_queries_labeled.csv")
X = df["query_clean"]
y = df["predicted_label"]

# 2) Fit vectorizer & model
vectorizer = TfidfVectorizer(ngram_range=(1,3), min_df=2)
X_vec = vectorizer.fit_transform(X)
clf    = LogisticRegression(class_weight="balanced", max_iter=1000)
clf.fit(X_vec, y)

# 3) Write both objects right here in the root
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
with open("model.pkl", "wb") as f:
    pickle.dump(clf, f)

print("Saved vectorizer.pkl and model.pkl")