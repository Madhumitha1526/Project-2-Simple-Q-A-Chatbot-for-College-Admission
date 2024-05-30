import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Sample data (You can expand this with more questions and answers)
data = {
    "question": [
        "When is the application deadline?",
        "What are the application requirements?",
        "Can I apply for financial aid?",
        "What are the tuition fees?",
        "How can I schedule a campus visit?",
        "Are there research opportunities?",
        "Tell me about the labs",
        "What seminars and events are available?",
        "How are the professors?",
        "What food options are available in the canteen?"
    ],
    "category": [
        "application deadline",
        "application requirements",
        "financial aid",
        "tuition fees",
        "campus visit",
        "research opportunities",
        "labs",
        "seminars and events",
        "professors",
        "canteen"
    ]
}

df = pd.DataFrame(data)

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(df['question'], df['category'], test_size=0.2, random_state=42)

# Creating a pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression())
])

# Training the model
pipeline.fit(X_train, y_train)

# Evaluating the model
y_pred = pipeline.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

# Saving the model
joblib.dump(pipeline, 'chatbot_model.pkl')
