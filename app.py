from flask import Flask, request, jsonify, render_template, redirect, url_for
from flask_cors import CORS
import joblib
import re

app = Flask(__name__)
CORS(app)

# Load the trained model
model = joblib.load('chatbot_model.pkl')

knowledge_base = {
    "application deadline": "The application deadline is December 1st for regular decision.",
    "application requirements": "You'll need to submit your transcripts, test scores, essays, and letters of recommendation.",
    "admission requirements": "The requirements vary depending on your program of interest, but generally include transcripts, test scores, essays, and letters of recommendation. You can find more specific details on the college website.",
    "scholarships": "To apply for scholarships, you need to fill out the scholarship application form and submit it along with your admission application.",
    "financial aid": "Yes, you can apply for financial aid by filling out the FAFSA form.",
    "tuition fees": "The tuition fees depend on the program and residency status. Please check the college website for detailed information.",
    "campus visit": "You can schedule a campus visit by contacting the admissions office. Campus tours are available on weekdays.",
    "research opportunities": "Several departments offer research opportunities for students. You can talk to your professors or department head to learn more about available options.",
    "labs": "The labs are modern and well-equipped with the latest technology. You'll have access to all the resources you need for your studies and research.",
    "seminars and events": "Yes, the college regularly hosts lectures, workshops, and seminars by renowned scholars and professionals.",
    "professors": "The faculty at our college are highly qualified and passionate about their subjects. They are generally approachable and supportive of their students.",
    "canteen": "The canteen offers a variety of options, including vegetarian and healthy choices. There are also several cafes and restaurants on campus if you're looking for something different."
}

conversation_history = []
user_info = {}

@app.route('/')
def home():
    return render_template('name.html')

@app.route('/chat')
def chat():
    if "name" not in user_info:
        return redirect(url_for('home'))
    return render_template('chat.html')

@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    question = data.get("question")
    response = handle_question(question)
    conversation_history.append((question, response))
    return jsonify({"response": response})

def handle_question(question):
    category = model.predict([question])[0]
    response = knowledge_base.get(category, "I'm not sure I understand that question. Could you rephrase it or try asking something different?")
    try:
        response = personalize_response(response, conversation_history, user_info)
    except KeyError:
        pass
    return response

def personalize_response(response, conversation_history, user_info):
    if "name" in user_info:
        response = response.replace("you", user_info["name"])
    return response

@app.route('/set_name', methods=['POST'])
def set_name():
    data = request.json
    name = data.get("name")
    user_info["name"] = name
    return jsonify({"message": f"Nice to meet you, {name}!", "redirect": url_for('chat')})

if __name__ == "__main__":
    app.run(debug=True)
