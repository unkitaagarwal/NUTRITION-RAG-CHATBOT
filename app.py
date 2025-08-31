from flask import Flask, request, jsonify
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from firebase_utils import get_user_context
from dotenv import load_dotenv
import os

load_dotenv()
app = Flask(__name__)

# Initialize once
vector_db = Chroma(persist_directory="./vector_store", embedding_function=OpenAIEmbeddings())
retriever = vector_db.as_retriever(search_kwargs={"k": 3})
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)  # Changed from gpt-5 to gpt-4o-mini

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    email = data["email"]
    user_question = data["question"]

    goal, categorized_meals = get_user_context(email)

    # Handle empty goal data better
    if goal:
        goal_summary = f"User goal: {goal.get('goalType', 'not set')}, Current: {goal.get('currentWeight')}kg, Target: {goal.get('targetWeight')}kg by {goal.get('targetDate')}"
    else:
        goal_summary = "User goal: No specific goals set"
    
    # Get all meals for general context
    all_meals = categorized_meals.get('all', [])
    meals_summary = "\n".join(all_meals) if all_meals else "No recent meals found"
    
    # Add categorized meal summaries for more accurate responses
    snacks_summary = "\n".join(categorized_meals.get('snacks', [])) if categorized_meals.get('snacks') else "No snacks recorded"
    breakfast_summary = "\n".join(categorized_meals.get('breakfast', [])) if categorized_meals.get('breakfast') else "No breakfast recorded"
    lunch_summary = "\n".join(categorized_meals.get('lunch', [])) if categorized_meals.get('lunch') else "No lunch recorded"
    dinner_summary = "\n".join(categorized_meals.get('dinner', [])) if categorized_meals.get('dinner') else "No dinner recorded"

    system_context = f"""
User Information:
{goal_summary}

Recent Meal History (All Meals):
{meals_summary}

Categorized Meals:
Snacks: {snacks_summary}
Breakfast: {breakfast_summary}
Lunch: {lunch_summary}
Dinner: {dinner_summary}

Please answer the user's question based on the meal data above. The meal data includes detailed nutritional information (calories, carbs, protein, fat) for each meal. When asked about specific meal types (like snacks, breakfast, etc.), use only the data from that specific category.
"""
    
    print(f"DEBUG: System context being sent to AI: {system_context}")
    
    rag_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    response = rag_chain.run(system_context + "\n\nUser question: " + user_question)
    return jsonify({"reply": response})

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
