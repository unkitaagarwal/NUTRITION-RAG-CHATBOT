from flask import Flask, request, jsonify
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from firebase_utils import get_user_context, get_user_chat_history, save_user_chat
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

    # Get recent chat history
    chat_history = get_user_chat_history(email)
    formatted_history = "\n".join([f"User: {q}\nBot: {a}" for q, a in chat_history])
    
    print(f"DEBUG: Raw chat history: {chat_history}")
    print(f"DEBUG: Formatted chat history: {formatted_history}")

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
You are a helpful nutrition assistant chatbot. You have access to the user's meal data and complete conversation history.

User Information:
{goal_summary}

COMPLETE CONVERSATION HISTORY:
{formatted_history if formatted_history else "This is the start of our conversation"}

Recent Meal History (All Meals):
{meals_summary}

Categorized Meals:
Snacks: {snacks_summary}
Breakfast: {breakfast_summary}
Lunch: {lunch_summary}
Dinner: {dinner_summary}

CONVERSATION CONTEXT INSTRUCTIONS:
1. ALWAYS maintain conversation context - remember everything the user has asked and your previous responses
2. Use the complete conversation history to provide contextual and personalized responses
3. When the user asks "what was my last question", refer to the question they asked BEFORE their current question (not the current one)
4. Build upon previous conversations - if they ask follow-up questions, reference what you've already discussed
5. Be conversational and remember what you've told them before
6. When asked about specific meal types (breakfast, lunch, dinner, snacks), use only the data from that category
7. The meal data includes detailed nutritional information (calories, carbs, protein, fat) for each meal
8. If the user asks about trends or patterns, analyze their meal history across multiple entries
9. Provide personalized insights based on their eating patterns and previous questions
10. Maintain a helpful, friendly tone throughout the conversation
"""
    
    print(f"DEBUG: System context being sent to AI: {system_context}")
    
    rag_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    response = rag_chain.run(system_context + "\n\nUser question: " + user_question)

    # Save chat interaction for future context
    save_user_chat(email, user_question, response)
    print(f"DEBUG: Saved chat - Question: {user_question}, Response: {response[:100]}...")
    return jsonify({"reply": response})

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5001)