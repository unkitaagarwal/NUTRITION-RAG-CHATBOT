from flask import Flask, request, jsonify, send_file
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from firebase_utils import get_user_context, get_user_chat_history, save_user_chat
from dotenv import load_dotenv
import os
import io
from openai import OpenAI
from threading import Thread

# Load environment variables first
load_dotenv()

# Initialize OpenAI client for voice functionality
client = OpenAI()
app = Flask(__name__)

# Initialize once
vector_db = Chroma(persist_directory="./vector_store", embedding_function=OpenAIEmbeddings())
retriever = vector_db.as_retriever(search_kwargs={"k": 5})
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)  # Changed from  gpt-4o-mini to gpt-3.5-turbo

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    email = data["email"]
    user_question = data["question"]

    # Parallelize Firestore calls
    context_result = {}
    history_result = {}

    def fetch_context():
        try:
            context_result["data"] = get_user_context(email)
        except Exception as e:
            print(f"ERROR fetching user context: {e}")
            context_result["data"] = ({}, {'all': []}, {},)

    def fetch_history():
        try:
            history_result["data"] = get_user_chat_history(email)
        except Exception as e:
            print(f"ERROR fetching chat history: {e}")
            history_result["data"] = []

    t1 = Thread(target=fetch_context)
    t2 = Thread(target=fetch_history)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    goal, categorized_meals, user_preferences = context_result.get("data", ({}, {'all': []}, {}))

    # Get recent chat history
    chat_history = history_result.get("data", [])
    formatted_history = "\n".join([f"User: {q}\nBot: {a}" for q, a in chat_history])
    
    print(f"DEBUG: Formatted chat history: {formatted_history}")

    # Build profile details from user preferences
    profile_parts = []
    if user_preferences:
        if user_preferences.get("age"):
            profile_parts.append(f"Age: {user_preferences.get('age')}")
        if user_preferences.get("gender"):
            profile_parts.append(f"Gender: {user_preferences.get('gender')}")
        if user_preferences.get("height"):
            unit = user_preferences.get("height_unit", "cm")
            profile_parts.append(f"Height: {user_preferences.get('height')} {unit}")
        if user_preferences.get("weight"):
            profile_parts.append(f"Weight: {user_preferences.get('weight')} {user_preferences.get('weigh_unit', 'kg')}")
        if user_preferences.get("weight_goal"):
            profile_parts.append(f"Weight Goal: {user_preferences.get('weight_goal')}")
        if user_preferences.get("lifestyle"):
            profile_parts.append(f"Lifestyle: {user_preferences.get('lifestyle')}")
        if user_preferences.get("meal_per_day"):
            profile_parts.append(f"Meals per day: {user_preferences.get('meal_per_day')}")
        if user_preferences.get("meal_type_list"):
            cuisines = ", ".join(user_preferences.get("meal_type_list"))
            profile_parts.append(f"Preferred cuisines: {cuisines}")
        if user_preferences.get("calorie_goal"):
            profile_parts.append(f"Calorie goal: {user_preferences.get('calorie_goal')} kcal")
        if user_preferences.get("protein_goal"):
            profile_parts.append(f"Protein goal: {user_preferences.get('protein_goal')} g")
        if user_preferences.get("carbs_goal"):
            profile_parts.append(f"Carbs goal: {user_preferences.get('carbs_goal')} g")
        if user_preferences.get("fat_goal"):
            profile_parts.append(f"Fat goal: {user_preferences.get('fat_goal')} g")

    if profile_parts:
        goal_summary = "User Profile:\n" + "\n".join(profile_parts)
    elif goal:
        goal_summary = f"User goal: {goal.get('goalType', 'not set')}, Current: {goal.get('currentWeight')}kg, Target: {goal.get('targetWeight')}kg by {goal.get('targetDate')}"
    else:
        goal_summary = "User goal: No specific goals set"
    
    # Debug: Print goal_summary to verify it contains weight_goal
    print(f"DEBUG: Goal summary for {email}: {goal_summary}")
    print(f"DEBUG: User preferences dict: {user_preferences}")
    
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


CRITICAL INSTRUCTIONS:
1. ALWAYS answer questions about the user's goals, weight goal, calorie goals, macro goals, preferences, etc. DIRECTLY from the "User Information" section above. DO NOT say the information is not available if it exists in the User Information section.
2. If the user asks "what is my weight goal?" or "what is my goal?", look for "Weight Goal:" in the User Information section and provide that EXACT information. NEVER say the weight goal is not set if it appears in the User Information section.
3. If the user asks about their calorie goal, protein goal, carbs goal, fat goal, age, lifestyle, preferred cuisines, etc., extract that information directly from the User Information section.
4. FORMATTING REQUIREMENT: When providing meal recommendations, food suggestions, or lists of meals, ALWAYS format them as bullet points using "- " or "* " at the start of each line. Each meal recommendation MUST include:
   - Meal name and brief description
   - Nutritional information (Calories, Protein, Carbs, Fat)
   - Complete recipe with ingredients and step-by-step cooking instructions
   Example format:
   - Meal Name 1: Description (Calories: X, Protein: Yg, Carbs: Zg, Fat: Wg)
     Recipe: Ingredients: [list ingredients]. Instructions: [step-by-step cooking instructions]
   - Meal Name 2: Description (Calories: X, Protein: Yg, Carbs: Zg, Fat: Wg)
     Recipe: Ingredients: [list ingredients]. Instructions: [step-by-step cooking instructions]
   ALWAYS include a complete recipe for every meal you recommend, even if the recipe is not in the retrieved context. Use your knowledge to provide accurate recipes.
5. ALWAYS maintain conversation context - remember everything the user has asked and your previous responses
6. Use the complete conversation history to provide contextual and personalized responses
7. When the user asks "what was my last question", refer to the question they asked BEFORE their current question (not the current one)
8. Build upon previous conversations - if they ask follow-up questions, reference what you've already discussed
9. Be conversational and remember what you've told them before
10. When asked about specific meal types (breakfast, lunch, dinner, snacks), use only the data from that category
11. The meal data includes detailed nutritional information (calories, carbs, protein, fat) for each meal
12. If the user asks about trends or patterns, analyze their meal history across multiple entries
13. Provide personalized insights based on their eating patterns and previous questions
14. Maintain a helpful, friendly tone throughout the conversation.
15. Use the "User Profile" section (age, lifestyle, calorie/macro goals, preferred cuisines, etc.) to tailor every recommendation. Respect their macros, calorie targets, and cuisine preferences when possible.
16. RECIPE REQUIREMENT: ALWAYS provide a complete recipe (ingredients list and step-by-step cooking instructions) for EVERY meal you recommend. Never skip the recipe, even if you need to use your general knowledge. The recipe should be detailed enough for the user to actually cook the meal.
"""


    rag_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    response = rag_chain.run(system_context + "\n\nUser question: " + user_question)

    # Save chat interaction for future context
    Thread(target=save_user_chat, args=(email, user_question, response)).start()
    return jsonify({"reply": response})

@app.route("/speak", methods=["POST"])
def speak():
    text = request.json["text"]
    response = client.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice="alloy",
        input=text
    )
    audio_stream = io.BytesIO(response.read())
    return send_file(audio_stream, mimetype="audio/mpeg")

@app.route("/listen", methods=["POST"])
def listen():
    audio_file = request.files["file"]
    email = request.form.get("email")  # Get email from form data
    
    # Transcribe audio to text
    transcript = client.audio.transcriptions.create(
        model="gpt-4o-mini-transcribe",
        file=(audio_file.filename, audio_file.stream, audio_file.mimetype)
    )
    user_question = transcript.text
    
    if not email:
        return jsonify({"error": "Email is required"}), 400
    
    # Use the same chat logic as the /chat endpoint (parallelized)
    context_result = {}
    history_result = {}

    def fetch_context():
        try:
            context_result["data"] = get_user_context(email)
        except Exception as e:
            print(f"ERROR fetching user context: {e}")
            context_result["data"] = ({}, {'all': []}, {})

    def fetch_history():
        try:
            history_result["data"] = get_user_chat_history(email)
        except Exception as e:
            print(f"ERROR fetching chat history: {e}")
            history_result["data"] = []

    t1 = Thread(target=fetch_context)
    t2 = Thread(target=fetch_history)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    goal, categorized_meals, user_preferences = context_result.get("data", ({}, {'all': []}, {}))

    # Get recent chat history
    chat_history = history_result.get("data", [])
    formatted_history = "\n".join([f"User: {q}\nBot: {a}" for q, a in chat_history])
    
    print(f"DEBUG: Raw chat history: {chat_history}")
    print(f"DEBUG: Formatted chat history: {formatted_history}")

    profile_parts = []
    if user_preferences:
        if user_preferences.get("age"):
            profile_parts.append(f"Age: {user_preferences.get('age')}")
        if user_preferences.get("gender"):
            profile_parts.append(f"Gender: {user_preferences.get('gender')}")
        if user_preferences.get("height"):
            unit = user_preferences.get("height_unit", "cm")
            profile_parts.append(f"Height: {user_preferences.get('height')} {unit}")
        if user_preferences.get("weight"):
            profile_parts.append(f"Weight: {user_preferences.get('weight')} {user_preferences.get('weigh_unit', 'kg')}")
        if user_preferences.get("target_weight"):
            profile_parts.append(f"Target weight: {user_preferences.get('target_weight')} {user_preferences.get('weight_unit', 'kg')}")
        if user_preferences.get("weight_goal"):
            profile_parts.append(f"Weight Goal: {user_preferences.get('weight_goal')}")
        if user_preferences.get("lifestyle"):
            profile_parts.append(f"Lifestyle: {user_preferences.get('lifestyle')}")
        if user_preferences.get("meal_per_day"):
            profile_parts.append(f"Meals per day: {user_preferences.get('meal_per_day')}")
        if user_preferences.get("meal_type_list"):
            cuisines = ", ".join(user_preferences.get("meal_type_list"))
            profile_parts.append(f"Preferred cuisines: {cuisines}")
        if user_preferences.get("calorie_goal"):
            profile_parts.append(f"Calorie goal: {user_preferences.get('calorie_goal')} kcal")
        if user_preferences.get("protein_goal"):
            profile_parts.append(f"Protein goal: {user_preferences.get('protein_goal')} g")
        if user_preferences.get("carbs_goal"):
            profile_parts.append(f"Carbs goal: {user_preferences.get('carbs_goal')} g")
        if user_preferences.get("fat_goal"):
            profile_parts.append(f"Fat goal: {user_preferences.get('fat_goal')} g")


    if profile_parts:
        goal_summary = "User Profile:\n" + "\n".join(profile_parts)
    elif goal:
        goal_summary = f"User goal: {goal.get('goalType', 'not set')}, Current: {goal.get('currentWeight')}kg, Target: {goal.get('targetWeight')}kg by {goal.get('targetDate')}"
    else:
        goal_summary = "User goal: No specific goals set"
    
    # Debug: Print goal_summary to verify it contains weight_goal
    print(f"DEBUG: Goal: {goal}")
    print(f"DEBUG: User preferences dict: {user_preferences}")
    print(f"DEBUG: Goal summary for {email}: {goal_summary}")
    
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


CRITICAL INSTRUCTIONS:
1. ALWAYS answer questions about the user's goals, weight goal, calorie goals, macro goals, preferences, etc. DIRECTLY from the "User Information" section above. DO NOT say the information is not available if it exists in the User Information section.
2. If the user asks "what is my weight goal?" or "what is my goal?", look for "Weight Goal:" in the User Information section and provide that EXACT information. NEVER say the weight goal is not set if it appears in the User Information section.
3. If the user asks about their calorie goal, protein goal, carbs goal, fat goal, age, lifestyle, preferred cuisines, etc., extract that information directly from the User Information section.
4. FORMATTING REQUIREMENT: When providing meal recommendations, food suggestions, or lists of meals, ALWAYS format them as bullet points using "- " or "* " at the start of each line. Each meal recommendation MUST include:
   - Meal name and brief description
   - Nutritional information (Calories, Protein, Carbs, Fat)
   - Complete recipe with ingredients and step-by-step cooking instructions
   Example format:
   - Meal Name 1: Description (Calories: X, Protein: Yg, Carbs: Zg, Fat: Wg)
     Recipe: Ingredients: [list ingredients]. Instructions: [step-by-step cooking instructions]
   - Meal Name 2: Description (Calories: X, Protein: Yg, Carbs: Zg, Fat: Wg)
     Recipe: Ingredients: [list ingredients]. Instructions: [step-by-step cooking instructions]
   ALWAYS include a complete recipe for every meal you recommend, even if the recipe is not in the retrieved context. Use your knowledge to provide accurate recipes.
5. ALWAYS maintain conversation context - remember everything the user has asked and your previous responses
6. Use the complete conversation history to provide contextual and personalized responses
7. When the user asks "what was my last question", refer to the question they asked BEFORE their current question (not the current one)
8. Build upon previous conversations - if they ask follow-up questions, reference what you've already discussed
9. Be conversational and remember what you've told them before
10. When asked about specific meal types (breakfast, lunch, dinner, snacks), use only the data from that category
11. The meal data includes detailed nutritional information (calories, carbs, protein, fat) for each meal
12. If the user asks about trends or patterns, analyze their meal history across multiple entries
13. Provide personalized insights based on their eating patterns and previous questions
14. Maintain a helpful, friendly tone throughout the conversation.
15. Use the "User Profile" section (age, lifestyle, calorie/macro goals, preferred cuisines, etc.) to tailor every recommendation. Respect their macros, calorie targets, and cuisine preferences when possible.
16. RECIPE REQUIREMENT: ALWAYS provide a complete recipe (ingredients list and step-by-step cooking instructions) for EVERY meal you recommend. Never skip the recipe, even if you need to use your general knowledge. The recipe should be detailed enough for the user to actually cook the meal.
"""
    
    print(f"DEBUG: System context being sent to AI: {system_context}")

    rag_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    response = rag_chain.run(system_context + "\n\nUser question: " + user_question)

    # Save chat interaction for future context
    Thread(target=save_user_chat, args=(email, user_question, response)).start()
    
    return jsonify({
        "transcript": user_question,
        "reply": response
    })
   

if __name__ == "__main__":
    # Disable Flask's built-in debugger when running in VS Code debugger
    # Set FLASK_DEBUG environment variable to enable Flask debug mode separately
    import os
    flask_debug = os.environ.get('FLASK_DEBUG', '0') == '1'
    app.run(debug=flask_debug, host='0.0.0.0', port=5002, use_reloader=False)