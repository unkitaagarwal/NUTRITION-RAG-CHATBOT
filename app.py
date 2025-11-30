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
retriever = vector_db.as_retriever(search_kwargs={"k": 3})  # Reduced from 5 to 3 for faster retrieval
llm = ChatOpenAI(
    model="gpt-3.5-turbo", 
    temperature=0.3,  # Lower temperature for faster, more deterministic responses
    max_tokens=4000  # Increased to ensure complete recipes for 3-5 meal recommendations with detailed instructions (each meal ~600-800 tokens, so 3-5 meals need ~3000-4000 tokens)
)

# Initialize RAG chain once at startup (not on every request)
# Note: system_context is included in the query string, not as a separate prompt variable

# Create RAG chain without custom prompt (system_context is included in query)
rag_chain = RetrievalQA.from_chain_type(
    llm=llm, 
    retriever=retriever
)

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

    # Get recent chat history (limit to last 2 for performance)
    chat_history = history_result.get("data", [])[-2:]  # Only last 2 messages
    # Truncate long bot responses to keep context manageable
    formatted_history = "\n".join([
        f"User: {q}\nBot: {a[:150] + '...' if len(a) > 150 else a}" 
        for q, a in chat_history
    ])

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
        if user_preferences.get("target_weight"):
            profile_parts.append(f"Target Weight: {user_preferences.get('target_weight')} {user_preferences.get('weigh_unit', 'kg')}")
        if user_preferences.get("weight_goal"):
            profile_parts.append(f"Weight Goal: {user_preferences.get('weight_goal')}")
        if user_preferences.get("lifestyle"):
            profile_parts.append(f"Lifestyle: {user_preferences.get('lifestyle')}")
        if user_preferences.get("meal_per_day"):
            profile_parts.append(f"Meals per day: {user_preferences.get('meal_per_day')}")
        if user_preferences.get("meal_type_list"):
            cuisines = ", ".join(user_preferences.get("meal_type_list"))
            profile_parts.append(f"Preferred cuisines: {cuisines}")
        if user_preferences.get("calorie_goal") is not None:
            profile_parts.append(f"Calorie goal: {user_preferences.get('calorie_goal')} kcal")
        if user_preferences.get("protein_goal") is not None:
            profile_parts.append(f"Protein goal: {user_preferences.get('protein_goal')} g")
        if user_preferences.get("carbs_goal") is not None:
            profile_parts.append(f"Carbs goal: {user_preferences.get('carbs_goal')} g")
        # Check for fat_goal (handle both snake_case and camelCase variants)
        if "fat_goal" in user_preferences:
            profile_parts.append(f"Fat goal: {user_preferences.get('fat_goal')} g")
        elif "fatGoal" in user_preferences:
            profile_parts.append(f"Fat goal: {user_preferences.get('fatGoal')} g")

    if profile_parts:
        goal_summary = "User Profile:\n" + "\n".join(profile_parts)
    elif goal:
        goal_summary = f"User goal: {goal.get('goalType', 'not set')}, Current: {goal.get('currentWeight')}kg, Target: {goal.get('targetWeight')}kg by {goal.get('targetDate')}"
    else:
        goal_summary = "User goal: No specific goals set"
    
    # Debug: Print goal_summary to verify it contains weight_goal
    print(f"DEBUG: Goal summary for {email}: {goal_summary}")
    print(f"DEBUG: User preferences keys: {list(user_preferences.keys()) if user_preferences else 'None'}")
    print(f"DEBUG: Fat goal value (fat_goal): {user_preferences.get('fat_goal')}")
    print(f"DEBUG: Fat goal value (fatGoal): {user_preferences.get('fatGoal')}")
    
    # Get recent meals (limit to last 2 for performance)
    all_meals = categorized_meals.get('all', [])[:2]  # Only last 2 meals
    meals_summary = "\n".join(all_meals) if all_meals else "No recent meals found"
    
    # Don't truncate goal_summary - user profile data is critical and must be complete
    # The performance impact of a few hundred extra characters is negligible compared to LLM processing time
    system_context = f"""You are a nutrition assistant. User: {goal_summary}. Recent: {formatted_history[:200] if formatted_history else 'New conversation'}. Meals: {meals_summary[:200] if meals_summary else 'None'}.

CRITICAL: Meal history (shown above as "Meals:") is COMPLETELY OPTIONAL and used ONLY for understanding patterns. It is NOT required to provide meal recommendations. When the user asks for meal recommendations (dinner ideas, lunch ideas, breakfast ideas, meal plans, lower calorie options, etc.), you MUST ALWAYS provide 3-5 meal recommendations with complete recipes. NEVER refuse by saying you don't have meal history. NEVER say "I don't have the specific meal history" or similar. You MUST use: (1) User Profile information, (2) Your general nutrition knowledge, and (3) The user's specific requirements. If meal history is empty or "None", you MUST STILL provide recommendations. Refusing to provide meals is FORBIDDEN.


CRITICAL INSTRUCTIONS:
1. MEAL RECOMMENDATION REQUIREMENT - HIGHEST PRIORITY: When the user asks for meal recommendations, food suggestions, meal options, dinner ideas, lunch ideas, breakfast ideas, or ANY variation (including "dinner ideas under 400 kcal", "lower calorie options", "show more dinner ideas", "meal plans", etc.), you MUST IMMEDIATELY provide EXACTLY 3-5 meal recommendations with complete recipes. This is THE HIGHEST PRIORITY instruction and is MANDATORY. 
   - NEVER refuse to provide meals
   - NEVER say "I don't have meal history" or "I don't have the specific meal history" or "I don't have any recent meals" or any variation of this refusal
   - NEVER say you cannot provide recommendations due to lack of history
   - Meal history is COMPLETELY OPTIONAL and NOT required - it is only for understanding patterns
   - You MUST ALWAYS generate new meal recommendations using: (1) User Profile information (calorie goals, macro goals, preferences, lifestyle, age, etc.), (2) Your general nutrition knowledge and recipe database, and (3) The specific requirements in the user's question (e.g., "under 400 kcal", "lower calorie", "high protein")
   - If meal history is empty or missing, you MUST STILL provide 3-5 meal recommendations based on user profile and general knowledge
   - If you refuse to provide meals or say you don't have history, you have VIOLATED this instruction
   - Example of CORRECT response: Start immediately with "- Meal Name 1: Description (Calories: X, Protein: Yg, Carbs: Zg, Fat: Wg)" followed by Recipe, Ingredients, and Instructions
   - Example of INCORRECT response: "I don't have the specific meal history for dinner options under 400 kcal" - THIS IS FORBIDDEN
2. ALWAYS answer questions about the user's goals, weight goal, calorie goals, macro goals, preferences, etc. DIRECTLY from the "User Information" section above. DO NOT say the information is not available if it exists in the User Information section.
3. SYNONYM RECOGNITION: Recognize that different phrasings mean the same thing. For example:
   - "weight target", "target weight", "weight goal", "goal weight" all refer to the same thing
   - "calorie goal" and "calorie target" are the same
   - "protein goal" and "protein target" are the same
   - When the user asks about ANY variation of these terms, look for the relevant information in the User Information section using ALL possible field names (Weight Goal, Target Weight, etc.)
4. WEIGHT GOAL/TARGET QUESTIONS: If the user asks about their weight goal, weight target, target weight, goal weight, or any variation, look for BOTH "Weight Goal:" AND "Target Weight:" in the User Information section. Use whichever is available. NEVER say the information is not available if either field exists. If both exist, use the most relevant one or combine them.
5. If the user asks about their calorie goal, protein goal, carbs goal, fat goal, age, lifestyle, preferred cuisines, etc., extract that information directly from the User Information section. Recognize synonyms and variations of these terms as well.
6. FORMATTING REQUIREMENT: When providing meal recommendations, food suggestions, or lists of meals, ALWAYS format them as bullet points using "- " or "* " at the start of each line. Each meal recommendation MUST include:
   - Meal name and brief description
   - Nutritional information (Calories, Protein, Carbs, Fat)
   - Complete recipe with ingredients and DETAILED step-by-step cooking instructions
   Example format:
   - Meal Name 1: Description (Calories: X, Protein: Yg, Carbs: Zg, Fat: Wg)
     Recipe: 
     Ingredients: 
     - Ingredient 1: exact quantity (e.g., "1 cup", "200g", "2 tablespoons")
     - Ingredient 2: exact quantity
     - Ingredient 3: exact quantity
     Instructions:
     1. Detailed step 1 with specific actions, temperatures, and times (e.g., "Heat 1 tablespoon olive oil in a large pan over medium-high heat for 2 minutes")
     2. Detailed step 2 with specific techniques and measurements
     3. Detailed step 3 with cooking times and temperatures
     4. Continue with numbered steps until the meal is complete
   - Meal Name 2: Description (Calories: X, Protein: Yg, Carbs: Zg, Fat: Wg)
     Recipe: [same detailed format]
   - Meal Name 3: Description (Calories: X, Protein: Yg, Carbs: Zg, Fat: Wg)
     Recipe: [same detailed format]
   ALWAYS include a complete recipe for every meal you recommend, even if the recipe is not in the retrieved context. Use your knowledge to provide accurate recipes.
7. ALWAYS maintain conversation context - remember everything the user has asked and your previous responses
8. Use the complete conversation history to provide contextual and personalized responses
9. When the user asks "what was my last question", refer to the question they asked BEFORE their current question (not the current one)
10. Build upon previous conversations - if they ask follow-up questions, reference what you've already discussed
11. Be conversational and remember what you've told them before
12. When asked about specific meal types (breakfast, lunch, dinner, snacks), use only the data from that category when analyzing history or patterns.
13. The meal data includes detailed nutritional information (calories, carbs, protein, fat) for each meal.
14. If the user asks about trends or patterns, analyze their meal history across multiple entries.
15. Provide personalized insights based on their eating patterns and previous questions.
16. Maintain a helpful, friendly tone throughout the conversation.
17. Use the "User Profile" section (age, lifestyle, calorie/macro goals, preferred cuisines, etc.) to tailor every recommendation. Respect their macros, calorie targets, and cuisine preferences when possible.
18. DETAILED RECIPE REQUIREMENT: ALWAYS provide a complete, DETAILED recipe for EVERY meal you recommend. REMEMBER: You must provide 3-5 meals (never just 1), and each meal needs a full recipe. The recipe MUST include:
    - Ingredients list with EXACT quantities (e.g., "1 cup", "200g", "2 tablespoons", "1 medium onion", "3 cloves garlic")
    - Numbered step-by-step instructions that are SPECIFIC and ACTIONABLE:
      * Include exact cooking temperatures (e.g., "375°F", "medium-high heat")
      * Include exact cooking times (e.g., "cook for 5-7 minutes", "bake for 25 minutes")
      * Include specific techniques (e.g., "sauté until golden brown", "whisk until smooth", "simmer uncovered")
      * Include preparation details (e.g., "dice into 1-inch cubes", "chop finely", "slice thinly")
      * Include when to add ingredients (e.g., "add after 2 minutes", "stir in at the end")
      * Include visual/textural cues (e.g., "until tender", "until golden", "until sauce thickens")
    Never skip the recipe or provide vague instructions. The recipe should be detailed enough that someone with basic cooking knowledge can successfully prepare the meal without additional research. You have 4000 tokens available, so use them to provide 3-5 complete meal recommendations with full recipes.
19. NO CROSS-QUESTIONING OR REFUSAL FOR MEAL REQUESTS - ABSOLUTELY FORBIDDEN: If the user asks for meal ideas, meal plans, or specific meal suggestions (for example, "Show dinner ideas under 400 kcal", "lower calorie options", "show more dinner ideas", "dinner ideas", etc.), you MUST directly provide the requested meal recommendations. 
   FORBIDDEN RESPONSES (DO NOT USE THESE):
   - "I don't have the specific meal history" or "I don't have the specific meal history for dinner options under 400 kcal"
   - "I don't have meal history" or "I don't have any recent meals"
   - "I don't have the necessary meal history" or any variation
   - "Would you like me to suggest..." or any follow-up questions
   - Any response that refuses to provide meals
   REQUIRED RESPONSE: You MUST ALWAYS provide 3-5 meal recommendations with complete recipes using: (1) User Profile (calorie/macro goals, preferences, lifestyle), (2) Your general nutrition knowledge, and (3) The user's specific requirements. Meal history is completely optional and NOT required. If meal history is missing or empty, you MUST still provide recommendations based on user profile and general knowledge. Start your response immediately with the first meal recommendation in the required format.
20. FRESH MEAL RECOMMENDATIONS: When the user asks for meal recommendations, DO NOT simply repeat or select meals from their past meal history. Always generate NEW meal ideas and recipes that fit their goals and preferences. You may use history only to understand patterns and preferences, but the recommended meals themselves should be fresh suggestions, not just a recap of what they already ate.
"""

    # Use invoke() instead of run() for better performance
    # Format the query with system context
    query = f"{system_context}\n\nUser question: {user_question}"
    result = rag_chain.invoke({"query": query})
    response = result.get("result", str(result))

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

    # Get recent chat history (limit to last 2 for performance)
    chat_history = history_result.get("data", [])[-2:]  # Only last 2 messages
    # Truncate long bot responses to keep context manageable
    formatted_history = "\n".join([
        f"User: {q}\nBot: {a[:150] + '...' if len(a) > 150 else a}" 
        for q, a in chat_history
    ])
    
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
        if user_preferences.get("calorie_goal") is not None:
            profile_parts.append(f"Calorie goal: {user_preferences.get('calorie_goal')} kcal")
        if user_preferences.get("protein_goal") is not None:
            profile_parts.append(f"Protein goal: {user_preferences.get('protein_goal')} g")
        if user_preferences.get("carbs_goal") is not None:
            profile_parts.append(f"Carbs goal: {user_preferences.get('carbs_goal')} g")
        # Check for fat_goal (handle both snake_case and camelCase variants)
        if "fat_goal" in user_preferences:
            profile_parts.append(f"Fat goal: {user_preferences.get('fat_goal')} g")
        elif "fatGoal" in user_preferences:
            profile_parts.append(f"Fat goal: {user_preferences.get('fatGoal')} g")


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
    print(f"DEBUG: Fat goal value: {user_preferences.get('fat_goal')} or {user_preferences.get('fatGoal')}")
    
    # Get recent meals (limit to last 2 for performance)
    all_meals = categorized_meals.get('all', [])[:2]  # Only last 2 meals
    meals_summary = "\n".join(all_meals) if all_meals else "No recent meals found"
    
    # Don't truncate goal_summary - user profile data is critical and must be complete
    # The performance impact of a few hundred extra characters is negligible compared to LLM processing time
    system_context = f"""You are a nutrition assistant. User: {goal_summary}. Recent: {formatted_history[:200] if formatted_history else 'New conversation'}. Meals: {meals_summary[:200] if meals_summary else 'None'}.

CRITICAL: Meal history (shown above as "Meals:") is COMPLETELY OPTIONAL and used ONLY for understanding patterns. It is NOT required to provide meal recommendations. When the user asks for meal recommendations (dinner ideas, lunch ideas, breakfast ideas, meal plans, lower calorie options, etc.), you MUST ALWAYS provide 3-5 meal recommendations with complete recipes. NEVER refuse by saying you don't have meal history. NEVER say "I don't have the specific meal history" or similar. You MUST use: (1) User Profile information, (2) Your general nutrition knowledge, and (3) The user's specific requirements. If meal history is empty or "None", you MUST STILL provide recommendations. Refusing to provide meals is FORBIDDEN.


CRITICAL INSTRUCTIONS:
1. MEAL RECOMMENDATION REQUIREMENT - HIGHEST PRIORITY: When the user asks for meal recommendations, food suggestions, meal options, dinner ideas, lunch ideas, breakfast ideas, or ANY variation (including "dinner ideas under 400 kcal", "lower calorie options", "show more dinner ideas", "meal plans", etc.), you MUST IMMEDIATELY provide EXACTLY 3-5 meal recommendations with complete recipes. This is THE HIGHEST PRIORITY instruction and is MANDATORY. 
   - NEVER refuse to provide meals
   - NEVER say "I don't have meal history" or "I don't have the specific meal history" or "I don't have any recent meals" or any variation of this refusal
   - NEVER say you cannot provide recommendations due to lack of history
   - Meal history is COMPLETELY OPTIONAL and NOT required - it is only for understanding patterns
   - You MUST ALWAYS generate new meal recommendations using: (1) User Profile information (calorie goals, macro goals, preferences, lifestyle, age, etc.), (2) Your general nutrition knowledge and recipe database, and (3) The specific requirements in the user's question (e.g., "under 400 kcal", "lower calorie", "high protein")
   - If meal history is empty or missing, you MUST STILL provide 3-5 meal recommendations based on user profile and general knowledge
   - If you refuse to provide meals or say you don't have history, you have VIOLATED this instruction
   - Example of CORRECT response: Start immediately with "- Meal Name 1: Description (Calories: X, Protein: Yg, Carbs: Zg, Fat: Wg)" followed by Recipe, Ingredients, and Instructions
   - Example of INCORRECT response: "I don't have the specific meal history for dinner options under 400 kcal" - THIS IS FORBIDDEN
2. ALWAYS answer questions about the user's goals, weight goal, calorie goals, macro goals, preferences, etc. DIRECTLY from the "User Information" section above. DO NOT say the information is not available if it exists in the User Information section.
3. SYNONYM RECOGNITION: Recognize that different phrasings mean the same thing. For example:
   - "weight target", "target weight", "weight goal", "goal weight" all refer to the same thing
   - "calorie goal" and "calorie target" are the same
   - "protein goal" and "protein target" are the same
   - When the user asks about ANY variation of these terms, look for the relevant information in the User Information section using ALL possible field names (Weight Goal, Target Weight, etc.)
4. WEIGHT GOAL/TARGET QUESTIONS: If the user asks about their weight goal, weight target, target weight, goal weight, or any variation, look for BOTH "Weight Goal:" AND "Target Weight:" in the User Information section. Use whichever is available. NEVER say the information is not available if either field exists. If both exist, use the most relevant one or combine them.
5. If the user asks about their calorie goal, protein goal, carbs goal, fat goal, age, lifestyle, preferred cuisines, etc., extract that information directly from the User Information section. Recognize synonyms and variations of these terms as well.
6. FORMATTING REQUIREMENT: When providing meal recommendations, food suggestions, or lists of meals, ALWAYS format them as bullet points using "- " or "* " at the start of each line. Each meal recommendation MUST include:
   - Meal name and brief description
   - Nutritional information (Calories, Protein, Carbs, Fat)
   - Complete recipe with ingredients and DETAILED step-by-step cooking instructions
   Example format:
   - Meal Name 1: Description (Calories: X, Protein: Yg, Carbs: Zg, Fat: Wg)
     Recipe: 
     Ingredients: 
     - Ingredient 1: exact quantity (e.g., "1 cup", "200g", "2 tablespoons")
     - Ingredient 2: exact quantity
     - Ingredient 3: exact quantity
     Instructions:
     1. Detailed step 1 with specific actions, temperatures, and times (e.g., "Heat 1 tablespoon olive oil in a large pan over medium-high heat for 2 minutes")
     2. Detailed step 2 with specific techniques and measurements
     3. Detailed step 3 with cooking times and temperatures
     4. Continue with numbered steps until the meal is complete
   - Meal Name 2: Description (Calories: X, Protein: Yg, Carbs: Zg, Fat: Wg)
     Recipe: [same detailed format]
   - Meal Name 3: Description (Calories: X, Protein: Yg, Carbs: Zg, Fat: Wg)
     Recipe: [same detailed format]
   ALWAYS include a complete recipe for every meal you recommend, even if the recipe is not in the retrieved context. Use your knowledge to provide accurate recipes.
7. ALWAYS maintain conversation context - remember everything the user has asked and your previous responses
8. Use the complete conversation history to provide contextual and personalized responses
9. When the user asks "what was my last question", refer to the question they asked BEFORE their current question (not the current one)
10. Build upon previous conversations - if they ask follow-up questions, reference what you've already discussed
11. Be conversational and remember what you've told them before
12. When asked about specific meal types (breakfast, lunch, dinner, snacks), use only the data from that category when analyzing history or patterns.
13. The meal data includes detailed nutritional information (calories, carbs, protein, fat) for each meal.
14. If the user asks about trends or patterns, analyze their meal history across multiple entries.
15. Provide personalized insights based on their eating patterns and previous questions.
16. Maintain a helpful, friendly tone throughout the conversation.
17. Use the "User Profile" section (age, lifestyle, calorie/macro goals, preferred cuisines, etc.) to tailor every recommendation. Respect their macros, calorie targets, and cuisine preferences when possible.
18. DETAILED RECIPE REQUIREMENT: ALWAYS provide a complete, DETAILED recipe for EVERY meal you recommend. REMEMBER: You must provide 3-5 meals (never just 1), and each meal needs a full recipe. The recipe MUST include:
    - Ingredients list with EXACT quantities (e.g., "1 cup", "200g", "2 tablespoons", "1 medium onion", "3 cloves garlic")
    - Numbered step-by-step instructions that are SPECIFIC and ACTIONABLE:
      * Include exact cooking temperatures (e.g., "375°F", "medium-high heat")
      * Include exact cooking times (e.g., "cook for 5-7 minutes", "bake for 25 minutes")
      * Include specific techniques (e.g., "sauté until golden brown", "whisk until smooth", "simmer uncovered")
      * Include preparation details (e.g., "dice into 1-inch cubes", "chop finely", "slice thinly")
      * Include when to add ingredients (e.g., "add after 2 minutes", "stir in at the end")
      * Include visual/textural cues (e.g., "until tender", "until golden", "until sauce thickens")
    Never skip the recipe or provide vague instructions. The recipe should be detailed enough that someone with basic cooking knowledge can successfully prepare the meal without additional research. You have 4000 tokens available, so use them to provide 3-5 complete meal recommendations with full recipes.
19. NO CROSS-QUESTIONING OR REFUSAL FOR MEAL REQUESTS - ABSOLUTELY FORBIDDEN: If the user asks for meal ideas, meal plans, or specific meal suggestions (for example, "Show dinner ideas under 400 kcal", "lower calorie options", "show more dinner ideas", "dinner ideas", etc.), you MUST directly provide the requested meal recommendations. 
   FORBIDDEN RESPONSES (DO NOT USE THESE):
   - "I don't have the specific meal history" or "I don't have the specific meal history for dinner options under 400 kcal"
   - "I don't have meal history" or "I don't have any recent meals"
   - "I don't have the necessary meal history" or any variation
   - "Would you like me to suggest..." or any follow-up questions
   - Any response that refuses to provide meals
   REQUIRED RESPONSE: You MUST ALWAYS provide 3-5 meal recommendations with complete recipes using: (1) User Profile (calorie/macro goals, preferences, lifestyle), (2) Your general nutrition knowledge, and (3) The user's specific requirements. Meal history is completely optional and NOT required. If meal history is missing or empty, you MUST still provide recommendations based on user profile and general knowledge. Start your response immediately with the first meal recommendation in the required format.
20. FRESH MEAL RECOMMENDATIONS: When the user asks for meal recommendations, DO NOT simply repeat or select meals from their past meal history. Always generate NEW meal ideas and recipes that fit their goals and preferences. You may use history only to understand patterns and preferences, but the recommended meals themselves should be fresh suggestions, not just a recap of what they already ate.
"""
    
    print(f"DEBUG: System context being sent to AI: {system_context}")

    # Use invoke() instead of run() for better performance
    # Format the query with system context
    query = f"{system_context}\n\nUser question: {user_question}"
    result = rag_chain.invoke({"query": query})
    response = result.get("result", str(result))

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