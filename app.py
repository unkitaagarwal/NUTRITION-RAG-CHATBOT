from flask import Flask, request, jsonify, send_file
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from firebase_utils import (
    get_user_context,
    get_user_chat_history,
    save_user_chat,
    init_firestore,
    try_persist_meal_image_from_openai_url,
    upload_meal_image_bytes_to_storage,
    save_recommend_meal_image_record,
)
from dotenv import load_dotenv
import os
import csv
import io
import json
import re
import base64
import tempfile
from datetime import datetime, timedelta
import uuid
from openai import OpenAI
from threading import Thread, Lock
from collections import defaultdict
import yt_dlp
import shutil
import time
import socket
import ipaddress
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed


# Load environment variables first
load_dotenv()

# Initialize OpenAI client for voice functionality
client = OpenAI()
app = Flask(__name__)

# In-memory de-dup memory for /recommend-meals/day calls per plan_id.
# This keeps day plans distinct for sequential calls using the same plan_id.
_plan_meal_name_history = defaultdict(set)
_plan_meal_history_lock = Lock()

# Initialize once
# ---------- Config ----------
MAX_VIDEO_SECONDS = int(os.getenv("MAX_VIDEO_SECONDS", "900"))  # 15 min default
# Cookies can be provided as: 1) file path, or 2) base64-encoded content in YTDLP_COOKIES_B64 env var
YTDLP_COOKIES_FILE = os.path.expanduser(os.path.expandvars(os.getenv("YTDLP_COOKIES_FILE", ""))) or None  # optional, helps IG/TikTok
YTDLP_COOKIES_B64 = os.getenv("YTDLP_COOKIES_B64")  # alternative: base64-encoded cookies content (for Render/cloud)
LLM_MODEL = os.getenv("RECIPE_LLM_MODEL", "gpt-4o-mini")  # change if needed
RECIPE_LLM_MODEL = os.getenv("RECIPE_LLM_MODEL", "gpt-4o-mini")

YT_PROXY = os.getenv("YT_PROXY")

# Note: Global ydl_opts is not used - proxy is conditionally applied in individual functions
# Proxy is ONLY used for YouTube URLs, not for TikTok/Instagram/webpages
# See ytdlp_base_opts(), _yt_meta(), get_video_metadata(), and _download_audio_mp3() functions

vector_db = Chroma(persist_directory="./vector_store", embedding_function=OpenAIEmbeddings())
retriever = vector_db.as_retriever(search_kwargs={"k": 3})  # Reduced from 5 to 3 for faster retrieval
llm = ChatOpenAI(
    model="gpt-3.5-turbo", 
    temperature=0.3,  # Lower temperature for faster, more deterministic responses
    max_tokens=4000  # Increased to ensure complete recipes for 3-5 meal recommendations with detailed instructions (each meal ~600-800 tokens, so 3-5 meals need ~3000-4000 tokens)
)

# Note: OpenAI client is initialized once above and reused for all endpoints
# The client.chat.completions.create() calls are just API requests, not re-initializations

# User-friendly message for any 500 (client can show this when status is 500)
DEFAULT_500_USER_MESSAGE = "We ran into a problem. Please try again in a moment."


@app.errorhandler(500)
def handle_500(err):
    """Ensure unhandled exceptions return a consistent JSON body with user_message."""
    return jsonify({
        "error": "Internal Server Error",
        "user_message": DEFAULT_500_USER_MESSAGE,
        "details": str(err) if app.debug else None,
    }), 500


@app.after_request
def ensure_500_user_message(response):
    """Ensure every 500 response includes user_message so clients can show a message."""
    if response.status_code != 500:
        return response
    if not response.is_json:
        return response
    try:
        data = response.get_json()
        if data is None:
            return response
        if "user_message" not in data or not data["user_message"]:
            data = dict(data) if data else {}
            data["user_message"] = data.get("user_message") or DEFAULT_500_USER_MESSAGE
            response.set_data(json.dumps(data))
    except Exception:
        pass
    return response


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


@app.route("/detect-ingredients", methods=["POST"])
def detect_ingredients():
    try:
        image_url_for_vision = None
        
        # Priority: file upload > base64 JSON
        # Check for file upload (multipart/form-data)
        if 'image' in request.files:
            uploaded_file = request.files['image']
            if uploaded_file.filename:
                print("🔍 Received image file upload:", uploaded_file.filename)
                
                # Read image bytes
                image_bytes = uploaded_file.read()
                
                # Determine content type from file extension
                filename = uploaded_file.filename.lower()
                if filename.endswith(('.jpg', '.jpeg')):
                    content_type = "image/jpeg"
                elif filename.endswith('.png'):
                    content_type = "image/png"
                elif filename.endswith('.gif'):
                    content_type = "image/gif"
                elif filename.endswith('.webp'):
                    content_type = "image/webp"
                else:
                    content_type = "image/jpeg"  # Default
                
                # Convert to base64
                image_base64 = base64.b64encode(image_bytes).decode("utf-8")
                image_url_for_vision = f"data:{content_type};base64,{image_base64}"
                print("✅ Image file converted to base64")
        
        # Check for base64 in JSON (application/json)
        elif request.is_json:
            data = request.get_json()
            image_base64 = data.get("imageBase64")
            image_format = data.get("imageFormat", "jpg")
            
            if image_base64:
                print("🔍 Received base64 image")
                # Remove data URL prefix if present (e.g., "data:image/jpeg;base64,")
                if "," in image_base64:
                    image_base64 = image_base64.split(",")[-1]
                
                # Determine content type from format
                format_to_mime = {
                    "jpg": "image/jpeg",
                    "jpeg": "image/jpeg",
                    "png": "image/png",
                    "gif": "image/gif",
                    "webp": "image/webp"
                }
                content_type = format_to_mime.get(image_format.lower(), "image/jpeg")
                
                # Create data URL for GPT-Vision
                image_url_for_vision = f"data:{content_type};base64,{image_base64}"
                print("✅ Using base64 image directly")
        
        # If no image was provided through any method
        if not image_url_for_vision:
            return jsonify({
                "error": "Image is required",
                "details": "Provide image in one of these formats: 1) File upload (multipart/form-data with 'image' field), or 2) Base64 JSON (imageBase64 + imageFormat)"
            }), 400

        print("🤖 Calling GPT Vision API...")

        # 2. Call GPT-Vision API using pre-initialized OpenAI client
        # Note: 'client' is initialized once at startup (line 20), so this is just an API call, not a re-initialization
        try:
            vision_response = client.chat.completions.create(
                model="gpt-4o-mini",  # Cost-effective: ~10x cheaper than gpt-4o, still supports vision
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "List visible food ingredients. Comma-separated only, no explanations.",
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": image_url_for_vision
                                },
                            },
                        ],
                    }
                ],
                max_tokens=200,  # Reduced from 300 - ingredients list doesn't need that many tokens
                temperature=0.2  # Lower temperature for more deterministic, cheaper responses
            )

            # 3. Extract content
            content = vision_response.choices[0].message.content
            print("📝 Raw AI content:", content)

            # 4. Parse comma-separated ingredients
            ingredients = [
                item.strip().lower()
                for item in content.replace("\n", ",").split(",")
                if item.strip() and ":" not in item.lower() and "sorry" not in item.lower()
            ][:15]

            print("🥬 Final ingredients:", ingredients)

            if not ingredients:
                return jsonify({
                    "ingredients": [],
                    "message": "No ingredients detected in this image",
                })

            return jsonify({
                "ingredients": ingredients,
                "message": f"Found {len(ingredients)} ingredients",
            })

        except Exception as vision_error:
            print("💥 GPT Vision error:", str(vision_error))
            return jsonify({
                "error": "Failed to analyze image with GPT Vision",
                "user_message": "We couldn't read ingredients from this image. Please try another photo or type them in.",
                "details": str(vision_error),
            }), 500

    except Exception as e:
        print("💥 Critical error:", str(e))
        return jsonify({
            "error": "Something went wrong analyzing the image",
            "user_message": "We couldn't analyze this image. Please try again or type your ingredients instead.",
            "details": str(e),
        }), 500


@app.route("/generate-meals", methods=["POST"])
def generate_meals():
    data = request.get_json()
    
    # Extract parameters
    ingredients = data.get("ingredients", [])
    cuisine = data.get("cuisine", "")
    cooking_time = data.get("cookingTime", "")
    diet = data.get("diet", "")
    macro_targets = data.get("macroTargets", {})
    meal_count = data.get("mealCount", 3)
    include_images = data.get("includeImages", True)
    
    # Validate ingredients list is not empty
    if not ingredients or len(ingredients) == 0:
        return jsonify({"error": "Ingredients list cannot be empty"}), 400
    
    # Compute macro-per-meal values if macroTargets exist
    calories_per_meal = None
    protein_per_meal = None
    carbs_per_meal = None
    fats_per_meal = None
    
    if macro_targets:
        total_calories = macro_targets.get("calories", 0)
        total_protein = macro_targets.get("protein", 0)
        total_carbs = macro_targets.get("carbs", 0)
        total_fats = macro_targets.get("fats", 0)
        
        if meal_count > 0:
            calories_per_meal = total_calories / meal_count if total_calories > 0 else None
            protein_per_meal = total_protein / meal_count if total_protein > 0 else None
            carbs_per_meal = total_carbs / meal_count if total_carbs > 0 else None
            fats_per_meal = total_fats / meal_count if total_fats > 0 else None
    
    # Build the GPT prompt
    prompt_parts = []
    prompt_parts.append(f"Generate exactly {meal_count} meal recommendations.")
    prompt_parts.append(f"\nRequired ingredients: {', '.join(ingredients)}")
    
    if cuisine:
        prompt_parts.append(f"Cuisine preference: {cuisine}")
    if cooking_time:
        prompt_parts.append(f"Cooking time preference: {cooking_time}")
    if diet:
        prompt_parts.append(f"Diet preference: {diet}")
    
    if macro_targets and (calories_per_meal or protein_per_meal or carbs_per_meal or fats_per_meal):
        prompt_parts.append("\nMacro target requirements per meal:")
        if calories_per_meal:
            prompt_parts.append(f"- Calories: approximately {calories_per_meal:.0f} kcal")
        if protein_per_meal:
            prompt_parts.append(f"- Protein: approximately {protein_per_meal:.1f}g")
        if carbs_per_meal:
            prompt_parts.append(f"- Carbs: approximately {carbs_per_meal:.1f}g")
        if fats_per_meal:
            prompt_parts.append(f"- Fats: approximately {fats_per_meal:.1f}g")
    
    # JSON schema for meal object (optimized - more concise)
    macro_summary = ""
    if macro_targets:
        totals = []
        if macro_targets.get('calories'):
            totals.append(f"{macro_targets.get('calories', 0):.0f} kcal")
        if macro_targets.get('protein'):
            totals.append(f"{macro_targets.get('protein', 0):.1f}g protein")
        if macro_targets.get('carbs'):
            totals.append(f"{macro_targets.get('carbs', 0):.1f}g carbs")
        if macro_targets.get('fats'):
            totals.append(f"{macro_targets.get('fats', 0):.1f}g fats")
        if totals:
            macro_summary = f" Total targets: {', '.join(totals)}."
    
    prompt_parts.append(f"""
JSON array with {meal_count} meals. Schema: {{"name":"string","description":"string","calories":number,"protein":number,"carbs":number,"fats":number,"ingredients":["string"],"instructions":["step"],"cookingTime":"string","servings":number}}
Rules: Use required ingredients. Step-by-step instructions.{macro_summary} Valid JSON only, no markdown, no trailing commas.
""")
    
    user_prompt = "\n".join(prompt_parts)
    
    # System message (cost-optimized - minimal tokens)
    system_message = """Nutrition expert. Return JSON array only. No markdown. No trailing commas. Accurate nutrition."""
    
    # Calculate optimal max_tokens based on meal count (cost-optimized)
    # Reduced estimate: each meal ~500-600 tokens (optimized prompts)
    estimated_tokens = meal_count * 550 + 150
    max_tokens = min(max(estimated_tokens, 1200), 3500)  # Reduced range: 1200-3500 tokens
    
    # Call OpenAI Chat Completion API (optimized for speed)
    response_text = None
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,  # Lower temperature for faster, more deterministic responses
            max_tokens=max_tokens
        )
        
        response_text = response.choices[0].message.content.strip()
        
        # Remove markdown code blocks if present
        response_text = re.sub(r'```json\s*', '', response_text)
        response_text = re.sub(r'```\s*', '', response_text)
        response_text = response_text.strip()
        
        # Clean trailing commas from JSON (common GPT issue)
        # Remove trailing commas before closing brackets/braces (handle nested structures)
        # This regex handles whitespace and newlines before closing brackets/braces
        response_text = re.sub(r',(\s*})', r'\1', response_text)  # Remove trailing comma before }
        response_text = re.sub(r',(\s*])', r'\1', response_text)  # Remove trailing comma before ]
        # Also handle cases with newlines
        response_text = re.sub(r',\s*\n\s*}', '\n}', response_text)
        response_text = re.sub(r',\s*\n\s*]', '\n]', response_text)
        
        # Parse JSON with retry logic for trailing commas
        meals = None
        parse_attempts = 0
        while parse_attempts < 3:
            try:
                meals = json.loads(response_text)
                break
            except json.JSONDecodeError as parse_error:
                parse_attempts += 1
                if parse_attempts >= 3:
                    raise  # Re-raise if all attempts failed
                # Try more aggressive cleaning
                # Remove trailing commas more aggressively line by line
                lines = response_text.split('\n')
                cleaned_lines = []
                for i, line in enumerate(lines):
                    # Remove trailing comma if next non-empty line starts with } or ]
                    if i < len(lines) - 1:
                        next_line = lines[i + 1].strip()
                        if next_line in ['}', ']'] and line.rstrip().endswith(','):
                            cleaned_lines.append(line.rstrip().rstrip(','))
                        else:
                            cleaned_lines.append(line)
                    else:
                        cleaned_lines.append(line)
                response_text = '\n'.join(cleaned_lines)
                # Try one more cleanup pass
                response_text = re.sub(r',(\s*})', r'\1', response_text)
                response_text = re.sub(r',(\s*])', r'\1', response_text)
        
        # Validate and fill missing fields with defaults
        default_values = {
            "cookingTime": "30 minutes",
            "servings": 1
        }
        
        validated_meals = []
        for meal in meals:
            # Ensure all required fields exist
            validated_meal = {
                "name": meal.get("name", "Unnamed Meal"),
                "description": meal.get("description", ""),
                "calories": meal.get("calories", 0),
                "protein": meal.get("protein", 0),
                "carbs": meal.get("carbs", 0),
                "fats": meal.get("fats", 0),
                "ingredients": meal.get("ingredients", []),
                "instructions": meal.get("instructions", []),
                "cookingTime": meal.get("cookingTime", default_values["cookingTime"]),
                "servings": meal.get("servings", default_values["servings"]),
                "imageUrl": None  # Will be populated if include_images is True
            }
            validated_meals.append(validated_meal)
        
        # Generate images for meals if requested (b64_json → Firebase Storage, same as /food-logging)
        if include_images:
            default_image_url = os.getenv(
                "DEFAULT_MEAL_IMAGE_URL",
                "https://images.unsplash.com/photo-1498837167922-ddd27525d352?auto=format&fit=crop&w=1200&q=80",
            )
            dalle_sz = _food_logging_dalle_size()
            workers = _food_logging_image_pool_size(len(validated_meals))

            def _generate_and_persist(m: dict) -> str:
                b64 = _generate_food_log_meal_image_b64(m, size=dalle_sz)
                return _food_logging_finalize_image_from_b64(b64, default_image_url)

            with ThreadPoolExecutor(max_workers=workers) as ex:
                finalized = list(ex.map(_generate_and_persist, validated_meals))
            for meal, url in zip(validated_meals, finalized):
                meal["imageUrl"] = url
        
        # Compute totals
        totals = {
            "calories": sum(meal["calories"] for meal in validated_meals),
            "protein": sum(meal["protein"] for meal in validated_meals),
            "carbs": sum(meal["carbs"] for meal in validated_meals),
            "fats": sum(meal["fats"] for meal in validated_meals)
        }
        
        return jsonify({
            "meals": validated_meals,
            "totals": totals,
            "mealCount": len(validated_meals)
        })
        
    except json.JSONDecodeError as e:
        error_msg = f"Failed to parse JSON response: {str(e)}"
        if response_text:
            error_msg += f"\nRaw response: {response_text[:500]}"
        return jsonify({
            "error": error_msg,
            "user_message": "We ran into a problem generating your meals. Please try again in a moment.",
        }), 500
    except Exception as e:
        # Handle OpenAI API errors specifically
        error_str = str(e)
        error_dict = {}
        
        # Try to extract OpenAI error details from the exception
        # OpenAI errors often contain nested error information
        try:
            # Check if error has response attribute (OpenAI SDK errors)
            if hasattr(e, 'response') and hasattr(e.response, 'json'):
                error_data = e.response.json()
                if 'error' in error_data:
                    openai_error = error_data['error']
                    error_code = openai_error.get('code', '')
                    error_type = openai_error.get('type', '')
                    error_message = openai_error.get('message', str(e))
                    
                    if error_code == 'insufficient_quota' or error_type == 'insufficient_quota' or '429' in error_str:
                        error_dict = {
                            "error": "OpenAI API quota exceeded",
                            "message": error_message,
                            "type": "quota_exceeded",
                            "status_code": 429,
                            "details": "Please check your OpenAI plan and billing details at https://platform.openai.com/account/billing"
                        }
                        return jsonify(error_dict), 429
                    elif error_code == 'invalid_api_key' or '401' in error_str:
                        error_dict = {
                            "error": "OpenAI API authentication failed",
                            "message": error_message,
                            "type": "authentication_error",
                            "status_code": 401
                        }
                        return jsonify(error_dict), 401
                    elif 'rate_limit' in error_code.lower() or 'rate_limit' in error_type.lower():
                        error_dict = {
                            "error": "OpenAI API rate limit exceeded",
                            "message": error_message,
                            "type": "rate_limit_exceeded",
                            "status_code": 429
                        }
                        return jsonify(error_dict), 429
        except:
            pass  # Fall through to string-based detection
        
        # String-based error detection (fallback)
        if "429" in error_str or "quota" in error_str.lower() or "insufficient_quota" in error_str.lower():
            error_dict = {
                "error": "OpenAI API quota exceeded",
                "message": "You have exceeded your OpenAI API quota. Please check your plan and billing details.",
                "type": "quota_exceeded",
                "status_code": 429,
                "details": "For more information, visit: https://platform.openai.com/docs/guides/error-codes/api-errors"
            }
            return jsonify(error_dict), 429
        elif "401" in error_str or "unauthorized" in error_str.lower():
            error_dict = {
                "error": "OpenAI API authentication failed",
                "message": "Invalid API key or authentication error",
                "type": "authentication_error",
                "status_code": 401
            }
            return jsonify(error_dict), 401
        elif "rate_limit" in error_str.lower() or "rate limit" in error_str.lower():
            error_dict = {
                "error": "OpenAI API rate limit exceeded",
                "message": "Too many requests. Please try again later.",
                "type": "rate_limit_exceeded",
                "status_code": 429
            }
            return jsonify(error_dict), 429
        else:
            # Generic error
            error_dict = {
                "error": "Error generating meals",
                "user_message": "We ran into a problem generating your meals. Please try again in a moment.",
                "message": str(e),
                "type": "unknown_error",
                "status_code": 500
            }
            print(f"Error generating meals: {error_str}")
            return jsonify(error_dict), 500

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


def _get_meal_keys(num_meals: int) -> list[str]:
    """
    Return canonical meal keys for dynamic meal count.
    First four keys are breakfast/lunch/dinner/snacks, extras are snacks_2, snacks_3, ...
    """
    base = ["breakfast", "lunch", "dinner", "snacks"]
    if num_meals <= 4:
        return base[:num_meals]
    extra = [f"snacks_{i}" for i in range(2, num_meals - 4 + 2)]
    return base + extra


def _meal_type_from_key(meal_key: str) -> str:
    if meal_key == "breakfast":
        return "Breakfast"
    if meal_key == "lunch":
        return "Lunch"
    if meal_key == "dinner":
        return "Dinner"
    return "Snack"


def _dietary_restrictions_plant_based_prompt_block(restrictions: list) -> str:
    """
    When restrictions include vegan or vegetarian, return strict LLM instructions so
    recommendations exclude meat (vegan: no animal products at all).
    """
    if not isinstance(restrictions, list):
        return ""
    lowered = [str(x).lower() for x in restrictions if x is not None]
    if any("vegan" in r for r in lowered):
        return (
            "CRITICAL — VEGAN: Every meal must be fully plant-based. "
            "Do NOT include meat, poultry, fish, shellfish, eggs, dairy, honey, gelatin, or other animal-derived ingredients. "
            "Use only vegan proteins and ingredients (legumes, tofu, tempeh, seitan, nuts, seeds, plant milks, etc.).\n"
        )
    if any("vegetarian" in r for r in lowered):
        return (
            "CRITICAL — VEGETARIAN: Every meal must be vegetarian. "
            "Do NOT include meat, poultry, fish, or shellfish. "
            "Eggs and dairy are allowed unless other restrictions forbid them.\n"
        )
    return ""


def _recommend_meals_dalle_size(*, image_quality: str | None = None) -> str:
    """
    DALL·E 2 only supports 256 / 512 / 1024. Default `image_quality` is low (256x256) for speed.
    Legacy: unknown `image_quality` falls back to RECOMMEND_MEALS_DALLE_SIZE env then 256x256.
    """
    q = (image_quality or "low").strip().lower()
    tier = {"low": "256x256", "medium": "512x512", "high": "1024x1024"}
    if q in tier:
        return tier[q]
    allowed = {"256x256", "512x512", "1024x1024"}
    s = (os.getenv("RECOMMEND_MEALS_DALLE_SIZE") or "256x256").strip()
    return s if s in allowed else "256x256"


def _recommend_meals_image_pool_size(num_images: int) -> int:
    """Parallel DALL·E calls; higher = lower wall-clock time (capped)."""
    try:
        w = int(os.getenv("RECOMMEND_MEALS_IMAGE_MAX_WORKERS", "16"))
    except ValueError:
        w = 8
    w = max(1, min(w, 16))
    return min(w, max(1, num_images))


def _meal_image_prompt_for_recommend(name: str, desc: str, *, fast_mode: bool) -> str:
    """Short prompts = slightly faster API handling; truncate description for lower token load."""
    n = (name or "Meal").strip()[:100]
    d = (desc or "").strip()
    d = d[:40] if fast_mode else d[:70]
    if d:
        return f"Food photo: {n}. {d}"
    return f"Food photo: {n}"


def _food_logging_dalle_size() -> str:
    """Default smallest for speed; FOOD_LOGGING_DALLE_SIZE overrides RECOMMEND_MEALS_DALLE_SIZE."""
    allowed = {"256x256", "512x512", "1024x1024"}
    s = (
        os.getenv("FOOD_LOGGING_DALLE_SIZE")
        or os.getenv("RECOMMEND_MEALS_DALLE_SIZE")
        or "256x256"
    ).strip()
    return s if s in allowed else "256x256"


def _food_logging_image_pool_size(num_images: int) -> int:
    try:
        w = int(
            os.getenv("FOOD_LOGGING_IMAGE_MAX_WORKERS")
            or os.getenv("RECOMMEND_MEALS_IMAGE_MAX_WORKERS", "8")
        )
    except ValueError:
        w = 8
    w = max(1, min(w, 16))
    return min(w, max(1, num_images))


def _generate_food_log_meal_image_b64(meal: dict, *, size: str) -> str | None:
    """DALL·E 2 with b64_json — avoids temporary CDN URLs, decode + upload directly to Firebase."""
    try:
        prompt = _meal_image_prompt_for_recommend(
            meal.get("name", "Meal"),
            meal.get("description", ""),
            fast_mode=True,
        )
        img = client.images.generate(
            model="dall-e-2",
            prompt=prompt,
            size=size,
            n=1,
            response_format="b64_json",
        )
        if img and img.data and img.data[0].b64_json:
            return img.data[0].b64_json
        return None
    except Exception:
        return None


def _food_logging_finalize_image_from_b64(b64_json: str | None, default_image_url: str) -> str:
    """Decode DALL·E b64_json and upload directly to Firebase Storage.
    Returns a stable firebasestorage.googleapis.com URL, or default_image_url on failure.
    """
    if not b64_json:
        return default_image_url
    try:
        raw = base64.b64decode(b64_json)
    except Exception as e:
        print(f"[food-logging] image b64 decode failed: {e}")
        return default_image_url
    url = upload_meal_image_bytes_to_storage(raw, "image/png")
    return url if url else default_image_url


def _finalize_meal_image_url(openai_url: str | None, default_image_url: str) -> str:
    """Upload DALL·E CDN URL to Firebase Storage when possible; else return default URL."""
    if not openai_url:
        return default_image_url
    permanent = try_persist_meal_image_from_openai_url(openai_url)
    return permanent or default_image_url


def _recommend_meals_finalize_image_from_b64(
    b64_json: str | None,
    default_image_url: str,
    plan_id: str,
    day_index: int,
    meal_key: str,
) -> str:
    """
    Decode DALL·E b64_json, upload to Firebase Storage, record URL in Firestore (recommend_meal_images).
    """
    if not b64_json:
        return default_image_url
    try:
        raw = base64.b64decode(b64_json)
    except Exception as e:
        print(f"[recommend-meals] image b64 decode failed: {e}")
        return default_image_url
    url = upload_meal_image_bytes_to_storage(raw, "image/png")
    if url:
        save_recommend_meal_image_record(plan_id, day_index, meal_key, url)
        return url
    return default_image_url


def _chat_completion_limit_kw(model: str, max_tokens: int) -> dict:
    """
    Some newer OpenAI chat models reject max_tokens and require max_completion_tokens.
    """
    m = (model or "").lower()
    if m.startswith("gpt-5") or m.startswith("o1") or m.startswith("o3"):
        return {"max_completion_tokens": max_tokens}
    return {"max_tokens": max_tokens}


def _recommend_llm_completion_budget(model: str, base_max: int) -> int:
    """
    GPT-5 / o-series may use much of max_completion_tokens for internal reasoning before
    emitting visible JSON; small budgets can yield empty message.content.
    """
    m = (model or "").lower()
    if m.startswith("gpt-5") or m.startswith("o1") or m.startswith("o3"):
        return max(base_max, 4096)
    return base_max


def _model_prefers_completion_token_param(model: str) -> bool:
    m = (model or "").lower()
    return m.startswith("gpt-5") or m.startswith("o1") or m.startswith("o3")


def _chat_response_message_text(completion) -> str:
    try:
        msg = completion.choices[0].message
        if msg and msg.content:
            return str(msg.content).strip()
    except (IndexError, AttributeError, TypeError):
        pass
    return ""


def _meal_has_name(meal: object) -> bool:
    return isinstance(meal, dict) and bool(str(meal.get("name", "")).strip())


def _unwrap_recommend_day_payload(day_obj: dict, meal_keys: list[str], day_number: int) -> dict:
    """
    Models sometimes return weekly-shaped JSON:
      { "days": [ { "breakfast": {...}, ... } ], "daily_macro_targets": ... }
    with empty top-level meal slots. Prefer the inner day object when it has the real meals.
    """
    if not isinstance(day_obj, dict):
        return day_obj
    days_arr = day_obj.get("days")
    if not isinstance(days_arr, list) or not days_arr:
        return day_obj
    first = days_arr[0]
    if not isinstance(first, dict):
        return day_obj

    def score(d: dict) -> int:
        return sum(1 for k in meal_keys if _meal_has_name(d.get(k)))

    if score(first) > score(day_obj):
        out = dict(first)
        out["day"] = out.get("day", day_number)
        return out
    return day_obj


def _log_recommend_api_request(route_label: str, data: dict) -> None:
    """
    Print request URL, method, and JSON body to server stdout (terminal).
    Set LOG_API_REQUESTS=1 (or true) in the environment to enable.
    """
    if os.getenv("LOG_API_REQUESTS", "").lower() not in ("1", "true", "yes"):
        return
    try:
        print(f"\n[{route_label}] ========== incoming request ==========")
        print(f"[{route_label}] {request.method} {request.url}")
        print(f"[{route_label}] remote: {request.remote_addr}")
        ct = request.headers.get("Content-Type", "")
        print(f"[{route_label}] Content-Type: {ct}")
        body = json.dumps(data, ensure_ascii=False, indent=2)
        if len(body) > 16000:
            body = body[:16000] + "\n... [truncated]"
        print(f"[{route_label}] JSON body:\n{body}")
        print(f"[{route_label}] ========== end request ==========\n")
    except Exception as e:
        print(f"[{route_label}] logging error: {e}")


@app.route("/recommend-meals", methods=["POST"])
def recommend_meals():
    """
    Generate a multi-day meal plan with dynamic num_days and num_meals.
    When include_images is true, image_quality may be low|medium|high (default low = 256px DALL·E).
    Images use response_format=b64_json; the API decodes, uploads to Firebase Storage, stores URLs in
    Firestore (collection recommend_meal_images), and sets each meal's imageUrl to the public URL.
    """
    data = request.get_json(silent=True) or {}
    _log_recommend_api_request("recommend-meals", data)

    # Accept both spellings to be resilient with client payloads.
    dietary_preferences = data.get("dietary_preferences", data.get("dietry_pref", []))
    dietary_restrictions = data.get("dietary_restrictions", data.get("dietry_restr", []))
    # Performance-first defaults for mobile UX:
    # - include_images defaults to False (image generation is the slowest part)
    # - fast_mode defaults to True (single attempt/day, lower token budget; also shorter image prompts)
    # - image_quality: low|medium|high → DALL·E 2 256/512/1024 (default low for fastest images)
    # - RECOMMEND_MEALS_IMAGE_MAX_WORKERS (default 16) caps parallel DALL·E calls
    include_images = data.get("include_images", False)
    include_steps = data.get("include_steps", True)
    include_macros = data.get("include_macros", True)
    if include_steps is None:
        include_steps = True
    if include_macros is None:
        include_macros = True
    include_steps = bool(include_steps)
    include_macros = bool(include_macros)
    fast_mode = data.get("fast_mode", True)
    strict_calorie_alignment = data.get("strict_calorie_alignment", True)
    if strict_calorie_alignment is None:
        strict_calorie_alignment = True
    strict_calorie_alignment = bool(strict_calorie_alignment)
    _img_quality = str(data.get("image_quality") or "low").strip().lower()
    if _img_quality not in ("low", "medium", "high"):
        _img_quality = "low"
    recommend_model = os.getenv("RECOMMEND_MEALS_MODEL", "gpt-5.4-mini")
    num_meals = int(data.get("num_meals", 4))
    if num_meals < 1 or num_meals > 8:
        return jsonify({"error": "num_meals must be between 1 and 8"}), 400
    num_days = int(data.get("num_days", 7))
    if num_days < 1 or num_days > 30:
        return jsonify({"error": "num_days must be between 1 and 30"}), 400
    meal_keys = _get_meal_keys(num_meals)
    plan_id = data.get("plan_id") or str(uuid.uuid4())
    default_image_url = data.get("default_image_url") or os.getenv(
        "DEFAULT_MEAL_IMAGE_URL",
        "https://images.unsplash.com/photo-1498837167922-ddd27525d352?auto=format&fit=crop&w=1200&q=80",
    )

    required_fields = [
        "goal", "gender", "age", "height_cm", "weight_kg",
        "target_weight_kg", "activity_level"
    ]
    missing = [f for f in required_fields if data.get(f) in (None, "")]
    if missing:
        return jsonify({
            "error": "Missing required fields",
            "missing_fields": missing,
        }), 400

    if not isinstance(dietary_preferences, list):
        return jsonify({"error": "dietary_preferences (or dietry_pref) must be an array"}), 400
    if not isinstance(dietary_restrictions, list):
        return jsonify({"error": "dietary_restrictions (or dietry_restr) must be an array"}), 400

    plant_diet_prompt = _dietary_restrictions_plant_based_prompt_block(dietary_restrictions)

    meal_split_percent = data.get("meal_split_percent")
    if not isinstance(meal_split_percent, dict):
        even = round(1.0 / len(meal_keys), 4)
        meal_split_percent = {k: even for k in meal_keys}
    for k in meal_keys:
        if k not in meal_split_percent:
            meal_split_percent[k] = 0.0

    # Estimate daily calorie/macronutrient targets before generation.
    def _calculate_daily_targets(payload: dict) -> dict:
        goal = str(payload.get("goal", "maintain")).lower()
        gender = str(payload.get("gender", "other")).lower()
        age = float(payload.get("age", 25))
        height_cm = float(payload.get("height_cm", 170))
        weight_kg = float(payload.get("weight_kg", 70))
        target_weight_kg = float(payload.get("target_weight_kg", weight_kg))
        activity_level = str(payload.get("activity_level", "lightly_active")).lower()

        # Mifflin-St Jeor BMR constants
        if gender == "male":
            s = 5
        elif gender == "female":
            s = -161
        else:
            s = -78  # midpoint approximation for non-binary/other

        bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age + s

        activity_multipliers = {
            "not_active": 1.2,
            "lightly_active": 1.375,
            "active": 1.55,
            "very_active": 1.725,
        }
        tdee = bmr * activity_multipliers.get(activity_level, 1.375)

        # Goal adjustment
        if goal == "fat_loss":
            calories = tdee - 450
        elif goal == "muscle_gain":
            calories = tdee + 300
        else:
            calories = tdee

        # Sensible floor
        min_cals = 1500 if gender == "male" else 1200
        calories = max(calories, min_cals)

        # Macro split
        # Protein scaled by goal (g/kg), fat baseline 0.8 g/kg, carbs fill remainder.
        protein_per_kg = 1.8 if goal == "fat_loss" else (2.0 if goal == "muscle_gain" else 1.6)
        protein_g = protein_per_kg * target_weight_kg
        fat_g = 0.8 * target_weight_kg

        protein_kcal = protein_g * 4
        fat_kcal = fat_g * 9
        carbs_kcal = max(calories - protein_kcal - fat_kcal, calories * 0.25)
        carbs_g = carbs_kcal / 4

        return {
            "calories": round(calories),
            "protein_g": round(protein_g),
            "carbs_g": round(carbs_g),
            "fat_g": round(fat_g),
            "bmr": round(bmr),
            "tdee": round(tdee),
        }

    target_macros = data.get("target_daily_intake")
    if target_macros is None:
        target_macros = _calculate_daily_targets(data)
    elif not isinstance(target_macros, dict):
        return jsonify({"error": "target_daily_intake must be an object when provided"}), 400

    def _to_float(v):
        try:
            return float(v)
        except Exception:
            return 0.0

    target_macros = {
        "calories": round(_to_float(target_macros.get("calories", 0)), 0),
        "protein_g": round(_to_float(target_macros.get("protein_g", 0)), 1),
        "carbs_g": round(_to_float(target_macros.get("carbs_g", 0)), 1),
        "fat_g": round(_to_float(target_macros.get("fat_g", 0)), 1),
        "bmr": round(_to_float(target_macros.get("bmr", 0)), 0) if "bmr" in target_macros else None,
        "tdee": round(_to_float(target_macros.get("tdee", 0)), 0) if "tdee" in target_macros else None,
    }

    meal_schema_keys = ", ".join([f"\"{k}\": {{...}}" for k in meal_keys])
    meal_names = ", ".join(meal_keys)
    meal_fields_parts = ["name", "description", "ingredients (array)", "cook_time_min (number)", "cuisine", "meal_type"]
    if include_macros:
        meal_fields_parts = [
            "name",
            "description",
            "calories",
            "protein_g",
            "carbs_g",
            "fat_g",
            "ingredients (array)",
            "cook_time_min (number)",
            "cuisine",
            "meal_type",
        ]
    if include_steps:
        meal_fields_parts.append("steps (array)")
    meal_fields_line = ", ".join(meal_fields_parts)
    steps_rule = (
        "Include steps array per meal (max 4 short steps each)."
        if include_steps
        else "Do NOT include a steps field for any meal."
    )
    macros_rule = (
        "Include per-meal calories, protein_g, carbs_g, fat_g (numeric)."
        if include_macros
        else "Do NOT include calories, protein_g, carbs_g, or fat_g on any meal."
    )

    system_prompt = (
        "You are an expert nutritionist and meal planner.\n"
        "Return ONLY valid JSON (no markdown, no commentary).\n"
        f"Build a {num_days}-day plan with {num_meals} meals per day using these keys: {meal_names}.\n"
        "Meals must align with the user's goal, demographics, body stats, activity level, dietary preferences, and dietary restrictions.\n"
        f"{plant_diet_prompt}"
        "Keep meals practical and realistic.\n"
        f"For each meal include ONLY these fields: {meal_fields_line}.\n"
        f"{macros_rule}\n"
        f"{steps_rule}\n\n"
        "Output schema:\n"
        "{\n"
        '  "days": [\n'
        "    {\n"
        '      "day": 1,\n'
        f"      {meal_schema_keys}\n"
        "    }\n"
        "  ],\n"
        '  "daily_macro_targets": {\n'
        '    "calories": 0,\n'
        '    "protein_g": 0,\n'
        '    "carbs_g": 0,\n'
        '    "fat_g": 0\n'
        "  },\n"
        '  "notes": []\n'
        "}\n\n"
        "Rules:\n"
        f'- Exactly {num_days} day objects in "days".\n'
        f"- Day values must be 1..{num_days}.\n"
        "- Use numeric values for cook_time_min, and for calories/protein_g/carbs_g/fat_g when included.\n"
        "- meal_type must be one of: Breakfast, Lunch, Dinner, Snack (mapped from meal key).\n"
        "- Keep output compact: max 8 ingredients and max 4 short steps per meal (when steps are included).\n"
        "- Keep description to a single short line.\n"
        "- Ensure unique meal names across different days in this same weekly plan.\n"
        "- When asked for a single day, return ONLY that day as one object; never return a top-level \"days\" array.\n"
        "- Return valid JSON only.\n"
    )

    user_payload = {
        "plan_id": plan_id,
        "goal": data.get("goal"),
        "gender": data.get("gender"),
        "age": data.get("age"),
        "height_cm": data.get("height_cm"),
        "weight_kg": data.get("weight_kg"),
        "target_weight_kg": data.get("target_weight_kg"),
        "activity_level": data.get("activity_level"),
        "dietary_preferences": dietary_preferences,
        "dietary_restrictions": dietary_restrictions,
        "target_daily_intake": target_macros,
        "meal_split_percent": meal_split_percent,
        "num_days": num_days,
        "num_meals": num_meals,
        "meal_keys": meal_keys,
        "include_steps": include_steps,
        "include_macros": include_macros,
    }

    response_text = None
    try:
        def _generate_day_plan(day_number: int, avoid_names: list[str]) -> tuple[dict | None, str | None]:
            """
            Generate one day's plan. Returns (day_obj, raw_text_on_failure).
            """
            max_attempts = 1 if fast_mode else 2
            if not include_steps and not include_macros:
                max_tokens = 700 if fast_mode else 1000
            elif not include_steps or not include_macros:
                max_tokens = 900 if fast_mode else 1300
            else:
                max_tokens = 1100 if fast_mode else 1600
            for attempt in range(max_attempts):
                no_repeat_instruction = ""
                if avoid_names:
                    no_repeat_instruction = (
                        "Avoid reusing these meal names from prior days in this same weekly plan: "
                        + ", ".join(avoid_names[:120])
                        + ". Use distinct/new meal names for this day."
                    )
                day_messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
                    {
                        "role": "user",
                        "content": (
                            f"Generate ONLY day {day_number} as ONE JSON object with keys: "
                            f"\"day\" (number {day_number}), {meal_schema_keys}. "
                            "Put breakfast/lunch/dinner/etc. at the ROOT of that object. "
                            "Do NOT wrap the day inside a \"days\" array or add weekly-plan wrappers."
                        ),
                    },
                    {"role": "user", "content": no_repeat_instruction},
                ]
                if attempt == 1:
                    day_messages.append({
                        "role": "user",
                        "content": "Regenerate compactly with very short steps and ensure fully valid JSON.",
                    })

                budget = _recommend_llm_completion_budget(recommend_model, max_tokens)
                completion = client.chat.completions.create(
                    model=recommend_model,
                    messages=day_messages,
                    temperature=0.1 if fast_mode else 0.2,
                    **_chat_completion_limit_kw(recommend_model, budget),
                    timeout=25 if fast_mode else 40,
                    response_format={"type": "json_object"},
                )
                raw = _chat_response_message_text(completion)
                if not raw and _model_prefers_completion_token_param(recommend_model):
                    completion = client.chat.completions.create(
                        model=recommend_model,
                        messages=day_messages,
                        temperature=0.1 if fast_mode else 0.2,
                        **_chat_completion_limit_kw(recommend_model, max(budget, 8192)),
                        timeout=60,
                    )
                    raw = _chat_response_message_text(completion)
                try:
                    day_obj = _force_json(raw)
                    if not isinstance(day_obj, dict):
                        continue
                    day_obj = _unwrap_recommend_day_payload(day_obj, meal_keys, day_number)
                    day_obj["day"] = day_obj.get("day", day_number)
                    return day_obj, None
                except Exception:
                    if attempt == (max_attempts - 1):
                        return None, raw
            return None, None

        # Sequential generation so each day can avoid meals from prior generated days.
        day_results = {}
        day_failures = []
        used_meal_names = set()
        for day_number in range(1, num_days + 1):
            try:
                day_obj, failed_raw = _generate_day_plan(day_number, sorted(used_meal_names))
                if day_obj is None:
                    day_failures.append({
                        "day": day_number,
                        "error": "Invalid JSON for day plan",
                        "raw_response": (failed_raw or "")[:500],
                    })
                    continue
                day_results[day_number] = day_obj
                for mk in meal_keys:
                    name = str(day_obj.get(mk, {}).get("name", "")).strip().lower()
                    if name:
                        used_meal_names.add(name)
            except Exception as e:
                day_failures.append({
                    "day": day_number,
                    "error": str(e),
                })

        if day_failures:
            return jsonify({
                "error": "Failed to generate weekly meal plan",
                "user_message": "We couldn't generate your weekly meal plan right now. Please try again.",
                "details": "One or more day plans failed to generate.",
                "failed_days": day_failures,
            }), 500

        days = [day_results[d] for d in range(1, num_days + 1)]
        plan = {"days": {}}

        required_meal_keys = meal_keys
        for i, day_obj in enumerate(days, start=1):
            if not isinstance(day_obj, dict):
                return jsonify({"error": f"Invalid day object at index {i-1}"}), 500
            day_obj["day"] = day_obj.get("day", i)
            for mk in required_meal_keys:
                if mk not in day_obj or not isinstance(day_obj.get(mk), dict):
                    day_obj[mk] = {}
                meal = day_obj[mk]
                meal.setdefault("name", "")
                meal.setdefault("description", "")
                if include_macros:
                    meal.setdefault("calories", 0)
                    meal.setdefault("protein_g", 0)
                    meal.setdefault("carbs_g", 0)
                    meal.setdefault("fat_g", 0)
                else:
                    meal.pop("calories", None)
                    meal.pop("protein_g", None)
                    meal.pop("carbs_g", None)
                    meal.pop("fat_g", None)
                meal.setdefault("ingredients", [])
                if include_steps:
                    meal.setdefault("steps", [])
                else:
                    meal.pop("steps", None)
                meal.setdefault("cook_time_min", 0)
                meal.setdefault("cuisine", "")
                meal.setdefault("imageUrl", default_image_url if include_images else None)
                meal["meal_type"] = _meal_type_from_key(mk)

            # Response shape requested: day number as key, details as value.
            day_payload = dict(day_obj)
            day_payload.pop("day", None)
            # Drop plan-level keys the model sometimes nests inside a "day" by mistake.
            day_payload.pop("days", None)
            day_payload.pop("daily_macro_targets", None)
            plan["days"][str(i)] = day_payload

        if include_images:
            meal_jobs: list[tuple[int, str, dict]] = []
            for day_i, day_obj in enumerate(days, start=1):
                for mk in required_meal_keys:
                    meal_jobs.append((day_i, mk, day_obj.get(mk, {})))

            _dalle_sz = _recommend_meals_dalle_size(image_quality=_img_quality)
            _img_workers = _recommend_meals_image_pool_size(len(meal_jobs))
            _img_prompt_fast = bool(fast_mode) or (_img_quality == "low")

            def _generate_meal_image_b64(job: tuple[int, str, dict]) -> str | None:
                _, _, meal_obj = job
                try:
                    meal_name = meal_obj.get("name", "Meal")
                    meal_desc = meal_obj.get("description", "")
                    prompt = _meal_image_prompt_for_recommend(
                        meal_name, meal_desc, fast_mode=_img_prompt_fast
                    )
                    img = client.images.generate(
                        model="dall-e-2",
                        prompt=prompt,
                        size=_dalle_sz,
                        n=1,
                        response_format="b64_json",
                    )
                    if img and img.data and img.data[0].b64_json:
                        return img.data[0].b64_json
                    return None
                except Exception:
                    return None

            with ThreadPoolExecutor(max_workers=_img_workers) as img_executor:
                futures = {img_executor.submit(_generate_meal_image_b64, job): job for job in meal_jobs}
                raw_quads: list[tuple[int, str, dict, str | None]] = []
                for fut, job in futures.items():
                    day_i, mk, meal_obj = job
                    try:
                        raw_quads.append((day_i, mk, meal_obj, fut.result()))
                    except Exception:
                        raw_quads.append((day_i, mk, meal_obj, None))
            persist_workers = min(max(1, len(raw_quads)), _img_workers)

            def _persist_recommend_image(
                quad: tuple[int, str, dict, str | None],
            ) -> str:
                day_i, mk, _meal_obj, b64 = quad
                return _recommend_meals_finalize_image_from_b64(
                    b64, default_image_url, plan_id, day_i, mk
                )

            with ThreadPoolExecutor(max_workers=persist_workers) as ex_persist:
                finals = list(ex_persist.map(_persist_recommend_image, raw_quads))
            for (_day_i, _mk, meal_obj, _b64), final_u in zip(raw_quads, finals):
                meal_obj["imageUrl"] = final_u

        def _sum_plan_macros():
            total_cal = total_pro = total_carb = total_fat = 0.0
            for day_obj in days:
                for mk in required_meal_keys:
                    meal = day_obj.get(mk, {})
                    total_cal += float(meal.get("calories", 0) or 0)
                    total_pro += float(meal.get("protein_g", 0) or 0)
                    total_carb += float(meal.get("carbs_g", 0) or 0)
                    total_fat += float(meal.get("fat_g", 0) or 0)
            return total_cal, total_pro, total_carb, total_fat

        # Derive average per-day macros from generated meals.
        total_cal, total_pro, total_carb, total_fat = _sum_plan_macros() if include_macros else (0.0, 0.0, 0.0, 0.0)
        target_daily_cal = float(target_macros["calories"])
        estimated_daily_cal = (total_cal / float(num_days)) if total_cal else 0.0
        lower_bound = target_daily_cal * 0.9
        upper_bound = target_daily_cal * 1.1
        correction_applied = False
        correction_scale = 1.0

        # Correction pass: keep generated daily calories within +/-10% of target
        # by scaling meal portions/macros proportionally.
        if include_macros and strict_calorie_alignment and estimated_daily_cal > 0 and (estimated_daily_cal < lower_bound or estimated_daily_cal > upper_bound):
            correction_applied = True
            correction_scale = target_daily_cal / estimated_daily_cal
            # Avoid extreme scaling.
            correction_scale = max(0.7, min(correction_scale, 1.5))

            for day_obj in days:
                for mk in required_meal_keys:
                    meal = day_obj.get(mk, {})
                    meal["calories"] = round(float(meal.get("calories", 0) or 0) * correction_scale, 1)
                    meal["protein_g"] = round(float(meal.get("protein_g", 0) or 0) * correction_scale, 1)
                    meal["carbs_g"] = round(float(meal.get("carbs_g", 0) or 0) * correction_scale, 1)
                    meal["fat_g"] = round(float(meal.get("fat_g", 0) or 0) * correction_scale, 1)
                    meal["portion_scale"] = round(correction_scale, 3)

            total_cal, total_pro, total_carb, total_fat = _sum_plan_macros()

        plan["plan_id"] = plan_id
        plan["num_days"] = num_days
        plan["num_meals"] = num_meals
        plan["include_steps"] = include_steps
        plan["include_macros"] = include_macros
        plan["meal_split_percent"] = meal_split_percent
        plan["daily_macro_targets"] = {
            "calories": target_macros["calories"],
            "protein_g": target_macros["protein_g"],
            "carbs_g": target_macros["carbs_g"],
            "fat_g": target_macros["fat_g"],
            "estimated_from_generated_plan": {
                "calories": round(total_cal / float(num_days), 1),
                "protein_g": round(total_pro / float(num_days), 1),
                "carbs_g": round(total_carb / float(num_days), 1),
                "fat_g": round(total_fat / float(num_days), 1),
            },
            "calorie_alignment": {
                "target_window": {
                    "min": round(lower_bound, 1),
                    "max": round(upper_bound, 1),
                },
                "correction_applied": correction_applied,
                "portion_scale": round(correction_scale, 3),
            },
        }
        plan["target_calories_by_day"] = [
            {"day": day_num, "target_calories": target_macros["calories"]}
            for day_num in range(1, num_days + 1)
        ]
        plan["metabolism"] = {
            "bmr": target_macros["bmr"],
            "tdee": target_macros["tdee"],
        }
        plan.setdefault("notes", [])

        return jsonify(plan)
    except Exception as e:
        return jsonify({
            "error": "Failed to generate weekly meal plan",
            "user_message": "We couldn't generate your weekly meal plan right now. Please try again.",
            "details": str(e),
            "raw_response": response_text[:500] if response_text else None,
        }), 500


@app.route("/recommend-meals/day", methods=["POST"])
def recommend_meals_day():
    """
    Generate one day meal plan (breakfast/lunch/dinner/snacks).
    Designed for client-side parallel calls (day 1..7) with merge on client.
    When include_images is true, image_quality may be low|medium|high (default low = 256px DALL·E).
    Images use response_format=b64_json; the API decodes and uploads to Firebase Storage, stores URLs in
    Firestore (collection recommend_meal_images), and sets each meal's imageUrl to the public URL.
    """
    data = request.get_json(silent=True) or {}
    _log_recommend_api_request("recommend-meals/day", data)

    dietary_preferences = data.get("dietary_preferences", data.get("dietry_pref", []))
    dietary_restrictions = data.get("dietary_restrictions", data.get("dietry_restr", []))
    include_images = data.get("include_images", False)
    include_steps = data.get("include_steps", True)
    include_macros = data.get("include_macros", True)
    if include_steps is None:
        include_steps = True
    if include_macros is None:
        include_macros = True
    include_steps = bool(include_steps)
    include_macros = bool(include_macros)
    fast_mode = data.get("fast_mode", True)
    strict_calorie_alignment = data.get("strict_calorie_alignment", True)
    _img_quality = str(data.get("image_quality") or "low").strip().lower()
    if _img_quality not in ("low", "medium", "high"):
        _img_quality = "low"
    recommend_model = os.getenv("RECOMMEND_MEALS_MODEL", "gpt-5.4-mini")
    num_meals = int(data.get("num_meals", 4))
    if num_meals < 1 or num_meals > 8:
        return jsonify({"error": "num_meals must be between 1 and 8"}), 400
    meal_keys = _get_meal_keys(num_meals)
    plan_id = data.get("plan_id") or str(uuid.uuid4())
    day_number = int(data.get("day_number", 1))
    if day_number < 1 or day_number > 7:
        return jsonify({"error": "day_number must be between 1 and 7"}), 400

    default_image_url = data.get("default_image_url") or os.getenv(
        "DEFAULT_MEAL_IMAGE_URL",
        "https://images.unsplash.com/photo-1498837167922-ddd27525d352?auto=format&fit=crop&w=1200&q=80",
    )

    required_fields = [
        "goal", "gender", "age", "height_cm", "weight_kg",
        "target_weight_kg", "activity_level"
    ]
    missing = [f for f in required_fields if data.get(f) in (None, "")]
    if missing:
        return jsonify({"error": "Missing required fields", "missing_fields": missing}), 400
    if not isinstance(dietary_preferences, list):
        return jsonify({"error": "dietary_preferences (or dietry_pref) must be an array"}), 400
    if not isinstance(dietary_restrictions, list):
        return jsonify({"error": "dietary_restrictions (or dietry_restr) must be an array"}), 400

    plant_diet_prompt = _dietary_restrictions_plant_based_prompt_block(dietary_restrictions)

    # Optional split to track calorie distribution per meal
    meal_split_percent = data.get("meal_split_percent")
    if not isinstance(meal_split_percent, dict):
        # default even split across requested meal count
        even = round(1.0 / len(meal_keys), 4)
        meal_split_percent = {k: even for k in meal_keys}
    for k in meal_keys:
        if k not in meal_split_percent:
            meal_split_percent[k] = 0.0

    # Reuse same target estimation logic as weekly endpoint when target is not provided.
    def _calculate_daily_targets(payload: dict) -> dict:
        goal = str(payload.get("goal", "maintain")).lower()
        gender = str(payload.get("gender", "other")).lower()
        age = float(payload.get("age", 25))
        height_cm = float(payload.get("height_cm", 170))
        weight_kg = float(payload.get("weight_kg", 70))
        target_weight_kg = float(payload.get("target_weight_kg", weight_kg))
        activity_level = str(payload.get("activity_level", "lightly_active")).lower()

        s = 5 if gender == "male" else (-161 if gender == "female" else -78)
        bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age + s
        activity_multipliers = {
            "not_active": 1.2,
            "lightly_active": 1.375,
            "active": 1.55,
            "very_active": 1.725,
        }
        tdee = bmr * activity_multipliers.get(activity_level, 1.375)
        if goal == "fat_loss":
            calories = tdee - 450
        elif goal == "muscle_gain":
            calories = tdee + 300
        else:
            calories = tdee
        min_cals = 1500 if gender == "male" else 1200
        calories = max(calories, min_cals)
        protein_per_kg = 1.8 if goal == "fat_loss" else (2.0 if goal == "muscle_gain" else 1.6)
        protein_g = protein_per_kg * target_weight_kg
        fat_g = 0.8 * target_weight_kg
        protein_kcal = protein_g * 4
        fat_kcal = fat_g * 9
        carbs_kcal = max(calories - protein_kcal - fat_kcal, calories * 0.25)
        carbs_g = carbs_kcal / 4
        return {
            "calories": round(calories),
            "protein_g": round(protein_g),
            "carbs_g": round(carbs_g),
            "fat_g": round(fat_g),
            "bmr": round(bmr),
            "tdee": round(tdee),
        }

    # Important: if client passes target_daily_intake, do NOT recalculate it.
    # Only calculate when it is missing.
    target_daily_intake = data.get("target_daily_intake")
    if target_daily_intake is None:
        target_daily_intake = _calculate_daily_targets(data)
    elif not isinstance(target_daily_intake, dict):
        return jsonify({"error": "target_daily_intake must be an object when provided"}), 400

    # Safe numeric casting (handles strings coming from client).
    def _to_float(v):
        try:
            return float(v)
        except Exception:
            return 0.0

    target_daily_intake = {
        "calories": round(_to_float(target_daily_intake.get("calories", 0)), 0),
        "protein_g": round(_to_float(target_daily_intake.get("protein_g", 0)), 1),
        "carbs_g": round(_to_float(target_daily_intake.get("carbs_g", 0)), 1),
        "fat_g": round(_to_float(target_daily_intake.get("fat_g", 0)), 1),
        "bmr": round(_to_float(target_daily_intake.get("bmr", 0)), 0) if "bmr" in target_daily_intake else None,
        "tdee": round(_to_float(target_daily_intake.get("tdee", 0)), 0) if "tdee" in target_daily_intake else None,
    }

    target_daily_cal = float(target_daily_intake.get("calories", 0) or 0)

    day_meal_schema_keys = ", ".join([f"\"{k}\": {{...}}" for k in meal_keys])
    day_meal_names = ", ".join(meal_keys)

    with _plan_meal_history_lock:
        prior_meal_names = sorted(_plan_meal_name_history.get(plan_id, set()))

    meal_fields_parts = ["name", "description", "ingredients (array)", "cook_time_min (number)", "cuisine", "meal_type"]
    if include_macros:
        meal_fields_parts = [
            "name",
            "description",
            "calories",
            "protein_g",
            "carbs_g",
            "fat_g",
            "ingredients (array)",
            "cook_time_min (number)",
            "cuisine",
            "meal_type",
        ]
    if include_steps:
        meal_fields_parts.append("steps (array)")
    meal_fields_line = ", ".join(meal_fields_parts)

    steps_rule = (
        "Include steps array per meal (max 4 short steps each)."
        if include_steps
        else "Do NOT include a steps field for any meal."
    )
    macros_rule = (
        "Include per-meal calories, protein_g, carbs_g, fat_g (numeric)."
        if include_macros
        else "Do NOT include calories, protein_g, carbs_g, or fat_g on any meal."
    )

    no_repeat_rule = ""
    if prior_meal_names:
        no_repeat_rule = (
            "For this plan_id, avoid reusing meal names from previous days: "
            + ", ".join(prior_meal_names[:120])
            + ". Use distinct/new meal names for this day."
        )

    system_prompt = f"""You are an expert nutritionist and meal planner.
Return ONLY valid JSON.
Generate ONE day meal plan with exactly {num_meals} meals using these keys: {day_meal_names}.
Meals must align with the user's goal, demographics, body stats, activity level, dietary preferences, and dietary restrictions.
{plant_diet_prompt}For each meal include ONLY these fields: {meal_fields_line}.
{macros_rule}
{steps_rule}
{no_repeat_rule}
Keep compact: max 8 ingredients per meal.
"""

    user_payload = {
        "plan_id": plan_id,
        "day_number": day_number,
        "goal": data.get("goal"),
        "gender": data.get("gender"),
        "age": data.get("age"),
        "height_cm": data.get("height_cm"),
        "weight_kg": data.get("weight_kg"),
        "target_weight_kg": data.get("target_weight_kg"),
        "activity_level": data.get("activity_level"),
        "dietary_preferences": dietary_preferences,
        "dietary_restrictions": dietary_restrictions,
        "target_daily_intake": target_daily_intake,
        "meal_split_percent": meal_split_percent,
        "num_meals": num_meals,
        "meal_keys": meal_keys,
        "include_steps": include_steps,
        "include_macros": include_macros,
        "avoid_meal_names": prior_meal_names,
    }

    response_text = None
    try:
        max_attempts = 1 if fast_mode else 2
        if not include_steps and not include_macros:
            max_tokens = 700 if fast_mode else 1000
        elif not include_steps or not include_macros:
            max_tokens = 900 if fast_mode else 1300
        else:
            max_tokens = 1100 if fast_mode else 1600
        day_obj = None
        for attempt in range(max_attempts):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
                {"role": "user", "content": f"Return JSON schema: {{\"day\": {day_number}, {day_meal_schema_keys}}}"},
            ]
            if attempt == 1:
                messages.append({"role": "user", "content": "Regenerate with stricter compactness and valid JSON."})
            budget = _recommend_llm_completion_budget(recommend_model, max_tokens)
            completion = client.chat.completions.create(
                model=recommend_model,
                messages=messages,
                temperature=0.1 if fast_mode else 0.2,
                **_chat_completion_limit_kw(recommend_model, budget),
                timeout=25 if fast_mode else 40,
                response_format={"type": "json_object"},
            )
            response_text = _chat_response_message_text(completion)
            if not response_text and _model_prefers_completion_token_param(recommend_model):
                completion = client.chat.completions.create(
                    model=recommend_model,
                    messages=messages,
                    temperature=0.1 if fast_mode else 0.2,
                    **_chat_completion_limit_kw(recommend_model, max(budget, 8192)),
                    timeout=60,
                )
                response_text = _chat_response_message_text(completion)
            try:
                parsed = _force_json(response_text)
                if isinstance(parsed, dict):
                    day_obj = parsed
                    break
            except Exception:
                day_obj = None

        if day_obj is None:
            raise ValueError("Could not parse one-day JSON output")

        day_obj["day"] = day_number
        required_meal_keys = meal_keys
        for mk in required_meal_keys:
            if mk not in day_obj or not isinstance(day_obj.get(mk), dict):
                day_obj[mk] = {}
            meal = day_obj[mk]
            meal.setdefault("name", "")
            meal.setdefault("description", "")
            if include_macros:
                meal.setdefault("calories", 0)
                meal.setdefault("protein_g", 0)
                meal.setdefault("carbs_g", 0)
                meal.setdefault("fat_g", 0)
            else:
                meal.pop("calories", None)
                meal.pop("protein_g", None)
                meal.pop("carbs_g", None)
                meal.pop("fat_g", None)
            meal.setdefault("ingredients", [])
            if include_steps:
                meal.setdefault("steps", [])
            else:
                meal.pop("steps", None)
            meal.setdefault("cook_time_min", 0)
            meal.setdefault("cuisine", "")
            meal.setdefault("imageUrl", default_image_url if include_images else None)
            meal["meal_type"] = _meal_type_from_key(mk)

        # Update per-plan de-dup history with generated meal names.
        generated_names = {
            str(day_obj.get(k, {}).get("name", "")).strip().lower()
            for k in required_meal_keys
            if str(day_obj.get(k, {}).get("name", "")).strip()
        }
        if generated_names:
            with _plan_meal_history_lock:
                _plan_meal_name_history[plan_id].update(generated_names)

        if include_images:
            _dalle_sz = _recommend_meals_dalle_size(image_quality=_img_quality)
            _img_workers = _recommend_meals_image_pool_size(len(required_meal_keys))
            _img_prompt_fast = bool(fast_mode) or (_img_quality == "low")

            def _generate_meal_image_b64(meal_obj: dict) -> str | None:
                try:
                    prompt = _meal_image_prompt_for_recommend(
                        meal_obj.get("name", "Meal"),
                        meal_obj.get("description", ""),
                        fast_mode=_img_prompt_fast,
                    )
                    img = client.images.generate(
                        model="dall-e-2",
                        prompt=prompt,
                        size=_dalle_sz,
                        n=1,
                        response_format="b64_json",
                    )
                    if img and img.data and img.data[0].b64_json:
                        return img.data[0].b64_json
                    return None
                except Exception:
                    return None

            with ThreadPoolExecutor(max_workers=_img_workers) as ex:
                futs = {
                    k: ex.submit(_generate_meal_image_b64, day_obj.get(k, {}))
                    for k in required_meal_keys
                }
                raws: list[tuple[str, str | None]] = []
                for k in required_meal_keys:
                    try:
                        raws.append((k, futs[k].result()))
                    except Exception:
                        raws.append((k, None))
            persist_workers = min(max(1, len(raws)), _img_workers)

            def _persist_day_image(pair: tuple[str, str | None]) -> str:
                meal_key, b64 = pair
                return _recommend_meals_finalize_image_from_b64(
                    b64, default_image_url, plan_id, day_number, meal_key
                )

            with ThreadPoolExecutor(max_workers=persist_workers) as ex_persist:
                finals = list(ex_persist.map(_persist_day_image, raws))
            for k, u in zip([x[0] for x in raws], finals):
                day_obj[k]["imageUrl"] = u

        # Day totals + optional correction alignment to target calories (requires per-meal macros)
        totals = {"calories": 0.0, "protein_g": 0.0, "carbs_g": 0.0, "fat_g": 0.0}
        if include_macros:
            for k in required_meal_keys:
                m = day_obj.get(k, {})
                totals["calories"] += float(m.get("calories", 0) or 0)
                totals["protein_g"] += float(m.get("protein_g", 0) or 0)
                totals["carbs_g"] += float(m.get("carbs_g", 0) or 0)
                totals["fat_g"] += float(m.get("fat_g", 0) or 0)

        lower = target_daily_cal * 0.9
        upper = target_daily_cal * 1.1
        correction_applied = False
        portion_scale = 1.0
        if (
            include_macros
            and strict_calorie_alignment
            and totals["calories"] > 0
            and (totals["calories"] < lower or totals["calories"] > upper)
        ):
            correction_applied = True
            portion_scale = max(0.7, min(target_daily_cal / totals["calories"], 1.5))
            for k in required_meal_keys:
                m = day_obj.get(k, {})
                m["calories"] = round(float(m.get("calories", 0) or 0) * portion_scale, 1)
                m["protein_g"] = round(float(m.get("protein_g", 0) or 0) * portion_scale, 1)
                m["carbs_g"] = round(float(m.get("carbs_g", 0) or 0) * portion_scale, 1)
                m["fat_g"] = round(float(m.get("fat_g", 0) or 0) * portion_scale, 1)
                m["portion_scale"] = round(portion_scale, 3)
            totals = {"calories": 0.0, "protein_g": 0.0, "carbs_g": 0.0, "fat_g": 0.0}
            for k in required_meal_keys:
                m = day_obj.get(k, {})
                totals["calories"] += float(m.get("calories", 0) or 0)
                totals["protein_g"] += float(m.get("protein_g", 0) or 0)
                totals["carbs_g"] += float(m.get("carbs_g", 0) or 0)
                totals["fat_g"] += float(m.get("fat_g", 0) or 0)

        response = {
            "plan_id": plan_id,
            "day_number": day_number,
            "include_steps": include_steps,
            "include_macros": include_macros,
            "day": day_obj,
            "target_daily_intake": {
                "calories": target_daily_intake.get("calories", 0),
                "protein_g": target_daily_intake.get("protein_g", 0),
                "carbs_g": target_daily_intake.get("carbs_g", 0),
                "fat_g": target_daily_intake.get("fat_g", 0),
            },
            "daily_totals": {
                "calories": round(totals["calories"], 1),
                "protein_g": round(totals["protein_g"], 1),
                "carbs_g": round(totals["carbs_g"], 1),
                "fat_g": round(totals["fat_g"], 1),
            },
            "variance_from_target": {
                "calories": round(totals["calories"] - target_daily_cal, 1),
                "calories_pct": round(((totals["calories"] - target_daily_cal) / target_daily_cal) * 100, 1) if target_daily_cal else 0,
                "target_window": {"min": round(lower, 1), "max": round(upper, 1)},
                "correction_applied": correction_applied,
                "portion_scale": round(portion_scale, 3),
            },
            "meal_split_percent": meal_split_percent,
            "metabolism": {
                "bmr": target_daily_intake.get("bmr"),
                "tdee": target_daily_intake.get("tdee"),
            },
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({
            "error": "Failed to generate day meal plan",
            "user_message": "We couldn't generate your day meal plan right now. Please try again.",
            "details": str(e),
            "raw_response": response_text[:500] if response_text else None,
        }), 500

#####video only code starts here#####

# ---------------------------
# Helper: Transcribe Audio
# ---------------------------
def transcribe_audio_file(
    audio_bytes: bytes,
    file_name: str,
    *,
    language: str | None = None,
) -> str:
    """
    Whisper transcription. Optional `language` (ISO-639-1, e.g. en) skips language detection and can reduce latency.
    """
    try:
        # Create file-like object
        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = file_name

        t_kwargs: dict = {
            "model": "whisper-1",  # Cost-effective: Much cheaper than gpt-4o-transcribe, same quality
            "file": audio_file,
            "response_format": "text",
        }
        if language:
            t_kwargs["language"] = language
        transcription = client.audio.transcriptions.create(**t_kwargs)

        transcript_text = (
            transcription if isinstance(transcription, str) else str(transcription)
        ).strip()

        if not transcript_text:
            raise ValueError("Transcription returned empty text")

        return transcript_text

    except Exception as e:
        raise Exception(f"Transcription failed: {e}")


# ---------------------------
# Helper: Extract Ingredients
# ---------------------------
def extract_ingredients(transcript: str):
    try:
        system_prompt = (
            "You are a food ingredient extraction assistant.\n"
            "Given a sentence, extract ONLY the food ingredients listed.\n"
            "Return them as a comma-separated list with NO extra explanations.\n"
            "Example:\n"
            "Input: 'I have chicken breast, broccoli and garlic.'\n"
            "Output: 'chicken breast, broccoli, garlic'\n"
        )

        user_prompt = f"Extract ingredients from: \"{transcript}\""

        completion = client.chat.completions.create(
            model="gpt-4o-mini",  # small + fast + cheap
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )

        content = completion.choices[0].message.content.strip()

        # Comma-split + cleanup
        raw_items = content.replace("\n", ",").split(",")
        ingredients = [
            item.strip()
            for item in raw_items
            if item.strip()
            and ":" not in item.lower()
            and "sorry" not in item.lower()
        ]

        # Remove duplicates while keeping order
        seen = set()
        unique = []
        for ing in ingredients:
            low = ing.lower()
            if low not in seen:
                seen.add(low)
                unique.append(ing)

        return unique

    except Exception as e:
        raise Exception(f"Ingredient extraction failed: {e}")


def _voice_logging_completion_budget(*, fast: bool) -> int:
    """Shared max output tokens for voice + photo food logging LLM calls."""
    default_cap = 500 if fast else 800
    try:
        max_out = int(os.getenv("VOICE_LOGGING_MAX_COMPLETION_TOKENS", str(default_cap)))
    except ValueError:
        max_out = default_cap
    if fast:
        max_out = min(max_out, 600)
    return max(256, min(max_out, 4000))


def _normalize_food_log_meals_from_obj(obj: dict) -> list[dict]:
    """Normalize LLM JSON `meals` array into the /food-logging meal shape."""
    meals = obj.get("meals", [])
    if not isinstance(meals, list):
        return []
    out: list[dict] = []
    for m in meals:
        if not isinstance(m, dict):
            continue
        try:
            cals = float(m.get("calories", 0) or 0)
        except (TypeError, ValueError):
            cals = 0.0

        def _nf(key: str) -> float:
            try:
                return float(m.get(key, 0) or 0)
            except (TypeError, ValueError):
                return 0.0

        out.append({
            "name": str(m.get("name", "Meal")).strip() or "Meal",
            "description": str(m.get("description", "")).strip(),
            "meal_type": str(m.get("meal_type", "Unknown")).strip() or "Unknown",
            "calories": int(round(cals)),
            "protein_g": round(_nf("protein_g"), 1),
            "carbs_g": round(_nf("carbs_g"), 1),
            "fat_g": round(_nf("fat_g"), 1),
        })
    return out


def extract_meals_from_voice_log_transcript(transcript: str, *, fast: bool = False) -> list[dict]:
    """
    Parse spoken food log into structured meals with estimated calories and macros.
    Tuned for latency: small completion budget, compact prompt.
    """
    model = os.getenv("VOICE_LOGGING_MODEL", "gpt-4o-mini")
    max_out = _voice_logging_completion_budget(fast=fast)

    system_prompt = (
        "Extract food log items from the transcript. Return ONLY JSON: "
        '{"meals":[{"name":"","description":"","meal_type":"Breakfast|Lunch|Dinner|Snack|Unknown",'
        '"calories":0,"protein_g":0,"carbs_g":0,"fat_g":0}]}. '
        "One entry per distinct meal/snack; estimate kcal and macros for typical portions. "
        "Non-food or empty transcript → {\"meals\":[]}. No markdown."
    )
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": transcript[:12000]},
        ],
        temperature=0.1,
        **_chat_completion_limit_kw(model, max_out),
        response_format={"type": "json_object"},
    )
    raw = completion.choices[0].message.content
    if not raw:
        return []
    obj = _force_json(raw)
    return _normalize_food_log_meals_from_obj(obj)


def extract_meals_from_food_photo(image_data_url: str, *, fast: bool = False) -> tuple[list[dict], str]:
    """
    Vision: identify food in a meal photo and return the same meal objects as voice/text logging.
    Second return value is a short summary for the `transcript` field (what was visible).
    """
    model = os.getenv("FOOD_LOGGING_PHOTO_MODEL") or os.getenv("VOICE_LOGGING_MODEL", "gpt-4o-mini")
    max_out = _voice_logging_completion_budget(fast=fast)

    system_prompt = (
        "You analyze photos of food for a meal logging app. Identify dishes and estimate portions. "
        "Return ONLY JSON: "
        '{"summary":"1–2 sentences describing what food is visible (or empty if none).","meals":['
        '{"name":"","description":"","meal_type":"Breakfast|Lunch|Dinner|Snack|Unknown",'
        '"calories":0,"protein_g":0,"carbs_g":0,"fat_g":0}]}. '
        "One meal entry per distinct dish; if one plate has several items, you may use one combined entry with a clear description. "
        "Estimate kcal and macros for likely portion sizes from the image. "
        "If the image shows no edible food, is too blurry, or is not food → {\"summary\":\"\",\"meals\":[]}. "
        "No markdown."
    )
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze this food photo and return the JSON."},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            },
        ],
        temperature=0.1,
        **_chat_completion_limit_kw(model, max_out),
        response_format={"type": "json_object"},
    )
    raw = completion.choices[0].message.content
    if not raw:
        return [], ""
    obj = _force_json(raw)
    summary = str(obj.get("summary", "")).strip()
    meals = _normalize_food_log_meals_from_obj(obj)
    return meals, summary


def _multipart_upload_mimetype(uploaded_file) -> str | None:
    for attr in ("mimetype", "content_type"):
        v = getattr(uploaded_file, attr, None)
        if v:
            return str(v).strip()
    return None


def _looks_like_image_multipart_file(filename: str | None, mimetype: str | None) -> bool:
    """True if upload is likely an image (extension or Content-Type)."""
    mt = (mimetype or "").strip().lower()
    if mt.startswith("image/"):
        return True
    fn = (filename or "").lower()
    return fn.endswith(
        (".jpg", ".jpeg", ".png", ".gif", ".webp", ".heic", ".heif", ".bmp", ".tif", ".tiff"),
    )


def _content_type_for_image_filename(filename: str) -> str:
    fn = filename.lower()
    if fn.endswith((".jpg", ".jpeg")):
        return "image/jpeg"
    if fn.endswith(".png"):
        return "image/png"
    if fn.endswith(".gif"):
        return "image/gif"
    if fn.endswith(".webp"):
        return "image/webp"
    if fn.endswith((".heic", ".heif")):
        return "image/heic"
    if fn.endswith((".bmp",)):
        return "image/bmp"
    if fn.endswith((".tif", ".tiff")):
        return "image/tiff"
    return "image/jpeg"


def _food_logging_image_data_url_from_request() -> tuple[str | None, str | None]:
    """
    Build a data: URL for vision from multipart or JSON (imageBase64 + imageFormat).

    Multipart field names tried in order: image, photo, file, audio.
    The audio key is accepted only when the part looks like an image (some clients reuse the voice field).

    Returns (data_url, None) or (None, error_message for 400 responses).
    """
    for key in ("image", "photo", "file", "audio"):
        if key not in request.files:
            continue
        uploaded_file = request.files.get(key)
        if not uploaded_file or not uploaded_file.filename:
            continue
        mt_part = _multipart_upload_mimetype(uploaded_file)
        if not _looks_like_image_multipart_file(uploaded_file.filename, mt_part):
            continue
        image_bytes = uploaded_file.read()
        if not image_bytes:
            return None, "Uploaded image file is empty"
        content_type = _content_type_for_image_filename(uploaded_file.filename)
        mt = (mt_part or "").strip().lower()
        if mt.startswith("image/"):
            content_type = mt
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        return f"data:{content_type};base64,{image_base64}", None

    if request.is_json:
        data = request.get_json(silent=True) or {}
        image_base64 = data.get("imageBase64") or data.get("image")
        if image_base64:
            if "," in str(image_base64):
                image_base64 = str(image_base64).split(",")[-1]
            image_format = (data.get("imageFormat") or "jpg").lower()
            format_to_mime = {
                "jpg": "image/jpeg",
                "jpeg": "image/jpeg",
                "png": "image/png",
                "gif": "image/gif",
                "webp": "image/webp",
            }
            content_type = format_to_mime.get(image_format, "image/jpeg")
            return f"data:{content_type};base64,{image_base64}", None

    return None, (
        "Image is required when action=photo. "
        "Use multipart fields image, photo, file, or audio (if the file is an image), or JSON imageBase64."
    )


# ---------------------------
# MAIN ENDPOINT (File Upload Only)
# ---------------------------
@app.route("/voice-ingredients", methods=["POST"])
def voice_ingredients():
    """
    Accepts audio file only.
    Returns list of ingredients + transcript.
    """
    try:
        # Check if audio file is in request
        if 'audio' not in request.files:
            return jsonify({"error": "Audio file is required"}), 400
        
        audio_file = request.files['audio']
        if not audio_file.filename:
            return jsonify({"error": "Uploaded audio file is empty"}), 400

        # Read audio file
        audio_bytes = audio_file.read()
        if not audio_bytes:
            return jsonify({"error": "Uploaded audio file is empty"}), 400

        print(f"Received audio file: {audio_file.filename}")

        # 1. Transcribe voice → text
        transcript = transcribe_audio_file(audio_bytes, audio_file.filename)

        # 2. Extract ingredients from transcript
        ingredients = extract_ingredients(transcript)

        response = {
            "ingredients": ingredients,
            "transcript": transcript,
            "message": f"Found {len(ingredients)} ingredient(s).",
        }

        return jsonify(response)

    except Exception as e:
        print(f"Error in voice_ingredients: {str(e)}")
        return jsonify({
            "error": "Unexpected server error",
            "user_message": "We couldn't process the audio. Please try again or type your ingredients instead.",
            "details": str(e),
        }), 500


@app.route("/food-logging", methods=["POST"])
def food_logging():
    """
    Form fields:
    - action: "voice" (default) — requires multipart audio; runs transcribe_audio_file (Whisper).
      "text" — provide transcript via form field transcript or text; no audio.
      "photo" — multipart image under `image`, `photo`, `file`, or `audio` (if the upload is an image),
        or JSON with imageBase64 (+ imageFormat). Uses vision to identify food and estimate macros;
        `transcript` is a short summary of what was seen.
    - audio: file (required when action=voice)
    - transcript / text: food log text (required when action=text)
    - fast (optional): 1/true — smaller LLM completion budget for meal extraction only.
    Each meal in the response includes imageUrl: for action=photo this is always the same image sent in the
    request (data URL); for voice/text, DALL·E then upload to Firebase Storage when configured (stable URL),
    otherwise the temporary OpenAI CDN URL or DEFAULT_MEAL_IMAGE_URL fallback.
    Whisper: optional env FOOD_LOGGING_WHISPER_LANGUAGE (e.g. en) can reduce latency.
    Photo: optional FOOD_LOGGING_PHOTO_MODEL (defaults to VOICE_LOGGING_MODEL / gpt-4o-mini).
    """
    try:
        json_body = request.get_json(silent=True) or {}
        action = (
            request.form.get("action")
            or request.args.get("action")
            or json_body.get("action")
            or "voice"
        )
        action = str(action).strip().lower()
        if action not in ("voice", "text", "photo"):
            return jsonify({
                "error": "Invalid action",
                "allowed": ["voice", "text", "photo"],
            }), 400

        default_image_url = os.getenv(
            "DEFAULT_MEAL_IMAGE_URL",
            "https://images.unsplash.com/photo-1498837167922-ddd27525d352?auto=format&fit=crop&w=1200&q=80",
        )
        fast_flag = (
            (request.form.get("fast") or request.args.get("fast") or "").strip().lower() in ("1", "true", "yes")
            or str(json_body.get("fast", "")).strip().lower() in ("1", "true", "yes")
        )

        if action == "voice":
            if "audio" not in request.files:
                return jsonify({"error": "Audio file is required when action=voice"}), 400

            audio_file = request.files["audio"]
            if not audio_file.filename:
                return jsonify({"error": "Uploaded audio file is empty"}), 400

            audio_bytes = audio_file.read()
            if not audio_bytes:
                return jsonify({"error": "Uploaded audio file is empty"}), 400

            print(f"[food-logging] action=voice audio: {audio_file.filename}, {len(audio_bytes)} bytes")

            whisper_lang = (os.getenv("FOOD_LOGGING_WHISPER_LANGUAGE") or "").strip() or None
            transcript = transcribe_audio_file(
                audio_bytes,
                audio_file.filename,
                language=whisper_lang,
            )
        elif action == "text":
            transcript = (request.form.get("transcript") or request.form.get("text") or "").strip()
            if not transcript:
                return jsonify({
                    "error": "transcript or text is required when action=text",
                }), 400
            print(f"[food-logging] action=text, len={len(transcript)} chars")
        else:
            # action == "photo"
            photo_data_url, img_err = _food_logging_image_data_url_from_request()
            if img_err or not photo_data_url:
                return jsonify({"error": img_err or "Image required"}), 400
            print(f"[food-logging] action=photo, data_url_len={len(photo_data_url)}")
            meals, transcript = extract_meals_from_food_photo(photo_data_url, fast=fast_flag)
            if not meals and not (transcript or "").strip():
                return jsonify({
                    "error": "No food detected",
                    "transcript": "",
                    "meals": [],
                    "user_message": "We couldn't identify food in this photo. Try a clearer picture, better lighting, or use voice or text logging.",
                }), 400
            # Skip the shared voice/text path below
            if not meals:
                return jsonify({
                    "transcript": transcript,
                    "meals": [],
                    "message": "No structured meals extracted from this photo.",
                    "action": "photo",
                })

            for m in meals:
                m["imageUrl"] = photo_data_url

            return jsonify({
                "transcript": transcript,
                "meals": meals,
                "message": f"Logged {len(meals)} meal(s) from photo.",
                "action": "photo",
            })

        if not transcript or not transcript.strip():
            return jsonify({
                "error": "No speech detected",
                "transcript": "",
                "meals": [],
                "user_message": "We couldn't detect speech in this recording. Try again or speak more clearly.",
            }), 400

        meals = extract_meals_from_voice_log_transcript(transcript, fast=fast_flag)

        if meals:
            dalle_sz = _food_logging_dalle_size()
            workers = _food_logging_image_pool_size(len(meals))

            # Generate b64_json and upload to Firebase in a single parallel pass.
            # This avoids the two-pool pattern (generate URL → download → re-upload)
            # and always returns a stable firebasestorage.googleapis.com URL.
            def _generate_and_persist(m: dict) -> str:
                b64 = _generate_food_log_meal_image_b64(m, size=dalle_sz)
                return _food_logging_finalize_image_from_b64(b64, default_image_url)

            with ThreadPoolExecutor(max_workers=workers) as ex:
                finalized = list(ex.map(_generate_and_persist, meals))
            for meal, url in zip(meals, finalized):
                meal["imageUrl"] = url

        return jsonify({
            "transcript": transcript,
            "meals": meals,
            "message": f"Logged {len(meals)} meal(s).",
        })

    except Exception as e:
        print(f"Error in food_logging: {str(e)}")
        return jsonify({
            "error": "Unexpected server error",
            "user_message": "We couldn't process your food log. Please try again or type what you ate.",
            "details": str(e),
        }), 500


# ---------- Security: SSRF protection ----------
def _is_private_host(hostname: str) -> bool:
    """
    Resolve hostname and block private / local / link-local ranges.
    Prevents SSRF attacks like http://127.0.0.1:... or cloud metadata IPs.
    """
    if not hostname:
        return True

    # Block obvious localhost names
    lowered = hostname.lower()
    if lowered in {"localhost", "localhost.localdomain"}:
        return True

    try:
        infos = socket.getaddrinfo(hostname, None)
    except socket.gaierror:
        return True

    for family, _, _, _, sockaddr in infos:
        ip_str = sockaddr[0]
        ip = ipaddress.ip_address(ip_str)

        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        ):
            return True

        # Block AWS/GCP/Azure metadata IP (most important)
        if ip_str == "169.254.169.254":
            return True

    return False


def validate_video_url(video_url: str):
    if not video_url or not isinstance(video_url, str):
        return False, "videoUrl is required"

    parsed = urlparse(video_url)
    if parsed.scheme not in {"http", "https"}:
        return False, "Only http/https URLs are allowed"

    if not parsed.netloc:
        return False, "Invalid URL"

    hostname = parsed.hostname
    if _is_private_host(hostname):
        return False, "URL host is not allowed"

    return True, ""


# ---------- yt-dlp helpers ----------
def ytdlp_base_opts(temp_dir: str, video_url: str = None):
    """
    Base yt-dlp options for audio extraction.
    Returns a dictionary of options that can be further customized.
    
    Args:
        temp_dir: Temporary directory for output files
        video_url: Optional video URL to determine if proxy should be used (YouTube only)
    """
    opts = {
        # IMPORTANT: robust selector with fallbacks (handles many edge cases)
        "format": "bestaudio/best/best",
        "outtmpl": os.path.join(temp_dir, "%(id)s.%(ext)s"),
        "restrictfilenames": True,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "socket_timeout": 20,
        "retries": 2,
        "fragment_retries": 2,
        # Helps for YouTube signature issues & format availability
        "extractor_args": {
            "youtube": {"player_client": ["android", "web"]}
        },
        # Convert to mp3
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }],
    }

    # Cookies greatly improve TikTok/Instagram reliability (and some YouTube cases)
    if YTDLP_COOKIES_FILE and os.path.exists(YTDLP_COOKIES_FILE):
        opts["cookiefile"] = YTDLP_COOKIES_FILE
    
    # Add proxy ONLY for YouTube URLs (not for TikTok/Instagram/webpages)
    if YT_PROXY and video_url and is_youtube_url(video_url):
        opts["proxy"] = YT_PROXY
        print(f"🌐 Using proxy for YouTube: {YT_PROXY}")

    return opts


def get_video_metadata(video_url: str):
    """
    Uses yt-dlp to fetch metadata without downloading.
    """
    opts = {"quiet": True, "no_warnings": True, "noplaylist": True}
    # Add proxy ONLY for YouTube URLs
    if YT_PROXY and is_youtube_url(video_url):
        opts["proxy"] = YT_PROXY
        print(f"🌐 Using proxy for YouTube metadata: {YT_PROXY}")
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(video_url, download=False)
    return info


def download_audio_mp3(video_url: str):
    """
    Downloads video audio and returns mp3 bytes + basic metadata.
    """
    temp_dir = tempfile.mkdtemp()
    try:
        # Pre-check duration before downloading (best effort)
        info = get_video_metadata(video_url)
        duration = info.get("duration")  # seconds
        title = info.get("title") or ""
        extractor = info.get("extractor_key") or info.get("extractor") or ""

        if duration and duration > MAX_VIDEO_SECONDS:
            raise ValueError(f"Video too long ({duration}s). Max allowed is {MAX_VIDEO_SECONDS}s")

        opts = ytdlp_base_opts(temp_dir, video_url)

        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([video_url])

        mp3s = [f for f in os.listdir(temp_dir) if f.endswith(".mp3")]
        if not mp3s:
            raise RuntimeError("Audio extraction failed: no .mp3 produced (is ffmpeg installed?)")

        mp3_path = os.path.join(temp_dir, mp3s[0])
        with open(mp3_path, "rb") as f:
            audio_bytes = f.read()

        if not audio_bytes:
            raise RuntimeError("Extracted audio file is empty")

        return audio_bytes, {"duration": duration, "title": title, "source": extractor}

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# ---------- LLM helpers ----------
def _force_json_or_raise(text: str):
    """
    Attempts to parse JSON; tries mild cleanup if model wrapped it.
    """
    if not text:
        raise ValueError("Empty LLM response")

    # Strip code fences if any
    cleaned = re.sub(r"```json\s*|```", "", text).strip()

    # If model put extra text, attempt to extract the first JSON object block
    # (simple heuristic: find first '{' and last '}' )
    if "{" in cleaned and "}" in cleaned:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        cleaned = cleaned[start:end+1]

    return json.loads(cleaned)


def extract_recipe_from_transcript_chunk(transcript_chunk: str):
    system_prompt = """You extract recipe data from cooking transcripts.
Return ONLY valid JSON matching this schema:
{
  "ingredients": [{"name": "...", "quantity": "..."}],
  "instructions": ["Step 1: ...", "Step 2: ..."]
}
Rules:
- Do NOT add explanations or markdown.
- If quantity is unknown, use "".
- Keep instructions in chronological order.
"""

    user_prompt = f"Transcript:\n{transcript_chunk}\n\nReturn the JSON now."

    # Try once with strict JSON if supported, else fallback
    try:
        completion = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=1200,
            response_format={"type": "json_object"},
        )
    except Exception:
        completion = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=1200,
        )

    text = completion.choices[0].message.content.strip()
    return _force_json_or_raise(text)


def merge_recipe_parts(parts):
    """
    Merge multiple partial extractions into a single clean recipe.
    Dedup ingredients, re-number and clean steps with a final LLM pass.
    """
    merge_system = """You merge multiple partial recipe JSONs into ONE final recipe JSON.
Return ONLY valid JSON with schema:
{
  "ingredients": [{"name": "...", "quantity": "..."}],
  "instructions": ["Step 1: ...", "Step 2: ..."]
}
Rules:
- Deduplicate ingredients (case-insensitive).
- If the same ingredient appears with different quantities, keep the most specific quantity.
- Combine instructions, remove duplicates, ensure chronological order, and ensure steps are detailed and actionable.
- Output ONLY JSON.
"""

    merge_user = json.dumps({"parts": parts}, ensure_ascii=False)

    try:
        completion = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": merge_system},
                {"role": "user", "content": merge_user},
            ],
            temperature=0.1,
            max_tokens=1800,
            response_format={"type": "json_object"},
        )
    except Exception:
        completion = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": merge_system},
                {"role": "user", "content": merge_user},
            ],
            temperature=0.1,
            max_tokens=1800,
        )

    text = completion.choices[0].message.content.strip()
    return _force_json_or_raise(text)


def chunk_text(text: str, max_chars: int = 6000):
    text = text.strip()
    if len(text) <= max_chars:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        # try to break on sentence boundary
        cut = text.rfind(".", start, end)
        if cut == -1 or cut < start + int(max_chars * 0.6):
            cut = end
        chunks.append(text[start:cut].strip())
        start = cut
    return [c for c in chunks if c]

# ---------- The endpoint ----------
@app.route("/extract-recipe-from-video", methods=["POST"])
def extract_recipe_from_video():
    """
    Accepts: {"videoUrl":"..."}
    Returns: {"ingredients":[...], "instructions":[...], "transcript":"...", "meta": {...}}
    """
    try:
        data = request.get_json(silent=True) or {}
        video_url = data.get("videoUrl")

        ok, err = validate_video_url(video_url)
        if not ok:
            return jsonify({"error": err}), 400

        print(f"🎥 Processing video URL: {video_url}")

        # 1) Download + extract audio
        t0 = time.time()
        try:
            audio_bytes, meta = download_audio_mp3(video_url)
        except ValueError as ve:
            return jsonify({"error": str(ve)}), 413
        except yt_dlp.utils.DownloadError as de:
            # very common for IG/TikTok without cookies
            return jsonify({
                "error": "Failed to download/extract audio from video URL",
                "details": str(de),
                "hint": "TikTok/Instagram often require cookies/login. Set YTDLP_COOKIES_FILE on the server."
            }), 400
        except Exception as e:
            return jsonify({
                "error": "Audio extraction failed",
                "user_message": "We couldn't get the audio from this video. Please try another link or add the recipe manually.",
                "details": str(e),
            }), 500

        print(f"✅ Audio extracted: {len(audio_bytes)} bytes in {time.time()-t0:.2f}s")

        # 2) Whisper transcription
        print("🎤 Transcribing audio...")
        audio_file_obj = io.BytesIO(audio_bytes)
        audio_file_obj.name = "audio.mp3"

        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file_obj,
            response_format="text",
        )
        transcript_text = transcript.strip() if isinstance(transcript, str) else str(transcript).strip()

        if not transcript_text:
            return jsonify({
                "error": "Failed to transcribe video",
                "user_message": "We couldn't understand the audio from this video. Try another link or add the recipe manually.",
                "message": "No transcript generated from video audio",
            }), 500

        print(f"✅ Transcript generated: {len(transcript_text)} characters")

        # 3) LLM extraction with chunking + merge
        print("🍳 Extracting recipe information...")
        chunks = chunk_text(transcript_text, max_chars=6000)

        parts = []
        for idx, ch in enumerate(chunks, start=1):
            try:
                part = extract_recipe_from_transcript_chunk(ch)
                parts.append(part)
            except Exception as e:
                return jsonify({
                    "error": "Failed to extract recipe from transcript chunk",
                    "user_message": "We couldn't extract a recipe from this link. Please try another or add the recipe manually.",
                    "chunk": idx,
                    "details": str(e),
                    "transcript": transcript_text
                }), 500

        final_recipe = merge_recipe_parts(parts) if len(parts) > 1 else parts[0]

        ingredients = final_recipe.get("ingredients", []) or []
        instructions = final_recipe.get("instructions", []) or []

        # Normalize types
        if ingredients and isinstance(ingredients[0], str):
            ingredients = [{"name": ing, "quantity": ""} for ing in ingredients]
        if not isinstance(instructions, list):
            instructions = [str(instructions)]

        return jsonify({
            "ingredients": ingredients,
            "instructions": instructions,
            "transcript": transcript_text,
            "meta": meta,
            "message": f"Successfully extracted recipe with {len(ingredients)} ingredients and {len(instructions)} instructions"
        })

    except Exception as e:
        print(f"💥 Critical error in extract_recipe_from_video: {str(e)}")
        return jsonify({
                "error": "Unexpected server error",
                "user_message": DEFAULT_500_USER_MESSAGE,
                "details": str(e),
            }), 500

#####video only code ends here#####


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

    ## video and webpage code starts here#####

    

VIDEO_DOMAINS = {"youtube.com", "www.youtube.com", "youtu.be", "tiktok.com", "www.tiktok.com", "instagram.com", "www.instagram.com"}

def is_video_url(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return any(host == d or host.endswith("." + d) for d in VIDEO_DOMAINS)

def is_youtube_url(url: str) -> bool:
    """Check if URL is a YouTube video/shorts URL."""
    if not url:
        return False
    host = (urlparse(url).hostname or "").lower()
    return any(host == d or host.endswith("." + d) for d in {"youtube.com", "www.youtube.com", "youtu.be"})


def determine_source_type(url: str) -> str:
    """
    Determines the source type based on URL.
    Returns: "TikTok", "YouTube", "Instagram", "Photos", or "Manual"
    """
    if not url:
        return "Manual"
    
    host = (urlparse(url).hostname or "").lower()
    
    if "tiktok.com" in host or "tiktok" in host:
        return "TikTok"
    elif "youtube.com" in host or "youtu.be" in host:
        return "YouTube"
    elif "instagram.com" in host or "instagram" in host:
        return "Instagram"
    elif any(ext in url.lower() for ext in [".jpg", ".jpeg", ".png", ".gif", ".webp"]):
        return "Photos"
    else:
        return "Manual"


def extract_recipe_tags(recipe: dict) -> list[str]:
    """
    Extracts tags from recipe data.
    Returns list of tags: ["High Protein", "Vegetarian", "Vegan", "Quick", "Easy", "Medium", "Hard"]
    """
    tags = []
    
    if not recipe:
        return tags
    
    # Get recipe data
    ingredients = recipe.get("ingredients", [])
    prep_time = recipe.get("prep_time", "")
    cook_time = recipe.get("cook_time", "")
    total_time = recipe.get("total_time", "")
    nutrition = recipe.get("nutrition", {})
    
    # Extract ingredient names for analysis
    ingredient_names = []
    for ing in ingredients:
        if isinstance(ing, dict):
            ingredient_names.append(ing.get("name", "").lower())
        elif isinstance(ing, str):
            ingredient_names.append(ing.lower())
    
    ingredient_text = " ".join(ingredient_names)
    
    # Check for Vegetarian (no meat, fish, poultry)
    meat_keywords = ["chicken", "beef", "pork", "lamb", "turkey", "duck", "fish", "salmon", "tuna", "shrimp", "crab", "lobster", "meat", "bacon", "sausage", "ham", "steak"]
    has_meat = any(keyword in ingredient_text for keyword in meat_keywords)
    if not has_meat and ingredient_text:
        tags.append("Vegetarian")
    
    # Check for Vegan (no animal products)
    animal_keywords = meat_keywords + ["milk", "cheese", "butter", "cream", "yogurt", "egg", "honey", "gelatin", "whey", "casein"]
    has_animal_products = any(keyword in ingredient_text for keyword in animal_keywords)
    if not has_animal_products and ingredient_text:
        tags.append("Vegan")
    
    # Check for High Protein (from nutrition data or ingredient analysis)
    protein_value = None
    if isinstance(nutrition, dict):
        # Try different possible keys
        protein_value = nutrition.get("protein") or nutrition.get("proteinContent") or nutrition.get("protein_g")
        if isinstance(protein_value, str):
            # Extract number from string like "25g" or "25 g"
            match = re.search(r'(\d+\.?\d*)', protein_value)
            if match:
                protein_value = float(match.group(1))
    
    # High protein threshold: >20g per serving (rough estimate)
    high_protein_ingredients = ["chicken", "beef", "turkey", "fish", "salmon", "tuna", "eggs", "tofu", "tempeh", "lentils", "beans", "chickpeas", "quinoa", "greek yogurt", "cottage cheese", "protein powder"]
    has_high_protein_ingredients = any(ing in ingredient_text for ing in high_protein_ingredients)
    
    if protein_value and protein_value > 20:
        tags.append("High Protein")
    elif has_high_protein_ingredients and len(ingredients) > 0:
        tags.append("High Protein")
    
    # Determine difficulty based on total time and instruction complexity
    def parse_time_to_minutes(time_str: str) -> int:
        """Convert time string to minutes. Handles formats like '30 mins', '1 hr 30 mins', 'PT30M'"""
        if not time_str:
            return 0
        
        time_str = time_str.lower()
        total_minutes = 0
        
        # Parse hours
        hour_match = re.search(r'(\d+)\s*hr', time_str)
        if hour_match:
            total_minutes += int(hour_match.group(1)) * 60
        
        # Parse minutes
        minute_match = re.search(r'(\d+)\s*min', time_str)
        if minute_match:
            total_minutes += int(minute_match.group(1))
        
        # Parse ISO 8601 format (PT30M, PT1H30M)
        if time_str.startswith("pt"):
            h_match = re.search(r'(\d+)h', time_str)
            m_match = re.search(r'(\d+)m', time_str)
            if h_match:
                total_minutes += int(h_match.group(1)) * 60
            if m_match:
                total_minutes += int(m_match.group(1))
        
        return total_minutes
    
    # Use total_time if available, otherwise sum prep + cook
    time_minutes = 0
    if total_time:
        time_minutes = parse_time_to_minutes(total_time)
    else:
        prep_min = parse_time_to_minutes(prep_time) if prep_time else 0
        cook_min = parse_time_to_minutes(cook_time) if cook_time else 0
        time_minutes = prep_min + cook_min
    
    instructions = recipe.get("instructions", [])
    num_steps = len(instructions) if isinstance(instructions, list) else 0
    
    # Difficulty classification
    if time_minutes == 0 and num_steps == 0:
        pass  # Can't determine
    elif time_minutes <= 30 and num_steps <= 5:
        tags.append("Quick")
        tags.append("Easy")
    elif time_minutes <= 60 and num_steps <= 8:
        tags.append("Easy")
    elif time_minutes <= 90 and num_steps <= 12:
        tags.append("Medium")
    elif time_minutes > 90 or num_steps > 12:
        tags.append("Hard")
    else:
        tags.append("Medium")  # Default
    
    return tags

def fetch_html(url: str) -> str:
    headers = {
        "User-Agent": "Mozilla/5.0 (RecipeBot/1.0)"
    }
    r = requests.get(url, headers=headers, timeout=20)
    r.raise_for_status()
    return r.text

def extract_og_image(soup: BeautifulSoup) -> str | None:
    og = soup.find("meta", property="og:image")
    if og and og.get("content"):
        return og["content"].strip()
    tw = soup.find("meta", attrs={"name": "twitter:image"})
    if tw and tw.get("content"):
        return tw["content"].strip()
    return None

def extract_jsonld_recipes(html: str):
    soup = BeautifulSoup(html, "lxml")
    scripts = soup.find_all("script", type="application/ld+json")
    recipes = []

    for s in scripts:
        try:
            data = json.loads(s.string or "")
        except Exception:
            continue

        # JSON-LD can be dict, list, or nested graph
        candidates = []
        if isinstance(data, list):
            candidates = data
        elif isinstance(data, dict):
            if "@graph" in data and isinstance(data["@graph"], list):
                candidates = data["@graph"]
            else:
                candidates = [data]

        for item in candidates:
            if not isinstance(item, dict):
                continue
            t = item.get("@type")
            if isinstance(t, list):
                is_recipe = any(x.lower() == "recipe" for x in t if isinstance(x, str))
            else:
                is_recipe = isinstance(t, str) and t.lower() == "recipe"
            if is_recipe:
                recipes.append(item)

    return recipes, soup

def _flatten_instructions(recipe_instructions):
    """
    Handles JSON-LD recipeInstructions that may be:
    - string
    - list of strings
    - list of HowToStep dicts
    - list of HowToSection dicts with itemListElement (steps)
    """
    steps = []

    def add_step(text: str):
        text = (text or "").strip()
        if text:
            steps.append(text)

    def handle_node(node):
        if node is None:
            return

        # Plain string
        if isinstance(node, str):
            add_step(node)
            return

        # List of nodes
        if isinstance(node, list):
            for item in node:
                handle_node(item)
            return

        # Dict node (HowToStep / HowToSection / etc.)
        if isinstance(node, dict):
            node_type = node.get("@type") or node.get("type")

            # HowToSection: keep section heading + recurse into itemListElement
            if isinstance(node_type, str) and node_type.lower() == "howtosection":
                heading = node.get("name") or node.get("headline") or ""
                if heading:
                    add_step(f"{heading.strip()}")
                handle_node(node.get("itemListElement") or node.get("steps"))
                return

            # HowToStep: take text
            if isinstance(node_type, str) and node_type.lower() == "howtostep":
                add_step(node.get("text") or node.get("name"))
                return

            # Generic fallback: try common keys
            add_step(node.get("text") or node.get("name"))
            # and recurse if there is nested list
            handle_node(node.get("itemListElement") or node.get("steps"))
            return

    handle_node(recipe_instructions)

    # Deduplicate while preserving order
    seen = set()
    out = []
    for s in steps:
        key = re.sub(r"\s+", " ", s.strip()).lower()
        if key and key not in seen:
            seen.add(key)
            out.append(s)
    return out


def normalize_duration(duration_str: str) -> str:
    """
    Converts ISO 8601 duration format (PT30M, PT1H30M, PT578M) to human-readable format.
    Examples:
        PT30M -> "30 mins"
        PT1H30M -> "1 hr 30 mins"
        PT578M -> "9 hrs 38 mins"
        PT2H -> "2 hrs"
    """
    if not duration_str or not isinstance(duration_str, str):
        return ""
    
    duration_str = duration_str.strip().upper()
    
    # If already human-readable (contains "hr", "min", "mins", etc.), return as-is
    if any(word in duration_str.lower() for word in ["hr", "hour", "min", "minute", "sec", "second"]):
        return duration_str
    
    # Parse ISO 8601 duration format (PT30M, PT1H30M, etc.)
    if not duration_str.startswith("PT"):
        return duration_str  # Not ISO 8601 format, return as-is
    
    # Extract hours, minutes, seconds
    hours = 0
    minutes = 0
    seconds = 0
    
    # Match hours (H)
    hour_match = re.search(r'(\d+)H', duration_str)
    if hour_match:
        hours = int(hour_match.group(1))
    
    # Match minutes (M)
    minute_match = re.search(r'(\d+)M', duration_str)
    if minute_match:
        minutes = int(minute_match.group(1))
    
    # Match seconds (S)
    second_match = re.search(r'(\d+)S', duration_str)
    if second_match:
        seconds = int(second_match.group(1))
        # Convert seconds to minutes if >= 60
        if seconds >= 60:
            minutes += seconds // 60
            seconds = seconds % 60
    
    # Build human-readable string
    parts = []
    if hours > 0:
        parts.append(f"{hours} hr{'s' if hours > 1 else ''}")
    if minutes > 0:
        parts.append(f"{minutes} min{'s' if minutes > 1 else ''}")
    if seconds > 0 and hours == 0 and minutes == 0:  # Only show seconds if no hours/minutes
        parts.append(f"{seconds} sec{'s' if seconds > 1 else ''}")
    
    return " ".join(parts) if parts else ""


def normalize_recipe_from_jsonld(recipe_obj: dict, soup: BeautifulSoup):
    name = recipe_obj.get("name") or ""
    image = recipe_obj.get("image")
    if isinstance(image, list) and image:
        image = image[0]
    if isinstance(image, dict):
        image = image.get("url")

    if not image:
        image = extract_og_image(soup)

    ingredients = recipe_obj.get("recipeIngredient") or []
    norm_ingredients = []
    for ing in ingredients:
        if isinstance(ing, str):
            norm_ingredients.append({"name": ing, "quantity": ""})

    # instructions can be list of strings or HowToStep objects
    # --- ✅ instructions (FIXED) ---
    instructions_raw = recipe_obj.get("recipeInstructions") or []
    norm_steps = _flatten_instructions(instructions_raw)

    # Normalize servings (can be string, number, or array)
    servings_raw = recipe_obj.get("recipeYield") or ""
    if isinstance(servings_raw, list):
        # If array, take the most descriptive one (usually the last)
        servings = str(servings_raw[-1]) if servings_raw else ""
    elif isinstance(servings_raw, (int, float)):
        servings = str(servings_raw)
    else:
        servings = str(servings_raw) if servings_raw else ""
    
    # Normalize time durations from ISO 8601 format to human-readable
    prep_raw = recipe_obj.get("prepTime") or ""
    prep = normalize_duration(prep_raw) if prep_raw else ""
    
    cook_raw = recipe_obj.get("cookTime") or ""
    cook = normalize_duration(cook_raw) if cook_raw else ""
    
    total_raw = recipe_obj.get("totalTime") or ""
    total = normalize_duration(total_raw) if total_raw else ""

    nutrition = recipe_obj.get("nutrition") or {}
    norm_nutrition = {}
    if isinstance(nutrition, dict):
        for k, v in nutrition.items():
            if isinstance(v, (str, int, float)):
                norm_nutrition[k] = str(v)

    # Cuisine: recipeCuisine can be string or list in JSON-LD
    cuisine_raw = recipe_obj.get("recipeCuisine") or ""
    if isinstance(cuisine_raw, list) and cuisine_raw:
        cuisine = ", ".join(str(x).strip() for x in cuisine_raw if x)
    else:
        cuisine = str(cuisine_raw).strip() if cuisine_raw else ""

    # Meal type: recipeCategory often has "Breakfast", "Lunch", etc.
    meal_type_raw = recipe_obj.get("recipeCategory") or ""
    if isinstance(meal_type_raw, list) and meal_type_raw:
        meal_type_raw = meal_type_raw[0] if meal_type_raw else ""
    meal_type_raw = str(meal_type_raw).strip() if meal_type_raw else ""

    return {
        "name": name,
        "ingredients": norm_ingredients,
        "instructions": norm_steps,
        "servings": servings,
        "prep_time": prep,
        "cook_time": cook,
        "total_time": total,
        "nutrition": norm_nutrition,
        "cuisine": cuisine,
        "meal_type": meal_type_raw,
    }, image
def clean_page_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text("\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()[:40000]  # cap to avoid huge prompts

def extract_recipe_from_webpage_llm(page_text: str):
    system = """Extract recipe data from webpage text.
Return ONLY valid JSON:
{
  "name": "",
  "ingredients": [{"name":"", "quantity":""}],
  "instructions": ["Step 1 ...", "..."],
  "servings": "",
  "prep_time": "",
  "cook_time": "",
  "total_time": "",
  "notes": [],
  "meal_type": "",
  "cuisine": "",
  "nutrition": {
    "calories": "",
    "protein_g": "",
    "carbs_g": "",
    "fat_g": ""
  }
}
Rules:
- Don't hallucinate for the structure. If fields like servings or times are unknown, use "" or [].
- meal_type: exactly one of Breakfast, Lunch, Dinner, Snack (infer from context).
- Infer cuisine from recipe name, ingredients, or context when evident (e.g. Italian, Mexican, Indian, American); otherwise use "".
- You MUST provide a best-effort numeric estimate (as strings) for nutrition macros PER SERVING: calories, protein_g, carbs_g, fat_g. Use your nutrition knowledge of typical ingredients/quantities to approximate. Only leave a macro field \"\" if there is literally no information about ingredients.
- Return JSON only."""
    user = f"Webpage text:\n{page_text}"

    completion = client.chat.completions.create(
        model=os.getenv("RECIPE_LLM_MODEL", "gpt-4o-mini"),
        messages=[{"role":"system","content":system},{"role":"user","content":user}],
        temperature=0.2,
        max_tokens=1800,
        response_format={"type":"json_object"},
        timeout=RECIPE_LLM_TIMEOUT,
    )
    return json.loads(completion.choices[0].message.content)


# Max images per request for recipe extraction (avoids token/rate limits)
MAX_EXTRACT_RECIPE_IMAGES = int(os.getenv("MAX_EXTRACT_RECIPE_IMAGES", "10"))

def _get_image_data_urls_from_extract_request():
    """Get one or more images as data URLs. Multipart: field 'image' (single or multiple files). JSON: imageBase64/image or images[]. Returns (list of data_urls, None) or (None, error_response_tuple)."""
    format_to_mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "gif": "image/gif", "webp": "image/webp"}

    def mime_for_filename(filename):
        f = (filename or "").lower()
        if f.endswith((".jpg", ".jpeg")): return "image/jpeg"
        if f.endswith(".png"): return "image/png"
        if f.endswith(".gif"): return "image/gif"
        if f.endswith(".webp"): return "image/webp"
        return "image/jpeg"

    # Multipart: multiple files under 'image' (e.g. <input name="image" multiple> or multiple fields)
    if "image" in request.files:
        urls = []
        for uploaded_file in request.files.getlist("image"):
            if not uploaded_file or not uploaded_file.filename:
                continue
            image_bytes = uploaded_file.read()
            if not image_bytes:
                continue
            content_type = mime_for_filename(uploaded_file.filename)
            b64 = base64.b64encode(image_bytes).decode("utf-8")
            urls.append(f"data:{content_type};base64,{b64}")
        if urls:
            if len(urls) > MAX_EXTRACT_RECIPE_IMAGES:
                return None, (jsonify({"error": f"Too many images; max {MAX_EXTRACT_RECIPE_IMAGES}"}), 400)
            return urls, None
        # single file (legacy): some clients send one file without getlist
        single = request.files["image"]
        if single and single.filename:
            single.seek(0)
            image_bytes = single.read()
            if image_bytes:
                content_type = mime_for_filename(single.filename)
                b64 = base64.b64encode(image_bytes).decode("utf-8")
                return [f"data:{content_type};base64,{b64}"], None
        return None, (jsonify({"error": "Uploaded image file is empty"}), 400)

    # JSON: images array or single imageBase64/image
    if request.is_json:
        data = request.get_json(silent=True) or {}
        images_list = data.get("images")
        if isinstance(images_list, list) and images_list:
            urls = []
            for i, item in enumerate(images_list):
                if i >= MAX_EXTRACT_RECIPE_IMAGES:
                    break
                if isinstance(item, dict):
                    b64 = item.get("imageBase64") or item.get("image")
                    fmt = (item.get("imageFormat") or "jpg").lower()
                elif isinstance(item, str):
                    b64, fmt = item, "jpg"
                else:
                    continue
                if not b64:
                    continue
                if "," in str(b64):
                    b64 = str(b64).split(",")[-1]
                content_type = format_to_mime.get(fmt, "image/jpeg")
                urls.append(f"data:{content_type};base64,{b64}")
            if urls:
                return urls, None
        # Single image (legacy)
        image_base64 = data.get("imageBase64") or data.get("image")
        image_format = (data.get("imageFormat") or "jpg").lower()
        if image_base64:
            if "," in str(image_base64):
                image_base64 = str(image_base64).split(",")[-1]
            content_type = format_to_mime.get(image_format, "image/jpeg")
            return [f"data:{content_type};base64,{image_base64}"], None
    return None, None


def extract_recipe_from_image_llm(image_data_url: str):
    """Extract recipe from a single image. Returns same structure as extract_recipe_from_webpage_llm."""
    return extract_recipe_from_images_llm([image_data_url])


def extract_recipe_from_images_llm(image_data_urls: list):
    """Extract one combined recipe from one or more images (e.g. multi-page recipe). Uses vision LLM."""
    if not image_data_urls:
        raise ValueError("At least one image is required")
    system = """Extract recipe data from the image(s). If multiple images are provided (e.g. multiple pages), combine them into ONE recipe.
Return ONLY valid JSON:
{
  "name": "",
  "ingredients": [{"name":"", "quantity":""}],
  "instructions": ["Step 1 ...", "..."],
  "servings": "",
  "prep_time": "",
  "cook_time": "",
  "total_time": "",
  "notes": [],
  "meal_type": "",
  "cuisine": "",
  "nutrition": {
    "calories": "",
    "protein_g": "",
    "carbs_g": "",
    "fat_g": ""
  }
}
Rules:
- Don't hallucinate. If unknown, use "" or [].
- meal_type: exactly one of Breakfast, Lunch, Dinner, Snack (infer from context).
- Infer cuisine from recipe name, ingredients, or visible context when evident (e.g. Italian, Mexican, Indian); otherwise use "".
- You MUST provide a best-effort numeric estimate (as strings) for nutrition macros PER SERVING: calories, protein_g, carbs_g, fat_g. Use your nutrition knowledge of typical ingredients/quantities to approximate from what you see. Only leave a macro field \"\" if there is literally no information about ingredients.
- Return JSON only. Read all text visible across the images. Merge ingredients and instructions from all pages into one recipe."""
    content = [{"type": "text", "text": "Extract the recipe from these image(s) and return the JSON. If there are multiple images, treat them as one multi-page recipe and merge into a single recipe."}]
    for url in image_data_urls:
        content.append({"type": "image_url", "image_url": {"url": url}})
    completion = client.chat.completions.create(
        model=os.getenv("RECIPE_LLM_MODEL", "gpt-4o-mini"),
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ],
        temperature=0.2,
        max_tokens=1800,
        response_format={"type": "json_object"},
        timeout=RECIPE_LLM_TIMEOUT,
    )
    return json.loads(completion.choices[0].message.content)


# Allowed values for /extract-recipe response attributes
MEAL_TYPES = ("Breakfast", "Lunch", "Dinner", "Snack")
CUISINES = ("Italian", "Mexican", "American", "Asian", "Mediterranean", "Indian", "Chinese", "Japanese", "Thai", "Middle Eastern")
DIET_FLAGS = ("High Protein", "High Fiber", "Low Carb", "Keto", "Vegetarian", "Vegan", "Gluten Free", "Balanced")


def normalize_meal_type(raw: str) -> str:
    """Map free-text meal type to one of Breakfast, Lunch, Dinner, Snack."""
    if not raw or not isinstance(raw, str):
        return "Lunch / Dinner"  # default
    s = raw.strip().lower()
    if not s:
        return "Lunch / Dinner"
    if s in ("breakfast", "brunch") or "breakfast" in s or "brunch" in s:
        return "Breakfast"
    if s in ("lunch", "brunch") or "lunch" in s:
        return "Lunch"
    if s in ("dinner", "supper", "main") or "dinner" in s or "supper" in s:
        return "Dinner"
    if s in ("snack", "appetizer", "appetiser", "side", "dessert") or "snack" in s:
        return "Snack"
    return "Lunch / Dinner"


def normalize_cuisine(raw: str) -> str:
    """Map free-text cuisine to one of the allowed CUISINES."""
    if not raw or not isinstance(raw, str):
        return "American"
    s = raw.strip().lower()
    if not s:
        return "American"
    # Direct and fuzzy matches
    mapping = [
        ("italian", "Italian"), ("mexican", "Mexican"), ("american", "American"),
        ("asian", "Asian"), ("mediterranean", "Mediterranean"), ("indian", "Indian"),
        ("chinese", "Chinese"), ("japanese", "Japanese"), ("japenese", "Japanese"),
        ("thai", "Thai"), ("middle eastern", "Middle Eastern"), ("middle east", "Middle Eastern"),
        ("korean", "Asian"), ("vietnamese", "Asian"), ("greek", "Mediterranean"),
        ("spanish", "Mediterranean"), ("french", "Mediterranean"),
    ]
    for key, value in mapping:
        if key in s or s in key:
            return value
    # Fallback: if single word might be cuisine, try match
    for c in CUISINES:
        if c.lower() in s or s in c.lower():
            return c
    return "American"


def extract_recipe_diet_flags(recipe: dict) -> list[str]:
    """Return diet flags from fixed set: High Protein, High Fiber, Low Carb, Keto, Vegetarian, Vegan, Gluten Free, Balanced."""
    flags = []
    if not recipe:
        return flags
    ingredients = recipe.get("ingredients", [])
    nutrition = recipe.get("nutrition") or {}
    ingredient_names = []
    for ing in ingredients:
        if isinstance(ing, dict):
            ingredient_names.append(ing.get("name", "").lower())
        elif isinstance(ing, str):
            ingredient_names.append(ing.lower())
    ingredient_text = " ".join(ingredient_names)

    # Vegetarian (no meat/fish/poultry)
    meat_keywords = ["chicken", "beef", "pork", "lamb", "turkey", "duck", "fish", "salmon", "tuna", "shrimp", "crab", "lobster", "meat", "bacon", "sausage", "ham", "steak"]
    if not any(k in ingredient_text for k in meat_keywords) and ingredient_text:
        flags.append("Vegetarian")

    # Vegan (no animal products)
    animal_keywords = meat_keywords + ["milk", "cheese", "butter", "cream", "yogurt", "egg", "honey", "gelatin", "whey", "casein"]
    if not any(k in ingredient_text for k in animal_keywords) and ingredient_text:
        flags.append("Vegan")

    # High Protein
    protein_val = nutrition.get("protein") or nutrition.get("proteinContent") or nutrition.get("protein_g")
    if isinstance(protein_val, str):
        match = re.search(r"(\d+\.?\d*)", protein_val)
        protein_val = float(match.group(1)) if match else None
    high_protein_ings = ["chicken", "beef", "turkey", "fish", "salmon", "tuna", "eggs", "tofu", "tempeh", "lentils", "beans", "chickpeas", "quinoa", "greek yogurt", "cottage cheese", "protein powder"]
    if (protein_val and protein_val > 20) or any(ing in ingredient_text for ing in high_protein_ings):
        flags.append("High Protein")

    # High Fiber
    fiber_ings = ["oats", "lentils", "beans", "chickpeas", "quinoa", "barley", "broccoli", "avocado", "chia", "flax", "whole grain", "brown rice"]
    if any(ing in ingredient_text for ing in fiber_ings):
        flags.append("High Fiber")

    # Low Carb / Keto (heuristic: few grains/sugar)
    carb_heavy = ["flour", "pasta", "rice", "bread", "sugar", "potato", "oat", "quinoa", "corn", "beans"]
    carb_count = sum(1 for c in carb_heavy if c in ingredient_text)
    if carb_count <= 1 and ingredient_text:
        flags.append("Low Carb")
    if carb_count <= 1 and any(f in ingredient_text for f in ["avocado", "cheese", "cream", "olive oil", "coconut"]):
        flags.append("Keto")

    # Gluten Free (no wheat, barley, rye)
    if not any(g in ingredient_text for g in ["wheat", "flour", "barley", "rye", "bread", "pasta", "couscous"]):
        flags.append("Gluten Free")

    # Balanced (default if nothing else or general)
    if not flags or len(flags) >= 2:
        flags.append("Balanced")

    # Return only from allowed set, deduplicated, order preserved
    return list(dict.fromkeys(f for f in flags if f in DIET_FLAGS))


def _short_recipe_description(recipe: dict) -> str:
    """
    Build a short, one-line summary of the recipe (max 100 chars) for /extract-recipe response.
    Uses recipe name + diet/style flags. Ends with a full stop; no ellipsis.
    """
    if not recipe:
        return ""
    name = (recipe.get("name") or "").strip()
    diet_flags = recipe.get("diet_flags") or []
    diet_words = " ".join(str(d).lower() for d in diet_flags if d).strip()
    if diet_words and name:
        desc = f"{diet_words} {name}"
    elif name:
        desc = name
    else:
        desc = diet_words or "Recipe"
    desc = re.sub(r"\s+", " ", desc).strip()
    if len(desc) > 100:
        desc = desc[:99].rstrip()
        if desc and not desc.endswith("."):
            desc = desc + "."
    elif desc and not desc.endswith("."):
        desc = desc + "."
    return desc


def _enrich_recipe_response(recipe: dict) -> None:
    """Set meal_type, cuisine (normalized), diet_flags, and description on recipe for /extract-recipe response."""
    if not recipe:
        return
    recipe["meal_type"] = normalize_meal_type(recipe.get("meal_type"))
    recipe["cuisine"] = normalize_cuisine(recipe.get("cuisine"))
    recipe["diet_flags"] = extract_recipe_diet_flags(recipe)
    recipe["description"] = _short_recipe_description(recipe)


def _guess_difficulty_from_tags(tags: list[str]) -> str:
    tags = [t.lower() for t in (tags or [])]
    if "easy" in tags:
        return "Easy"
    if "medium" in tags:
        return "Medium"
    if "hard" in tags:
        return "Hard"
    return "Easy"


def _time_str_to_minutes_for_cook_time(time_str: str) -> int:
    """
    Convert a human-readable time string into minutes (numeric only for cookTime).
    Handles formats like '20 mins', '1 hr 30 mins', '45 min', '2 hours', 'PT30M'.
    """
    if not time_str:
        return 0
    s = str(time_str).strip().lower()
    if not s:
        return 0

    total = 0

    # Hours
    m = re.search(r"(\d+)\s*(h|hr|hrs|hour|hours)", s)
    if m:
        total += int(m.group(1)) * 60

    # Minutes
    m = re.search(r"(\d+)\s*(m|min|mins|minute|minutes)", s)
    if m:
        total += int(m.group(1))

    # ISO 8601 PTxxHxxM
    if s.startswith("pt"):
        h = re.search(r"(\d+)h", s)
        m2 = re.search(r"(\d+)m", s)
        if h:
            total += int(h.group(1)) * 60
        if m2:
            total += int(m2.group(1))

    # Fallback: if still zero, try to parse first integer as minutes
    if total == 0:
        m = re.search(r"(\d+)", s)
        if m:
            total = int(m.group(1))

    return total


@app.route("/bulk-import-recipes", methods=["POST"])
def bulk_import_recipes():
    """
    Bulk operations for recipes using the recipe-log collection.

    - Default behavior (no `action`): accept a CSV file (form field `file`)
      with a `recipe_url` column. For each row, call /extract-recipe with that
      URL and save the result into the Firestore `recipe-log` collection.

    - Delete behavior (`action=delete` + `createdDate=YYYY-MM-DD`): delete all
      recipe-log documents whose createdAt falls on that date (UTC).
    """
    action = (request.form.get("action") or "").strip().lower()
    created_date_str = (request.form.get("createdDate") or "").strip()

    db = init_firestore()
    collection = db.collection("recipe-log")

    # If action=delete, delete all entries for that created date and return.
    if action == "delete":
        if not created_date_str:
            return jsonify({"error": "createdDate is required when action=delete (format: YYYY-MM-DD)"}), 400
        try:
            day = datetime.fromisoformat(created_date_str).date()
        except ValueError:
            return jsonify({"error": "createdDate must be in YYYY-MM-DD format"}), 400

        start = datetime(day.year, day.month, day.day)
        end = start + timedelta(days=1)

        docs = collection.where("createdAt", ">=", start).where("createdAt", "<", end).stream()
        deleted = 0
        for doc in docs:
            doc.reference.delete()  
            deleted += 1

        return jsonify({
            "action": "delete",
            "createdDate": created_date_str,
            "deleted": deleted,
        })

    # Existing behavior: CSV import
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "CSV file is required (field name: 'file')"}), 400

    try:
        content = file.read().decode("utf-8")
    except Exception:
        return jsonify({"error": "Failed to read CSV file; must be UTF-8 text"}), 400

    reader = csv.DictReader(io.StringIO(content))
    if "recipe_url" not in reader.fieldnames:
        return jsonify({"error": "CSV must contain a 'recipe_url' column"}), 400

    # Base URL for this service (can point to Render, etc.)
    base_url = os.getenv("EXTRACT_RECIPE_BASE_URL", "http://localhost:5002")

    imported = 0
    failed = []

    for idx, row in enumerate(reader, start=1):
        url = (row.get("recipe_url") or "").strip()
        if not url:
            failed.append({"row": idx, "reason": "Empty recipe_url"})
            continue

        try:
            resp = requests.post(
                f"{base_url}/extract-recipe",
                json={"url": url, "mode": "auto"},
                timeout=120,
            )
        except Exception as e:
            failed.append({"row": idx, "url": url, "reason": f"Request error: {e}"})
            continue

        try:
            data = resp.json()
        except Exception:
            failed.append({"row": idx, "url": url, "reason": "Non-JSON response from /extract-recipe"})
            continue

        if not resp.ok or not data.get("recipe"):
            failed.append({
                "row": idx,
                "url": url,
                "reason": data.get("user_message") or data.get("error") or "extract-recipe failed",
            })
            continue

        recipe = data["recipe"]
        source = data.get("source") or {}
        tags = data.get("tags") or []

        # Ensure enrichment (just in case /extract-recipe behavior changes)
        _enrich_recipe_response(recipe)

        # Map to Firestore recipe-log schema (as per screenshot)
        recipe_id = str(uuid.uuid4())
        title = (recipe.get("name") or "").strip()
        cook_time_str = recipe.get("cook_time") or recipe.get("total_time", "")
        cook_time_minutes = _time_str_to_minutes_for_cook_time(cook_time_str)

        # Short description (max 100 chars, ends with full stop; already set by _enrich_recipe_response)
        desc = (recipe.get("description") or title or "").strip()
        if len(desc) > 100:
            desc = desc[:99].rstrip()
            if desc and not desc.endswith("."):
                desc = desc + "."
        elif desc and not desc.endswith("."):
            desc = desc + "."

        # Steps: Firestore format [{ instruction, order, duration }, ...] (plain text, no "1.", "2." in instruction)
        raw_instructions = recipe.get("instructions") or []
        steps = []
        for i, raw in enumerate(raw_instructions, start=1):
            text = (raw.get("instruction", raw) if isinstance(raw, dict) else str(raw)).strip()
            # Strip leading "1. ", "Step 1: ", etc.
            text = re.sub(r"^(?:\d+[.)]\s*|step\s*\d+\s*[.:]\s*)", "", text, flags=re.IGNORECASE).strip()
            steps.append({
                "instruction": text,
                "order": i,
                "duration": None,
            })

        doc = {
            "recipeId": recipe_id,
            "category": recipe.get("meal_type", "Dinner"),
            "cookTime": cook_time_minutes,
            "createdAt": datetime.utcnow(),
            "title": title,
            "description": desc,
            "difficulty": _guess_difficulty_from_tags(tags),
            "imageUrl": source.get("image") or "",
            "ingredients": [
                {
                    "name": ing.get("name", ""),
                    "amount": ing.get("quantity", ""),
                }
                for ing in (recipe.get("ingredients") or [])
                if isinstance(ing, dict)
            ],
            "isFavorite": False,
            "nutrition": recipe.get("nutrition") or {},
            "steps": steps,
            "sourceUrl": source.get("url") or url,
            "cuisine": recipe.get("cuisine", ""),
            "dietFlags": recipe.get("diet_flags", []),
            "tags": tags,
        }

        try:
            # Store under deterministic recipeId so each recipe has a stable ID
            collection.document(recipe_id).set(doc)
            imported += 1
        except Exception as e:
            failed.append({"row": idx, "url": url, "reason": f"Firestore error: {e}"})

    return jsonify({
        "imported": imported,
        "failed": failed,
    })


@app.route("/extract-recipe", methods=["POST"])
def extract_recipe():
    # Image input: multipart (field 'image', single or multiple files) or JSON (imageBase64/images array)
    image_data_urls, image_error = _get_image_data_urls_from_extract_request()
    if image_error:
        return image_error[0], image_error[1]

    if image_data_urls:
        try:
            recipe = extract_recipe_from_images_llm(image_data_urls)
            _enrich_recipe_response(recipe)
            tags = extract_recipe_tags(recipe)
            source = {
                "type": "image",
                "url": None,
                "provider": None,
                "title": recipe.get("name", "") or "Recipe from image",
                "image": None,
                "source_type": "Photos",
            }
            return jsonify({
                "source": source,
                "recipe": recipe,
                "tags": tags,
                "transcript": None,
                "extraction": {"method": "image_vision", "confidence": 0.6},
            })
        except Exception as e:
            return jsonify({
                "error": "Failed to extract recipe from image",
                "user_message": "We couldn't extract a recipe from this image. Please try another photo or add the recipe manually.",
                "details": str(e),
            }), 500

    data = request.get_json(silent=True) or {}
    url = data.get("url") or data.get("videoUrl") or data.get("recipeUrl")
    mode = (data.get("mode") or "auto").lower()

    if not url:
        return jsonify({"error": "url or image is required"}), 400

    # Use your existing SSRF validation here too
    ok, err = validate_video_url(url)  # rename this to validate_url (works for all)
    if not ok:
        return jsonify({"error": err}), 400

    # Decide type
    url_is_video = is_video_url(url)
    if mode == "video":
        url_is_video = True
    elif mode == "webpage":
        url_is_video = False

    if url_is_video:
        # Call your existing video pipeline
        # (audio -> whisper -> chunk+merge LLM)
        return extract_recipe_from_video_internal(url)

    # Webpage pipeline
    try:
        html = fetch_html(url)
        recipes, soup = extract_jsonld_recipes(html)

        image = None
        recipe = None
        method = None

        if recipes:
            recipe, image = normalize_recipe_from_jsonld(recipes[0], soup)
            method = "jsonld"
        else:
            # fallback to LLM on cleaned page text
            page_text = clean_page_text(html)
            recipe = extract_recipe_from_webpage_llm(page_text)
            image = extract_og_image(soup)
            method = "html_llm"

        # Determine source type and extract tags
        source_type = determine_source_type(url)
        _enrich_recipe_response(recipe)
        tags = extract_recipe_tags(recipe)
        
        source = {
            "type": "webpage",
            "url": url,
            "provider": (urlparse(url).hostname or ""),
            "title": recipe.get("name", "") or (soup.title.string.strip() if soup.title and soup.title.string else ""),
            "image": image,
            "source_type": source_type  # Add source type: TikTok, YouTube, Instagram, Photos, Manual
        }

        return jsonify({
            "source": source,
            "recipe": recipe,
            "tags": tags,  # Add tags: High Protein, Vegetarian, Vegan, Quick, Easy, Medium, Hard
            "transcript": None,
            "extraction": {"method": method, "confidence": 0.7 if method == "jsonld" else 0.5}
        })

    except requests.exceptions.HTTPError as e:
        status = e.response.status_code if e.response is not None else 502

        if status == 402:
            user_message = "We couldn't extract a recipe from this link. This site may be paywalled or block automated access—try another link or add the recipe manually."
        elif 400 <= status < 500:
            user_message = "We couldn't extract a recipe from this link. Please check the URL or try a different recipe page."
        else:
            user_message = "We couldn't extract a recipe from this link. Please try again later or add the recipe manually."

        return jsonify({
            "error": "WEBPAGE_FETCH_FAILED",
            "status": status,
            "user_message": user_message,
            "details": str(e),
        }), 400

    except Exception as e:
        return jsonify({
            "error": "UNEXPECTED_EXTRACT_RECIPE_ERROR",
            "user_message": "We couldn't extract a recipe from this link. Please try another link or add the recipe manually.",
            "details": str(e),
        }), 500




def _chunk_text(text: str, max_chars: int = 6000):
    text = (text or "").strip()
    if len(text) <= max_chars:
        return [text] if text else []
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        cut = text.rfind(".", start, end)
        if cut == -1 or cut < start + int(max_chars * 0.6):
            cut = end
        chunks.append(text[start:cut].strip())
        start = cut
    return [c for c in chunks if c]


def _force_json(text: str) -> dict:
    if not text:
        raise ValueError("Empty LLM response")
    cleaned = re.sub(r"```json\s*|```", "", text).strip()
    # Heuristic: pull first JSON object if extra text exists
    if "{" in cleaned and "}" in cleaned:
        cleaned = cleaned[cleaned.find("{"): cleaned.rfind("}") + 1]
    return json.loads(cleaned)


def _extract_recipe_chunk(transcript_chunk: str) -> dict:
    system_prompt = """You extract recipe data from cooking transcripts.
Return ONLY valid JSON matching:
{
  "name": "",
  "ingredients": [{"name": "...", "quantity": "..."}],
  "instructions": ["Step 1: ...", "Step 2: ..."],
  "servings": "",
  "prep_time": "",
  "cook_time": "",
  "total_time": "",
  "notes": [],
  "meal_type": "",
  "cuisine": "",
  "nutrition": {
    "calories": "",
    "protein_g": "",
    "carbs_g": "",
    "fat_g": ""
  }
}
Rules:
- Don't hallucinate for the structure. If fields like servings or times are unknown, use "" or [].
- meal_type: exactly one of Breakfast, Lunch, Dinner, Snack (infer from context).
- Infer cuisine from recipe name, ingredients, or speaker context when evident (e.g. Italian, Mexican, Indian); otherwise use "".
- Ingredients must include quantities when stated; else quantity "".
- Instructions must be actionable, chronological, and detailed.
- You MUST provide a best-effort numeric estimate (as strings) for nutrition macros PER SERVING: calories, protein_g, carbs_g, fat_g. Use your nutrition knowledge of typical ingredients/quantities and transcript context to approximate. Only leave a macro field \"\" if there is literally no information about ingredients.
- Output JSON only (no markdown, no commentary)."""

    user_prompt = f"Transcript:\n{transcript_chunk}\n\nReturn the JSON now."

    # Prefer strict JSON if supported
    try:
        completion = client.chat.completions.create(
            model=RECIPE_LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=1400,
            response_format={"type": "json_object"},
            timeout=RECIPE_LLM_TIMEOUT,
        )
    except Exception:
        completion = client.chat.completions.create(
            model=RECIPE_LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=1400,
            timeout=RECIPE_LLM_TIMEOUT,
        )

    return _force_json(completion.choices[0].message.content.strip())


def _merge_recipe_parts(parts: list[dict]) -> dict:
    system_prompt = """Merge multiple partial recipe JSONs into ONE final recipe JSON.
Return ONLY valid JSON matching:
{
  "name": "",
  "ingredients": [{"name": "...", "quantity": "..."}],
  "instructions": ["Step 1: ...", "Step 2: ..."],
  "servings": "",
  "prep_time": "",
  "cook_time": "",
  "total_time": "",
  "notes": [],
  "meal_type": "",
  "cuisine": "",
  "nutrition": {
    "calories": "",
    "protein_g": "",
    "carbs_g": "",
    "fat_g": ""
  }
}
Rules:
- Deduplicate ingredients case-insensitively; keep most specific quantity.
- Remove duplicate steps, ensure correct chronological order.
- Ensure steps are detailed and actionable.
- For meal_type: use the first non-empty from parts (one of Breakfast, Lunch, Dinner, Snack); if none, use "Dinner".
- For cuisine: use the first non-empty cuisine from the parts; if none, use "".
- Merge/average any provided nutrition macros (calories, protein_g, carbs_g, fat_g) into a single best-effort estimate PER SERVING. If some parts omit macros, use available information from other parts. Only leave a macro field \"\" if ALL parts lack enough information.
- Output JSON only."""

    user_prompt = json.dumps({"parts": parts}, ensure_ascii=False)

    try:
        completion = client.chat.completions.create(
            model=RECIPE_LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=1800,
            response_format={"type": "json_object"},
            timeout=RECIPE_LLM_TIMEOUT,
        )
    except Exception:
        completion = client.chat.completions.create(
            model=RECIPE_LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=1800,
            timeout=RECIPE_LLM_TIMEOUT,
        )

    return _force_json(completion.choices[0].message.content.strip())


def _yt_meta(video_url: str) -> dict:
    """Extract metadata without downloading. Uses cookies if available."""
    opts = {"quiet": True, "no_warnings": True, "noplaylist": True}
    
    # Add cookies if available (needed for Instagram/TikTok)
    if YTDLP_COOKIES_FILE and os.path.exists(YTDLP_COOKIES_FILE):
        opts["cookiefile"] = YTDLP_COOKIES_FILE
        print(f"🍪 Using cookies file for metadata: {YTDLP_COOKIES_FILE}")
    
    # Add proxy ONLY for YouTube URLs (not for TikTok/Instagram)
    if YT_PROXY and is_youtube_url(video_url):
        opts["proxy"] = YT_PROXY
        print(f"🌐 Using proxy for YouTube metadata: {YT_PROXY}")
    
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(video_url, download=False)
    return info or {}


def _download_audio_mp3(video_url: str):
    """
    Returns: (audio_bytes, meta_dict)
    """
    temp_dir = tempfile.mkdtemp()
    try:
        info = _yt_meta(video_url)
        duration = info.get("duration")
        title = info.get("title") or ""
        extractor = info.get("extractor_key") or info.get("extractor") or ""
        webpage_url = info.get("webpage_url") or video_url
        thumbnail = info.get("thumbnail")  # often present for YouTube/IG/TikTok

        if duration and duration > MAX_VIDEO_SECONDS:
            raise ValueError(f"Video too long ({duration}s). Max allowed is {MAX_VIDEO_SECONDS}s")

        ydl_opts = {
            # IMPORTANT: robust selector with fallbacks (handles many edge cases)
            "format": "bestaudio/best/best",
            "outtmpl": os.path.join(temp_dir, "%(id)s.%(ext)s"),
            "restrictfilenames": True,
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,

            # Helps for YouTube signature issues & format availability
            "extractor_args": {
                "youtube": {"player_client": ["android", "web"]}
            },

            # Convert to mp3
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }],
        }

        # Cookies (very helpful for TikTok/IG + some YouTube)
        cookies_used = False
        if YTDLP_COOKIES_FILE and os.path.exists(YTDLP_COOKIES_FILE):
            ydl_opts["cookiefile"] = YTDLP_COOKIES_FILE
            cookies_used = True
            print(f"🍪 Using cookies file: {YTDLP_COOKIES_FILE}")
        elif YTDLP_COOKIES_B64:
            # Create temp cookies file from base64 (for Render/cloud deployment)
            temp_cookies_path = os.path.join(temp_dir, 'cookies.txt')
            try:
                cookies_content = base64.b64decode(YTDLP_COOKIES_B64).decode('utf-8')
                with open(temp_cookies_path, 'w') as f:
                    f.write(cookies_content)
                ydl_opts["cookiefile"] = temp_cookies_path
                cookies_used = True
                print("🍪 Using cookies from environment variable (base64)")
            except Exception as e:
                print(f"⚠️ Failed to decode cookies from YTDLP_COOKIES_B64: {str(e)}")
        
        # For Instagram, add additional extractor args if cookies are available
        if "instagram.com" in video_url.lower() and cookies_used:
            ydl_opts.setdefault("extractor_args", {})["instagram"] = {
                "webpage_display": ["Desktop"]
            }
        
        # Add proxy ONLY for YouTube URLs (not for TikTok/Instagram/webpages)
        if YT_PROXY and is_youtube_url(video_url):
            ydl_opts["proxy"] = YT_PROXY
            print(f"🌐 Using proxy for YouTube: {YT_PROXY}")

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])

        mp3s = [f for f in os.listdir(temp_dir) if f.endswith(".mp3")]
        if not mp3s:
            raise RuntimeError("No .mp3 produced. Check ffmpeg installation and yt-dlp extraction.")

        mp3_path = os.path.join(temp_dir, mp3s[0])
        with open(mp3_path, "rb") as f:
            audio_bytes = f.read()

        if not audio_bytes:
            raise RuntimeError("Extracted audio is empty")

        meta = {
            "duration": duration,
            "title": title,
            "provider": (urlparse(video_url).hostname or ""),
            "extractor": extractor,
            "webpage_url": webpage_url,
            "thumbnail": thumbnail,
        }
        return audio_bytes, meta

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# When transcript is shorter than this (chars), treat as no speech and use frame+vision fallback
MIN_TRANSCRIPT_LENGTH_FOR_VIDEO = int(os.getenv("MIN_TRANSCRIPT_LENGTH_VIDEO", "80"))
# Fewer frames = faster (set NUM_VIDEO_FRAMES_RECIPE=8 for more accuracy)
NUM_VIDEO_FRAMES_FOR_RECIPE = min(int(os.getenv("NUM_VIDEO_FRAMES_RECIPE", "5")), MAX_EXTRACT_RECIPE_IMAGES)
# Timeout for OpenAI recipe/vision calls (seconds)
RECIPE_LLM_TIMEOUT = int(os.getenv("RECIPE_LLM_TIMEOUT", "120"))


def _download_video_to_file(video_url: str):
    """
    Download video (not just audio) to a temp file for frame extraction.
    Returns: (temp_dir, video_path, meta_dict). Caller must shutil.rmtree(temp_dir) when done.
    """
    temp_dir = tempfile.mkdtemp()
    try:
        info = _yt_meta(video_url)
        duration = info.get("duration")
        title = info.get("title") or ""
        extractor = info.get("extractor_key") or info.get("extractor") or ""
        webpage_url = info.get("webpage_url") or video_url
        thumbnail = info.get("thumbnail")
        if duration and duration > MAX_VIDEO_SECONDS:
            raise ValueError(f"Video too long ({duration}s). Max allowed is {MAX_VIDEO_SECONDS}s")

        ydl_opts = {
            "format": "best[height<=720]/best",
            "outtmpl": os.path.join(temp_dir, "%(id)s.%(ext)s"),
            "restrictfilenames": True,
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,
            "extractor_args": {"youtube": {"player_client": ["android", "web"]}},
        }
        if YTDLP_COOKIES_FILE and os.path.exists(YTDLP_COOKIES_FILE):
            ydl_opts["cookiefile"] = YTDLP_COOKIES_FILE
        elif YTDLP_COOKIES_B64:
            temp_cookies_path = os.path.join(temp_dir, "cookies.txt")
            try:
                cookies_content = base64.b64decode(YTDLP_COOKIES_B64).decode("utf-8")
                with open(temp_cookies_path, "w") as f:
                    f.write(cookies_content)
                ydl_opts["cookiefile"] = temp_cookies_path
            except Exception:
                pass
        if "instagram.com" in video_url.lower() and ydl_opts.get("cookiefile"):
            ydl_opts.setdefault("extractor_args", {})["instagram"] = {"webpage_display": ["Desktop"]}
        if YT_PROXY and is_youtube_url(video_url):
            ydl_opts["proxy"] = YT_PROXY

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])

        candidates = [f for f in os.listdir(temp_dir) if f.endswith((".mp4", ".webm", ".mkv", ".mov"))]
        if not candidates:
            raise RuntimeError("No video file produced by yt-dlp")
        video_path = os.path.join(temp_dir, candidates[0])
        meta = {
            "duration": duration,
            "title": title,
            "provider": (urlparse(video_url).hostname or ""),
            "extractor": extractor,
            "webpage_url": webpage_url,
            "thumbnail": thumbnail,
        }
        return temp_dir, video_path, meta
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


def _extract_one_frame(video_path: str, timestamp: float, out_path: str) -> str | None:
    """Extract a single frame at timestamp; return data URL or None. Used for parallel extraction."""
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y", "-ss", str(timestamp), "-i", video_path,
                "-vframes", "1", "-q:v", "2", out_path,
            ],
            capture_output=True,
            timeout=30,
        )
        if result.returncode != 0 or not os.path.exists(out_path):
            return None
        with open(out_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"
    except Exception:
        return None


def _extract_frame_data_urls_from_video(video_path: str, duration_sec: float | None, num_frames: int = 8) -> list:
    """
    Extract evenly spaced frames from video as JPEG data URLs (parallel ffmpeg for speed).
    """
    num_frames = min(max(1, num_frames), MAX_EXTRACT_RECIPE_IMAGES)
    out_dir = tempfile.mkdtemp()
    try:
        if duration_sec and duration_sec > 0:
            interval = duration_sec / (num_frames + 1)
            timestamps = [interval * (i + 1) for i in range(num_frames)]
        else:
            timestamps = [float(i * 10) for i in range(num_frames)]

        # Extract frames in parallel for faster response
        results = [None] * len(timestamps)
        max_workers = min(4, len(timestamps))  # cap parallelism
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    _extract_one_frame,
                    video_path,
                    t,
                    os.path.join(out_dir, f"frame_{i:02d}.jpg"),
                ): i
                for i, t in enumerate(timestamps)
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    data_url = future.result()
                    if data_url:
                        results[idx] = data_url
                except Exception:
                    pass
        return [r for r in results if r is not None]
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)


def _is_likely_non_speech(transcript: str) -> bool:
    """True if transcript looks like music/no recipe speech (e.g. '[Music]', '[Applause]', or very short)."""
    t = (transcript or "").strip()
    if len(t) < MIN_TRANSCRIPT_LENGTH_FOR_VIDEO:
        return True
    cleaned = re.sub(r"\[[\w\s]+\]", "", t, flags=re.IGNORECASE).strip()
    return len(cleaned) < MIN_TRANSCRIPT_LENGTH_FOR_VIDEO


def _extract_audio_from_video_file(video_path: str) -> bytes | None:
    """Extract audio from an already-downloaded video file using ffmpeg. Returns mp3 bytes or None on failure."""
    out_dir = tempfile.mkdtemp()
    try:
        out_path = os.path.join(out_dir, "audio.mp3")
        result = subprocess.run(
            [
                "ffmpeg", "-y", "-i", video_path,
                "-vn", "-acodec", "libmp3lame", "-q:a", "2", out_path,
            ],
            capture_output=True,
            timeout=120,
        )
        if result.returncode != 0 or not os.path.exists(out_path):
            return None
        with open(out_path, "rb") as f:
            return f.read()
    except Exception:
        return None
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)


def _run_frame_vision_fallback(video_url: str, meta: dict) -> tuple[dict | None, dict | None, str | None]:
    """
    Fallback when audio transcript doesn't yield a recipe: download video, extract frames, run vision.
    Returns (recipe_dict, source_dict, extraction_method) or (None, None, None) on failure.
    meta can be from _download_audio_mp3 (we have duration/title/etc.) or from _download_video_to_file.
    """
    temp_dir = None
    try:
        temp_dir, video_path, video_meta = _download_video_to_file(video_url)
        duration = video_meta.get("duration") or 0
        frame_urls = _extract_frame_data_urls_from_video(
            video_path, duration, num_frames=NUM_VIDEO_FRAMES_FOR_RECIPE
        )
        if not frame_urls:
            return None, None, None
        recipe = extract_recipe_from_images_llm(frame_urls)
        if not recipe:
            return None, None, None
        ingredients = recipe.get("ingredients") or []
        instructions = recipe.get("instructions") or []
        if ingredients and isinstance(ingredients[0], str):
            ingredients = [{"name": ing, "quantity": ""} for ing in ingredients]
        if not isinstance(instructions, list):
            instructions = [str(instructions)]
        recipe["ingredients"] = ingredients
        recipe["instructions"] = instructions
        source = {
            "type": "video",
            "url": video_url,
            "provider": video_meta.get("provider", ""),
            "title": video_meta.get("title", "") or recipe.get("name", ""),
            "image": video_meta.get("thumbnail"),
            "source_type": determine_source_type(video_url),
        }
        return recipe, source, "video_frames_vision"
    except Exception as e:
        print(f"⚠️ Frame+vision fallback failed: {e}")
        return None, None, None
    finally:
        if temp_dir:
            shutil.rmtree(temp_dir, ignore_errors=True)


def _run_frame_vision_fallback_from_path(
    video_path: str, video_url: str, meta: dict
) -> tuple[dict | None, dict | None, str | None]:
    """
    Frame+vision fallback using an already-downloaded video file (no extra download).
    Returns (recipe_dict, source_dict, extraction_method) or (None, None, None) on failure.
    """
    try:
        duration = meta.get("duration") or 0
        frame_urls = _extract_frame_data_urls_from_video(
            video_path, duration, num_frames=NUM_VIDEO_FRAMES_FOR_RECIPE
        )
        if not frame_urls:
            return None, None, None
        recipe = extract_recipe_from_images_llm(frame_urls)
        if not recipe:
            return None, None, None
        ingredients = recipe.get("ingredients") or []
        instructions = recipe.get("instructions") or []
        if ingredients and isinstance(ingredients[0], str):
            ingredients = [{"name": ing, "quantity": ""} for ing in ingredients]
        if not isinstance(instructions, list):
            instructions = [str(instructions)]
        recipe["ingredients"] = ingredients
        recipe["instructions"] = instructions
        source = {
            "type": "video",
            "url": video_url,
            "provider": meta.get("provider", ""),
            "title": meta.get("title", "") or recipe.get("name", ""),
            "image": meta.get("thumbnail"),
            "source_type": determine_source_type(video_url),
        }
        return recipe, source, "video_frames_vision"
    except Exception as e:
        print(f"⚠️ Frame+vision fallback (from path) failed: {e}")
        return None, None, None


def extract_recipe_from_video_internal(video_url: str):
    """
    Internal helper used by unified /extract-recipe endpoint.
    Downloads video once; extracts audio for Whisper and reuses same file for frame+vision fallback.
    Returns a Flask response: jsonify({...}), status_code
    """
    ok, err = validate_video_url(video_url)
    if not ok:
        return jsonify({"error": err}), 400

    temp_dir = None
    try:
        print(f"🎥 Processing video URL: {video_url}")

        # 1) Download video once (reused for audio extraction and for frame+vision fallback)
        t0 = time.time()
        try:
            temp_dir, video_path, meta = _download_video_to_file(video_url)
        except ValueError as ve:
            return jsonify({"error": str(ve)}), 413
        except yt_dlp.utils.DownloadError as de:
            error_str = str(de)
            is_instagram = "instagram.com" in video_url.lower()
            cookies_configured = (YTDLP_COOKIES_FILE and os.path.exists(YTDLP_COOKIES_FILE)) or YTDLP_COOKIES_B64
            if is_instagram:
                if not cookies_configured:
                    hint = "Instagram requires authentication. Please set YTDLP_COOKIES_FILE or YTDLP_COOKIES_B64."
                elif "empty media response" in error_str.lower() or "unavailable" in error_str.lower():
                    hint = "Instagram post may be private or cookies may be expired."
                else:
                    hint = "Instagram extraction failed. The post may be private or require fresh cookies."
            else:
                hint = "For TikTok/Instagram (and some YouTube), set YTDLP_COOKIES_FILE."
            return jsonify({
                "error": "Failed to download video",
                "details": error_str,
                "hint": hint,
                "cookies_configured": cookies_configured,
            }), 400
        except Exception as e:
            return jsonify({
                "error": "Video download failed",
                "user_message": "We couldn't download this video. Please try another link or add the recipe manually.",
                "details": str(e),
            }), 500

        print(f"✅ Video downloaded in {time.time() - t0:.2f}s")

        # 2) Extract audio from the downloaded video (no second download)
        audio_bytes = _extract_audio_from_video_file(video_path)
        if not audio_bytes:
            print("⚠️ Could not extract audio from video; using frame+vision fallback...")
            recipe, source, extraction_method = _run_frame_vision_fallback_from_path(video_path, video_url, meta)
            if recipe and source:
                _enrich_recipe_response(recipe)
                tags = extract_recipe_tags(recipe)
                return jsonify({
                    "source": source,
                    "recipe": recipe,
                    "tags": tags,
                    "transcript": None,
                    "extraction": {"method": extraction_method, "confidence": 0.5},
                    "meta": meta,
                }), 200
            return jsonify({
                "error": "Failed to extract recipe from video",
                "user_message": "We couldn't extract a recipe from this link. Please try another or add the recipe manually.",
                "message": "Could not extract audio and frame+vision fallback failed. Ensure ffmpeg is installed.",
            }), 500

        # 3) Whisper transcription
        print("🎤 Transcribing audio...")
        audio_file_obj = io.BytesIO(audio_bytes)
        audio_file_obj.name = "audio.mp3"
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file_obj,
            response_format="text",
        )
        transcript_text = transcript.strip() if isinstance(transcript, str) else str(transcript).strip()

        # If no usable transcript (empty or just music/noise), use frame+vision fallback (reuse same video)
        if not transcript_text or _is_likely_non_speech(transcript_text):
            print("🎬 No usable recipe instructions from audio; using frame+vision fallback (reusing video)...")
            recipe, source, extraction_method = _run_frame_vision_fallback_from_path(video_path, video_url, meta)
            if recipe and source:
                _enrich_recipe_response(recipe)
                tags = extract_recipe_tags(recipe)
                return jsonify({
                    "source": source,
                    "recipe": recipe,
                    "tags": tags,
                    "transcript": None,
                    "extraction": {"method": extraction_method, "confidence": 0.5},
                    "meta": meta,
                }), 200
            return jsonify({
                "error": "Failed to extract recipe from video",
                "user_message": "We couldn't extract a recipe from this link. Please try another or add the recipe manually.",
                "message": "No transcript from audio and frame+vision fallback failed. Ensure ffmpeg is installed.",
            }), 500

        # 4) LLM extraction from transcript (chunking + merge)
        print("🍳 Extracting recipe information from transcript...")
        chunks = _chunk_text(transcript_text, max_chars=6000)
        if not chunks:
            print("🎬 No chunks from transcript; using frame+vision fallback (reusing video)...")
            recipe, source, extraction_method = _run_frame_vision_fallback_from_path(video_path, video_url, meta)
            if recipe and source:
                _enrich_recipe_response(recipe)
                tags = extract_recipe_tags(recipe)
                return jsonify({
                    "source": source,
                    "recipe": recipe,
                    "tags": tags,
                    "transcript": None,
                    "extraction": {"method": extraction_method, "confidence": 0.5},
                    "meta": meta,
                }), 200
            return jsonify({
                "error": "Transcript is empty after cleanup",
                "user_message": "We couldn't extract a recipe from this link. Please try another or add the recipe manually.",
            }), 500

        # Extract recipe from transcript chunks in parallel for faster response
        parts = [None] * len(chunks)
        chunk_failed = {"idx": None, "error": None}
        max_workers = min(4, len(chunks))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_extract_recipe_chunk, ch): i for i, ch in enumerate(chunks)}
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    parts[idx] = future.result()
                except Exception as e:
                    chunk_failed["idx"] = idx + 1
                    chunk_failed["error"] = e
                    break
        if chunk_failed["idx"] is not None:
            print(f"⚠️ Transcript chunk extraction failed (chunk {chunk_failed['idx']}); using frame+vision fallback (reusing video)...")
            recipe, source, extraction_method = _run_frame_vision_fallback_from_path(video_path, video_url, meta)
            if recipe and source:
                _enrich_recipe_response(recipe)
                tags = extract_recipe_tags(recipe)
                return jsonify({
                    "source": source,
                    "recipe": recipe,
                    "tags": tags,
                    "transcript": None,
                    "extraction": {"method": extraction_method, "confidence": 0.5},
                    "meta": meta,
                }), 200
            return jsonify({
                "error": "Failed to extract recipe from transcript chunk",
                "user_message": "We couldn't extract a recipe from this link. Please try another or add the recipe manually.",
                "chunk": chunk_failed["idx"],
                "details": str(chunk_failed["error"]),
                "transcript": transcript_text
            }), 500
        parts = [p for p in parts if p is not None]
        if not parts:
            return jsonify({
                "error": "No recipe parts extracted from transcript",
                "user_message": "We couldn't extract a recipe from this link. Please try another or add the recipe manually.",
            }), 500

        recipe = _merge_recipe_parts(parts) if len(parts) > 1 else parts[0]
        ingredients = recipe.get("ingredients") or []
        instructions = recipe.get("instructions") or []
        if ingredients and isinstance(ingredients[0], str):
            ingredients = [{"name": ing, "quantity": ""} for ing in ingredients]
        if not isinstance(instructions, list):
            instructions = [str(instructions)]
        recipe["ingredients"] = ingredients
        recipe["instructions"] = instructions

        # If transcript yielded empty instructions, use frame+vision fallback (reuse same video)
        if not instructions:
            print("🎬 Instructions empty from transcript; using frame+vision fallback (reusing video)...")
            fallback_recipe, fallback_source, extraction_method = _run_frame_vision_fallback_from_path(
                video_path, video_url, meta
            )
            if fallback_recipe and fallback_source and (
                fallback_recipe.get("instructions") or fallback_recipe.get("ingredients")
            ):
                _enrich_recipe_response(fallback_recipe)
                tags = extract_recipe_tags(fallback_recipe)
                return jsonify({
                    "source": fallback_source,
                    "recipe": fallback_recipe,
                    "tags": tags,
                    "transcript": None,
                    "extraction": {"method": extraction_method, "confidence": 0.5},
                    "meta": meta,
                }), 200
        elif not ingredients:
            print("🎬 Ingredients empty from transcript; using frame+vision fallback (reusing video)...")
            fallback_recipe, fallback_source, extraction_method = _run_frame_vision_fallback_from_path(
                video_path, video_url, meta
            )
            if fallback_recipe and fallback_source:
                _enrich_recipe_response(fallback_recipe)
                tags = extract_recipe_tags(fallback_recipe)
                return jsonify({
                    "source": fallback_source,
                    "recipe": fallback_recipe,
                    "tags": tags,
                    "transcript": None,
                    "extraction": {"method": extraction_method, "confidence": 0.5},
                    "meta": meta,
                }), 200

        source_type = determine_source_type(video_url)
        _enrich_recipe_response(recipe)
        tags = extract_recipe_tags(recipe)
        source = {
            "type": "video",
            "url": video_url,
            "provider": meta.get("provider", ""),
            "title": meta.get("title", "") or recipe.get("name", ""),
            "image": meta.get("thumbnail"),
            "source_type": source_type,
        }
        return jsonify({
            "source": source,
            "recipe": recipe,
            "tags": tags,
            "transcript": transcript_text,
            "extraction": {"method": "transcript_llm", "confidence": 0.55},
            "meta": meta,
        }), 200

    except Exception as e:
        print(f"💥 Critical error in extract_recipe_from_video_internal: {str(e)}")
        return jsonify({
                "error": "Unexpected server error",
                "user_message": DEFAULT_500_USER_MESSAGE,
                "details": str(e),
            }), 500
    finally:
        if temp_dir:
            shutil.rmtree(temp_dir, ignore_errors=True)



    ## video and webpage code ends here#####
   

if __name__ == "__main__":
    # Disable Flask's built-in debugger when running in VS Code debugger
    # Set FLASK_DEBUG environment variable to enable Flask debug mode separately
    import os
    flask_debug = os.environ.get('FLASK_DEBUG', '0') == '1'
    app.run(debug=flask_debug, host='0.0.0.0', port=5002, use_reloader=False)
