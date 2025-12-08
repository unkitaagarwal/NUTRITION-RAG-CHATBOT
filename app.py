from flask import Flask, request, jsonify, send_file
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from firebase_utils import get_user_context, get_user_chat_history, save_user_chat
from dotenv import load_dotenv
import os
import io
import json
import re
import base64
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

        # 2. Call GPT-Vision API using OpenAI client
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
                "details": str(vision_error)
            }), 500

    except Exception as e:
        print("💥 Critical error:", str(e))
        return jsonify({
            "error": "Something went wrong analyzing the image",
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
        
        # Generate images for meals if requested
        if include_images:
            def generate_meal_image(meal_data, index):
                """Generate image for a single meal using DALL-E"""
                try:
                    meal_name = meal_data["name"]
                    meal_desc = meal_data["description"]
                    # Create a descriptive prompt for the image
                    image_prompt = f"Professional food photography of {meal_name}. {meal_desc}. High quality, appetizing, well-lit, restaurant style, on a plate, food photography"
                    
                    # Generate image using DALL-E 2 (most cost-effective option)
                    # Pricing: DALL-E 2 is cheaper than DALL-E 3
                    # Size pricing: 256x256 < 512x512 < 1024x1024 (smaller = cheaper)
                    # Using DALL-E 2 with 256x256 for maximum cost savings
                    image_response = client.images.generate(
                        model="dall-e-2",  # Most inexpensive model
                        prompt=image_prompt,
                        size="512x512", 
                        n=1,
                    )
                    
                    return image_response.data[0].url
                except Exception as e:
                    error_msg = str(e)
                    print(f"❌ Error generating image for {meal_data['name']}: {error_msg}")
                    # Log more details if it's an OpenAI API error
                    if hasattr(e, 'response'):
                        try:
                            error_data = e.response.json() if hasattr(e.response, 'json') else {}
                            print(f"   API Error details: {error_data}")
                        except:
                            pass
                    return None
            
            # Generate images in parallel using threads for faster processing
            image_results = {}
            threads = []
            
            def generate_with_index(meal_idx, meal_data):
                url = generate_meal_image(meal_data, meal_idx)
                image_results[meal_idx] = url
            
            for idx, meal in enumerate(validated_meals):
                thread = Thread(target=generate_with_index, args=(idx, meal))
                threads.append(thread)
                thread.start()
            
            # Wait for all image generation threads to complete (max 60 seconds timeout)
            for thread in threads:
                thread.join(timeout=60)
            
            # Assign image URLs to meals
            for idx, meal in enumerate(validated_meals):
                meal["imageUrl"] = image_results.get(idx)
        
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
        return jsonify({"error": error_msg}), 500
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


# ---------------------------
# Helper: Transcribe Audio
# ---------------------------
def transcribe_audio_file(audio_bytes: bytes, file_name: str) -> str:
    try:
        # Create file-like object
        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = file_name

        transcription = client.audio.transcriptions.create(
            model="whisper-1",  # Cost-effective: Much cheaper than gpt-4o-transcribe, same quality
            file=audio_file,
            response_format="text",
        )

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
            "details": str(e)
        }), 500



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