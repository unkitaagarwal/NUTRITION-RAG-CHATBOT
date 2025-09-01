import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime

def init_firestore():
    if not firebase_admin._apps:
        cred = credentials.Certificate("firebase_service_account.json")
        firebase_admin.initialize_app(cred)
    return firestore.client()

def get_user_context(email, max_logs=10):
    db = init_firestore()
    
    # Load meal logs directly from log_entry collection using user_email
    # Temporarily removed order_by to avoid needing composite index
    logs = db.collection("log_entry").where("user_email", "==", email).limit(max_logs).stream()
    
    meal_history = []
    snacks_only = []
    breakfast_only = []
    lunch_only = []
    dinner_only = []
    
    for log in logs:
        log_data = log.to_dict()
        
        # Debug: Print what we're getting from Firebase
        print(f"DEBUG: Retrieved log data: {log_data}")
        
        # Handle item_name - it might be a string or array
        item_name = log_data.get('item_name', '')
        if isinstance(item_name, list):
            items_str = ', '.join(item_name)
        else:
            items_str = str(item_name) if item_name else 'Unknown item'
        
        meal_type = log_data.get('meal_type', 'Unknown meal')
        date_time = log_data.get('date_time')
        date_str = date_time.date() if date_time else 'Unknown date'
        calories = log_data.get('total_calories', 0)
        carbs = log_data.get('total_carbs', 0)
        protein = log_data.get('total_protein', 0)
        fat = log_data.get('total_fat', 0)
        
        # Enhanced meal entry with nutritional info
        meal_entry = f"{meal_type} on {date_str}: {items_str} - {calories} kcal (Carbs: {carbs}g, Protein: {protein}g, Fat: {fat}g)"
        meal_history.append(meal_entry)
        
        # Categorize by meal type for filtered responses
        if meal_type.lower() == 'snacks':
            snacks_only.append(meal_entry)
        elif meal_type.lower() == 'breakfast':
            breakfast_only.append(meal_entry)
        elif meal_type.lower() == 'lunch':
            lunch_only.append(meal_entry)
        elif meal_type.lower() == 'dinner':
            dinner_only.append(meal_entry)
    
    print(f"DEBUG: Final meal_history: {meal_history}")
    print(f"DEBUG: Snacks only: {snacks_only}")
    
    # For now, return empty goal since we're focusing on meal data
    # TODO: Implement user preferences/goals lookup when the structure is clear
    goal = {}
    
    # Return both all meals and categorized meals
    categorized_meals = {
        'all': meal_history,
        'snacks': snacks_only,
        'breakfast': breakfast_only,
        'lunch': lunch_only,
        'dinner': dinner_only
    }
    
    return goal, categorized_meals

def save_user_chat(email, question, answer):
    db = init_firestore()
    chat_ref = db.collection("users").document(email).collection("chats")
    chat_ref.add({
        "question": question,
        "answer": answer,
        "timestamp": datetime.utcnow()
    })

def get_user_chat_history(email, max_chats=5):
    db = init_firestore()
    chat_ref = db.collection("users").document(email).collection("chats")
    
    docs = chat_ref.order_by("timestamp", direction=firestore.Query.DESCENDING).limit(max_chats).stream()
    
    history = []
    for doc in docs:
        data = doc.to_dict()
        question = data.get("question", "")
        answer = data.get("answer", "")
        history.append((question, answer))
    
    # Reverse to maintain chronological order (oldest → newest)
    return history[::-1]

