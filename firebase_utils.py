import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime
import threading

_firestore_client = None
_firestore_lock = threading.Lock()

def init_firestore():
    global _firestore_client
    if _firestore_client:
        return _firestore_client

    with _firestore_lock:
        if not firebase_admin._apps:
            cred = credentials.Certificate("firebase_service_account.json")
            firebase_admin.initialize_app(cred)
        if _firestore_client is None:
            _firestore_client = firestore.client()
    return _firestore_client

def get_user_context(email, max_logs=5):
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

    # Fetch user preferences/goals (stored in subcollection user_preferences)
    user_preferences = {}
    try:
        # Find the user document by email in nutrilensai-77be2 collection
        user_doc_ref = None
        user_query = (
            db.collection("nutrilensai-77be2")
            .where("email", "==", email)
            .limit(1)
            .stream()
        )
        for user_doc in user_query:
            user_doc_ref = user_doc.reference
            print(f"DEBUG: Found user document for {email} with ID: {user_doc_ref.id}")
            break

        if user_doc_ref:
            # Access user_preferences subcollection under the user document
            # Get the first document (assuming one document per user, or get any if multiple exist)
            prefs_query = (
                user_doc_ref.collection("user_preferences")
                .limit(1)
                .stream()
            )
            
            pref_count = 0
            for pref_doc in prefs_query:
                user_preferences = pref_doc.to_dict() or {}
                pref_count += 1
                print(f"DEBUG: Found user_preferences document with ID: {pref_doc.id}")
                print(f"DEBUG: User preferences keys: {list(user_preferences.keys())}")
                break
            
            if pref_count == 0:
                print(f"DEBUG: No user_preferences subcollection found under user document {user_doc_ref.id}")
                # List all subcollections to help debug
                print(f"DEBUG: Checking if user_preferences subcollection exists...")
        else:
            print(f"DEBUG: No user document found for email: {email} in nutrilensai-77be2 collection")

        print(f"DEBUG: User preferences: {user_preferences}")
    except Exception as e:
        print(f"DEBUG: Error fetching user preferences for {email}: {e}")
        import traceback
        traceback.print_exc()
        user_preferences = {}
    
    # goal kept for backwards compatibility (now mirrors preferences)
    goal = user_preferences.copy()
    
    # Return both all meals and categorized meals
    categorized_meals = {
        'all': meal_history,
        'snacks': snacks_only,
        'breakfast': breakfast_only,
        'lunch': lunch_only,
        'dinner': dinner_only
    }
    
    return goal, categorized_meals, user_preferences

def save_user_chat(email, question, answer):
    db = init_firestore()
    
    # Ensure the user document exists first
    user_ref = db.collection("users").document(email)
    user_doc = user_ref.get()
    
    if not user_doc.exists:
        # Create user document if it doesn't exist
        user_ref.set({
            "email": email,
            "created_at": datetime.utcnow()
        })
        print(f"DEBUG: Created new user document for {email}")
    
    # Now add the chat to the chats subcollection
    chat_ref = user_ref.collection("chats")
    chat_doc = chat_ref.add({
        "question": question,
        "answer": answer,
        "timestamp": datetime.utcnow()
    })
    
    print(f"DEBUG: Successfully saved chat for {email} with ID: {chat_doc[1].id}")

def get_user_chat_history(email, max_chats=2):  # Reduced from 3 to 2 for faster queries
    db = init_firestore()
    print(f"DEBUG: Looking for chat history for email: {email}")
    
    chat_ref = db.collection("users").document(email).collection("chats")
    print(f"DEBUG: Chat reference path: users/{email}/chats")
    
    docs = chat_ref.order_by("timestamp", direction=firestore.Query.DESCENDING).limit(max_chats).stream()
    
    history = []
    for doc in docs:
        data = doc.to_dict()
        question = data.get("question", "")
        answer = data.get("answer", "")
        history.append((question, answer))
        print(f"DEBUG: Found chat entry - Q: {question[:50]}..., A: {answer[:50]}...")
    
    print(f"DEBUG: Total chat history entries found: {len(history)}")
    
    # Reverse to maintain chronological order (oldest → newest)
    return history[::-1]

