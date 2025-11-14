# chatbot_module.py

import joblib
import google.generativeai as genai
from dotenv import load_dotenv
import os

# -----------------------------
# 1. LOAD MODEL & VECTORIZER
# -----------------------------
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# -----------------------------
# 2. LOAD GEMINI API KEY
# -----------------------------
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Gemini model
gemini_model = genai.GenerativeModel("gemini-flash-latest")

def call_gemini(prompt):
    """
    Fallback to Gemini when NB model confidence is low.
    """
    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return "I'm facing a technical issue, but I can help with electricity bills or energy-saving tips! ⚡"


# -----------------------------
# 3. INTENT → RESPONSE MAPPING
# -----------------------------
responses = {
    "high_bill": "Your electricity bill seems high. Try monitoring AC usage, switching to LED bulbs, and reducing standby power. ⚡",
    "low_bill": "Great job! Your low electricity bill shows efficient usage. Keep it up! 🌱",

    "energy_tips": "Here are some tips: unplug chargers, use LED bulbs, avoid peak hours, clean appliance filters, and turn off unused devices. 💡",

    "ac_tips": "To save AC electricity: set temperature to 24°C, clean filters monthly, use eco mode, avoid frequent on/off cycles, and close doors/windows while cooling. ❄️",

    "greeting": "Hello! I'm your Energy Advisor. How can I help you today? 😊",

    "thanks": "You're welcome! Happy to help anytime. 🌟",

    "general_faq": (
        "That's a general electricity-related question! ⚡\n"
        "- Power Saving Mode reduces appliance energy consumption by optimizing performance.\n"
        "- A kWh is the standard unit for measuring electricity usage.\n"
        "- High-watt appliances consume more power.\n"
        "Feel free to ask me anything related to electricity, appliances, or energy usage! 📘"
    ),

    "unknown": "I'm not sure about that, but I can help with electricity bills, AC usage, or energy-saving tips! 🔎"
}

# -----------------------------
# 4. PREDICT INTENT WITH THRESHOLD + GEMINI FALLBACK
# -----------------------------
def predict_intent(user_message):
    vector = vectorizer.transform([user_message])
    probs = model.predict_proba(vector)[0]

    max_prob = max(probs)
    predicted_intent = model.classes_[probs.argmax()]

    # If confidence is low → fallback to Gemini
    if max_prob < 0.50:
        return "gemini_fallback"

    return predicted_intent


# -----------------------------
# 5. MAIN CHATBOT RESPONSE LOGIC
# -----------------------------
def chatbot_response(user_message):
    intent = predict_intent(user_message)

    # GEMINI FALLBACK
    if intent == "gemini_fallback":
        return call_gemini(user_message)

    return responses[intent]


# -----------------------------
# 6. TERMINAL TEST MODE
# -----------------------------
if __name__ == "__main__":
    print("Hybrid Energy Chatbot Ready! Type 'exit' to stop.\n")
    while True:
        user = input("You: ")
        if user.lower() in ["exit", "quit", "stop"]:
            break
        print("Bot:", chatbot_response(user))
