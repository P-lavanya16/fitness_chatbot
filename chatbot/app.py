import pandas as pd
import cohere
import os
from dotenv import load_dotenv
import streamlit as st

# Load environment variables
load_dotenv()
cohere_api_key = os.getenv("COHERE_API_KEY")
co = cohere.Client(cohere_api_key)

# --- Dataset Loading ---
def load_exercise_data(csv_file):
    try:
        df = pd.read_csv(csv_file)
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

# Replace 'megaGymDataset.csv' with your actual filename
exercise_data = load_exercise_data('chatbot/megaGymDataset.csv')

# --- Gather User Preferences ---
def gather_user_preferences():
    st.sidebar.header("Your Preferences")
    goal = st.sidebar.selectbox(
        "What's your main fitness goal?",
        ["Weight Loss", "Build Muscle", "Endurance", "General Fitness"]
    )
    experience = st.sidebar.radio(
        "What's your experience level?",
        ["Beginner", "Intermediate", "Advanced"]
    )
    restrictions = st.sidebar.multiselect(
        "Any injuries or limitations?",
        options=[
            "None", "Pregnancy", "Lung Infection", "Back Pain",
            "Knee Injury", "Shoulder Injury", "Diabetes", "Heart Condition"
        ],
        default=["None"]
    )
    restrictions_text = ', '.join([r for r in restrictions if r != "None"])
    return {"goal": goal, "experience": experience, "restrictions": restrictions_text}

# --- Process Query ---
def process_query(query, exercise_data, user_preferences):
    if exercise_data is None:
        return "The exercise data is not available. Please check the dataset."

    prompt = craft_fitness_prompt(query, user_preferences, exercise_data)
    try:
        response = co.generate(
            model='command-nightly',
            prompt=prompt,
            max_tokens=200,
            stop_sequences=["--"]
        )
        return response.generations[0].text.strip()
    except Exception as e:
        return f"Error generating response: {e}"

# --- Helper Functions ---
def craft_fitness_prompt(query, user_preferences, data):
    user_info = (
        f"Goal: {user_preferences['goal']}, "
        f"Experience: {user_preferences['experience']}, "
        f"Restrictions: {user_preferences['restrictions']}."
    )
    return (
        f"You are a fitness expert helping a user. "
        f"User preferences: {user_info}\n"
        f"User's question: {query}\n"
        f"Relevant exercise data: {data.head(3).to_string(index=False)}\n"
        f"Provide a detailed and helpful response."
    )

# --- Streamlit UI ---
st.title("Fitness Knowledge Bot")
user_preferences = gather_user_preferences()

st.write("### Ask about workouts or fitness tips!")
user_input = st.text_input("Enter your question here:")

if st.button("Submit"):
    chatbot_response = process_query(user_input, exercise_data, user_preferences)
    st.write("**Chatbot Response:**")
    st.write(chatbot_response)

    # Feedback section
    feedback = st.radio(
        "Was this response helpful?",
        options=["Yes", "No"],
        index=0
    )
    if st.button("Submit Feedback"):
        st.write("Thank you for your feedback!")
