import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import torch
import numpy as np
import pyttsx3
import speech_recognition as sr
import pandas as pd
from fpdf import FPDF

from models.dqn_agent import DQN
from environment.chatbot_env_multiturn import MultiTurnChatbotEnv

# Initialize environment and model
env = MultiTurnChatbotEnv()
state_size = env.state_size
action_size = env.action_size

model = DQN(state_size, action_size)
model.load_state_dict(torch.load("dqn_chatbot_model_multiturn.pth"))
model.eval()

# Streamlit page config
st.set_page_config(page_title="RL Chatbot", layout="centered")
st.markdown("<h1 style='text-align:center;'>🤖 RL-Powered Customer Support Chatbot</h1>", unsafe_allow_html=True)

# Initialize chat session
if "conversation" not in st.session_state:
    st.session_state.conversation = []
    st.session_state.turn_count = 0

# Custom CSS for chat bubbles
st.markdown("""
    <style>
    .user-bubble {
        background-color: #1f77b4;
        color: white;
        padding: 10px;
        border-radius: 12px;
        margin: 5px 0px 5px auto;
        width: fit-content;
        max-width: 80%;
    }
    .bot-bubble {
        background-color: #444;
        color: white;
        padding: 10px;
        border-radius: 12px;
        margin: 5px auto 5px 0px;
        width: fit-content;
        max-width: 80%;
    }
    </style>
""", unsafe_allow_html=True)

# User input section
st.markdown("### 🎤 Voice or Keyboard Input")
col1, col2 = st.columns([3, 1])

with col1:
    user_input = st.text_input("Type your message here:")

with col2:
    if st.button("🎙️ Use Voice"):
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            st.info("🎧 Listening... Speak now.")
            audio = recognizer.listen(source, phrase_time_limit=5)

        try:
            user_input = recognizer.recognize_google(audio)
            st.success(f"Recognized: {user_input}")

            # Store user input
            st.session_state.conversation.append(("user", user_input))

            # Predict and respond immediately
            env.conversation = [msg for role, msg in st.session_state.conversation if role == "user"]
            state = env.get_state()
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                q_vals = model(state_tensor)
                action = torch.argmax(q_vals).item()

            bot_response = env.responses[action]
            st.session_state.conversation.append(("bot", bot_response))

            # Speak the response
            engine = pyttsx3.init()
            engine.say(bot_response)
            engine.runAndWait()

        except sr.UnknownValueError:
            st.warning("Sorry, I couldn't understand. Try again.")
        except sr.RequestError:
            st.error("Speech recognition service unavailable.")


# Process input and respond
if st.button("Send") and user_input.strip():
    # Save user input
    st.session_state.conversation.append(("user", user_input))

    # Prepare state for prediction
    env.conversation = [msg for role, msg in st.session_state.conversation if role == "user"]
    state = env.get_state()
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

    # Predict bot action
    with torch.no_grad():
        q_vals = model(state_tensor)
        action = torch.argmax(q_vals).item()
    bot_response = env.responses[action]
    st.session_state.conversation.append(("bot", bot_response))

    # Speak bot response
    try:
        engine = pyttsx3.init()
        engine.say(bot_response)
        engine.runAndWait()
    except:
        st.warning("Voice output failed. Continuing with text only.")

# Display chat history
for role, msg in st.session_state.conversation:
    if role == "user":
        st.markdown(f"<div class='user-bubble'>🧑 {msg}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='bot-bubble'>🤖 {msg}</div>", unsafe_allow_html=True)

# Reset button
if st.button("🔄 Start New Conversation"):
    st.session_state.conversation = []
    st.session_state.turn_count = 0

# Export options
st.markdown("---")
st.subheader("📤 Export Chat")

col_csv, col_pdf = st.columns(2)

with col_csv:
    if st.button("Download as CSV"):
        csv_data = pd.DataFrame(st.session_state.conversation, columns=["Role", "Message"])
        csv_data.to_csv("chat_export.csv", index=False)
        st.success("✅ Saved as chat_export.csv")

