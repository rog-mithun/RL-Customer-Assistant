# 🤖 Reinforcement Learning-Based Customer Support Chatbot with Voice Assistant

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A Reinforcement Learning-based chatbot built using Deep Q-Network (DQN), capable of handling multi-turn conversations with a voice interface. Designed for customer support, this assistant integrates voice input, real-time response generation, learning-based improvement, and conversation exports via UI.

---

## 📌 Key Features

- 🧠 Multi-turn RL-based response generation (Deep Q-Network)
- 🎙️ Voice command input using SpeechRecognition
- 🔁 Real-time reward-based learning loop
- 💬 BERT-based state encoding
- 🗂️ Export conversations as CSV/PDF
- 🖥️ Streamlit interface with dynamic chat view
- 📊 Experience replay buffer and policy updates

---

## 🛠️ Tech Stack

### Languages:
![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)

### Frameworks & Libraries:
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)

### NLP & RL:
![BERT](https://img.shields.io/badge/BERT-NLP-blue?style=flat)
![DQN](https://img.shields.io/badge/DQN-RL-black?style=flat)

### Voice Processing:
![SpeechRecognition](https://img.shields.io/badge/SpeechRecognition-voice-yellow?style=flat)
![pyttsx3](https://img.shields.io/badge/pyttsx3-voice-green?style=flat)

### Reporting & Utilities:
![pandas](https://img.shields.io/badge/pandas-data-blue?style=flat&logo=pandas&logoColor=white)
![fpdf](https://img.shields.io/badge/fpdf-PDF-red?style=flat)

---

## 📁 Project Structure

| File/Folder                  | Description                                     |
|------------------------------|-------------------------------------------------|
| `training/`                  | RL training loop, reward functions              |
| `models/`                    | Deep Q-Network architecture & buffer logic      |
| `ui/`                        | Streamlit frontend with voice and chat handling |
| `media/`                     | Sample outputs and exported results             |
| `environment/`               | Environment simulation for RL reward modeling   |
| `logs/`                      | Policy/reward logs during runtime               |
| `chat_export.csv`            | Example exported conversation                   |
| `dqn_chatbot_model_multiturn.pth` | Pre-trained model weights              |
| `requirements.txt`           | Dependency list                                 |

---

## 🚀 How to Run

1. **Clone the repository:**
   ```bash
   git clone https://github.com/rog-mithun/RL-Customer-Assistant.git
   cd RL-Customer-Assistant

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt

3. **Run the Streamlit App:**
   ```bash
   streamlit run ui/multiturn_chat.py

---

## 📂 Demo & Output Samples

### 📄 Chat Export & Reward Files
- [Chat Log](chat_export.csv)
- [Reward Log](logs/multiturn_log.csv)
- [Reward Graph](logs/multiturn_rewards.png)

### 🎥 Model Output or Voice Demo
[▶️ Play Demo Output](media/demo_video.mp4)

---

## 📖 License

This project is licensed under the [MIT License](LICENSE) © 2025 Mithunsankar S.
