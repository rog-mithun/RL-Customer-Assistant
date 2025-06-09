from sentence_transformers import SentenceTransformer
import numpy as np
import random

class MultiTurnChatbotEnv:
    def __init__(self, max_turns=3, history_length=2, similarity_threshold=0.6):
        self.queries = [
            # Greetings
            "hi", "hello", "hey", "good morning",

            # Farewells
            "thanks", "thank you", "bye", "see you",

            # Password Reset
            "forgot my password", "how to reset password?",

            # Business Hours
            "when are you open?", "business hours?",

            # Order Issues
            "where is my order?", "order not arrived", "order is delayed",

            # Cancel Order
            "cancel my order", "stop my purchase", "how to cancel order",

            # Refund Policy
            "what's your refund policy?", "can I get a refund?", "return an item",

            # Small Talk
            "hi darling", "what's up?", "how are you?", "yo!", "sup bot",

            # Confusing Inputs
            "play music", "do you love me?", "what's 2+2"
        ]

        self.responses = [
            # 0 - Greeting
            "Hello! How can I help you today?",

            # 1 - Farewell
            "You're welcome! Have a great day!",

            # 2 - Password Reset
            "You can reset your password using the link we sent to your email.",

            # 3 - Business Hours
            "Our business hours are 9 AM to 5 PM, Monday to Friday.",

            # 4 - Order Issues
            "I'm sorry to hear that. Let me check the tracking info for you.",

            # 5 - Cancel Order
            "Sure, I can help you cancel your order.",

            # 6 - Refund Policy
            "Yes, we offer a 30-day refund policy for eligible returns.",

            # 7 - Small Talk
            "I'm just a bot, but I’m here to help you!",

            # 8 - Fallback
            "Sorry, I didn’t quite get that. Could you please rephrase?"
        ]

        self.intent_mapping = {
            0: list(range(0, 4)),      # Greetings
            1: list(range(4, 8)),      # Farewell
            2: list(range(8, 10)),     # Password Reset
            3: list(range(10, 12)),    # Business Hours
            4: list(range(12, 15)),    # Order Issues
            5: list(range(15, 18)),    # Cancel Order
            6: list(range(18, 21)),    # Refund Policy
            7: list(range(21, 26)),    # Small Talk
            8: list(range(26, 29))     # Fallback
        }

        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.state_size = 384 * history_length
        self.action_size = len(self.responses)
        self.max_turns = max_turns
        self.history_length = history_length
        self.similarity_threshold = similarity_threshold

        self.turn_count = 0
        self.conversation = []

    def reset(self):
        self.turn_count = 0
        self.conversation = []
        init_query = random.choice(self.queries)
        self.conversation.append(init_query)
        return self.get_state()

    def step(self, action):
        user_input = self.conversation[-1]
        correct_action = self.get_best_response_index(user_input)

        reward = 1 if action == correct_action else -1
        done = False

        if self.turn_count + 1 >= self.max_turns or user_input.lower().startswith("thanks"):
            done = True
            next_query = "thanks"
        else:
            next_query = random.choice(self.queries)

        self.conversation.append(next_query)
        self.turn_count += 1

        return self.get_state(), reward, done, {
            "user_input": user_input,
            "bot_response": self.responses[action],
            "expected_response": self.responses[correct_action],
            "next_query": next_query
        }

    def get_state(self):
        history = self.conversation[-self.history_length:]
        embeddings = [self.model.encode(q) for q in history]
        while len(embeddings) < self.history_length:
            embeddings.insert(0, np.zeros(384))
        return np.concatenate(embeddings)

    def get_best_response_index(self, user_input):
        user_vec = self.model.encode(user_input)
        similarities = [np.dot(user_vec, self.model.encode(q)) for q in self.queries]
        max_sim = max(similarities)
        if max_sim < self.similarity_threshold:
            return 8  # fallback response
        query_index = np.argmax(similarities)
        for resp_id, query_indices in self.intent_mapping.items():
            if query_index in query_indices:
                return resp_id
        return 8  # fallback
