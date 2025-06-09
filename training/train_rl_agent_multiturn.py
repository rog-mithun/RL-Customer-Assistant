import torch
import numpy as np
from environment.chatbot_env_multiturn import MultiTurnChatbotEnv
from models.dqn_agent import DQNAgent
import os, csv
import matplotlib.pyplot as plt



# Setup
os.makedirs("logs", exist_ok=True)
env = MultiTurnChatbotEnv()
state_size = env.state_size
action_size = env.action_size
agent = DQNAgent(state_size, action_size)

# Logging
log_file = open("logs/multiturn_log.csv", mode='w', newline='')
csv_writer = csv.writer(log_file)
csv_writer.writerow(["Episode", "Turn", "User Query", "Bot Response", "Expected Response", "Reward", "Done"])

# Training loop
episodes = 500
target_update_freq = 20
reward_log = []

for ep in range(episodes):
    state = env.reset()
    episode_reward = 0

    for t in range(env.max_turns):
        action = agent.act_vector(state)
        next_state, reward, done, info = env.step(action)

        agent.remember_vector(state, action, reward, next_state, done)
        agent.replay_vector()

        # Logging
        csv_writer.writerow([
            ep+1, t+1, info["user_input"], info["bot_response"],
            info["expected_response"], reward, done
        ])

        state = next_state
        episode_reward += reward

        if done:
            break

    reward_log.append(episode_reward)
    print(f"Episode {ep+1}/{episodes} | Total Reward: {episode_reward} | Epsilon: {agent.epsilon:.2f}")

    if ep % target_update_freq == 0:
        agent.update_target_model()

# Save model
torch.save(agent.model.state_dict(), "dqn_chatbot_model_multiturn.pth")

# Plot rewards
plt.plot(reward_log)
plt.title("Multi-Turn Chatbot Training")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.savefig("logs/multiturn_rewards.png")
plt.show()

log_file.close()
