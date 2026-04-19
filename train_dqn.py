import numpy as np
import os
import torch
from rl_environment import NegotiationEnv
from dqn_agent import DQNAgent

print("=" * 60)
print("ENTRAINEMENT DE L AGENT DQN - REINFORCEMENT LEARNING")
print("=" * 60)

env   = NegotiationEnv()
agent = DQNAgent(state_size=8, action_size=6)

PRETRAINED_PATH = "models/dqn_agent_pretrained.pth"
if os.path.exists(PRETRAINED_PATH):
    agent.load(PRETRAINED_PATH)
    agent.epsilon = 0.4
    print(f"Checkpoint pre-entraine charge : {PRETRAINED_PATH}")
    print(f"Epsilon de depart : {agent.epsilon}")
else:
    print("Pas de checkpoint - entrainement from scratch (epsilon=1.0)")

print()

episodes            = 6000
batch_size          = 64
update_target_every = 50

print(f"Debut de l entrainement sur {episodes} episodes...\n")

rewards_history = []
success_history = []

for e in range(episodes):
    state        = env.reset()
    total_reward = 0
    done         = False
    steps        = 0
    actions_log  = []          # ✅ Pour détecter les répétitions

    while not done and steps < 15:
        action                         = agent.act(state)
        next_state, reward, done, info = env.step(action)

        # ✅ Pénalité si l'agent répète la même action 2 fois de suite
        if len(actions_log) > 0 and action == actions_log[-1]:
            reward -= 2.0

        # ✅ Pénalité forte si l'agent répète 3 fois la même action
        if len(actions_log) >= 2 and action == actions_log[-1] == actions_log[-2]:
            reward -= 5.0

        actions_log.append(action)

        agent.remember(state, action, reward, next_state, done)
        agent.replay(batch_size)

        state         = next_state
        total_reward += reward
        steps        += 1

    rewards_history.append(total_reward)
    success_history.append(1 if info.get('deal_conclu') else 0)

    if e % update_target_every == 0:
        agent.update_target_model()

    if (e + 1) % 100 == 0:
        avg_reward = np.mean(rewards_history[-100:])
        deal_rate  = np.mean(success_history[-100:]) * 100
        print(f"Episode {e+1:>4}/{episodes} | "
              f"Reward moyen : {avg_reward:>7.2f} | "
              f"Deals conclus : {deal_rate:>5.1f}% | "
              f"Epsilon : {agent.epsilon:.4f}")

print("\n" + "=" * 60)
print("Entrainement termine !")
print("=" * 60)

os.makedirs("models", exist_ok=True)
agent.save("models/dqn_agent.pth")
print("Agent sauvegarde dans : models/dqn_agent.pth")

print(f"\nReward moyen final  : {np.mean(rewards_history):.2f}")
print(f"Reward max          : {np.max(rewards_history):.2f}")
print(f"Taux de deals       : {np.mean(success_history) * 100:.1f}%")

print("\n" + "=" * 60)
print("EVALUATION DE L AGENT ENTRAINE (epsilon=0)")
print("=" * 60)

agent.epsilon = 0.0
eval_episodes = 20
eval_rewards  = []
eval_deals    = []

for ep in range(eval_episodes):
    state        = env.reset()
    total_reward = 0
    done         = False
    steps        = 0
    actions_log  = []

    while not done and steps < 15:
        action                         = agent.act(state)
        next_state, reward, done, info = env.step(action)
        actions_log.append(env.ACTION_NAMES[action])
        state         = next_state
        total_reward += reward
        steps        += 1

    eval_rewards.append(total_reward)
    eval_deals.append(1 if info.get('deal_conclu') else 0)
    status = "DEAL" if info.get('deal_conclu') else "ECHEC"

    print(f"Ep {ep+1:>2} | {env.destination:<30} | "
          f"Budget: {env.budget_client:.0f} | "
          f"Prix final: {env.prix_actuel:.0f} | "
          f"Reward: {total_reward:>7.2f} | {status}")
    print(f"       Actions : {' -> '.join(actions_log)}")

print("\n" + "-" * 60)
print(f"Reward moyen eval  : {np.mean(eval_rewards):.2f}")
print(f"Taux de deals eval : {np.mean(eval_deals) * 100:.1f}%")
print("=" * 60)