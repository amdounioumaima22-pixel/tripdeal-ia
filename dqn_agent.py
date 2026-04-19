import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim

class DQNAgent:
    """
    Agent DQN — VERSION PRO
    
    Corrections vs v1 :
    ✅ Optimizer créé UNE FOIS (pas à chaque replay)
    ✅ state_size=8, action_size=6 (aligné avec env v2)
    ✅ Réseau plus profond (24→64 neurones)
    ✅ Gradient clipping pour stabilité
    ✅ Double DQN : target network pour sélection d'action
    """

    def __init__(self, state_size=8, action_size=6):
        self.state_size   = state_size
        self.action_size  = action_size

        # Mémoire de replay
        self.memory       = deque(maxlen=5000)

        # Hyperparamètres
        self.gamma         = 0.95       # discount factor
        self.epsilon       = 1.0        # exploration initiale
        self.epsilon_min   = 0.01
        self.epsilon_decay = 0.998
        self.learning_rate = 0.001

        # Réseaux
        self.model        = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

        # ✅ FIX : optimizer créé UNE SEULE FOIS ici (pas dans replay)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def _build_model(self):
        """Réseau plus profond : 8 → 64 → 64 → 6"""
        return nn.Sequential(
            nn.Linear(self.state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_size)
        )

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """Epsilon-greedy : exploration vs exploitation"""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.model(state_t)
        return int(torch.argmax(q_values).item())

    def replay(self, batch_size=64):
        """
        Entraînement sur un minibatch.
        
        ✅ FIX CRITIQUE v1 : l'optimizer n'est plus recréé à chaque itération.
           Dans v1, Adam recréé à chaque sample = pas de mémoire de momentum
           = apprentissage très inefficace.
        
        ✅ Double DQN : on utilise le target_model pour évaluer la valeur
           de l'action choisie par le model principal → moins de surestimation.
        """
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)

        states      = torch.FloatTensor([e[0] for e in minibatch])
        actions     = torch.LongTensor([e[1] for e in minibatch])
        rewards     = torch.FloatTensor([e[2] for e in minibatch])
        next_states = torch.FloatTensor([e[3] for e in minibatch])
        dones       = torch.FloatTensor([e[4] for e in minibatch])

        # Q-values actuelles
        q_current = self.model(states)

        # Double DQN : target = r + γ * Q_target(s', argmax Q_main(s'))
        with torch.no_grad():
            best_actions = self.model(next_states).argmax(dim=1)
            q_next       = self.target_model(next_states)
            q_next_best  = q_next.gather(1, best_actions.unsqueeze(1)).squeeze(1)

        targets = rewards + self.gamma * q_next_best * (1 - dones)

        # Construire les targets complets (copie + update de l'action choisie)
        q_targets = q_current.clone().detach()
        for i in range(batch_size):
            q_targets[i][actions[i]] = targets[i]

        # Backprop
        self.optimizer.zero_grad()
        loss = self.criterion(q_current, q_targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # gradient clipping
        self.optimizer.step()

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss.item()

    def save(self, filename):
        torch.save({
            'model_state': self.model.state_dict(),
            'epsilon': self.epsilon,
        }, filename)
        print(f"[AGENT] Modèle sauvegardé : {filename}")

    def load(self, filename):
        checkpoint = torch.load(filename)
        self.model.load_state_dict(checkpoint['model_state'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon_min)
        self.update_target_model()
        print(f"[AGENT] Modèle chargé : {filename} (epsilon={self.epsilon:.4f})")


# ── Test rapide ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    agent = DQNAgent(state_size=8, action_size=6)
    print("✅ Agent DQN v2 créé")
    print(f"   State size  : {agent.state_size}")
    print(f"   Action size : {agent.action_size}")
    print(f"   Epsilon     : {agent.epsilon}")
    print(f"   Réseau      : {agent.model}")