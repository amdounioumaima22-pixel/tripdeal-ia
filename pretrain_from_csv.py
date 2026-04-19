"""
pretrain_from_csv.py
────────────────────
Pré-entraîne l'agent DQN par imitation des trajectoires du CSV
AVANT le training RL normal.

Principe : Behavioral Cloning
  - On lit chaque tour du CSV comme (state, action)
  - On entraîne le réseau à prédire la bonne action (supervised)
  - Ensuite train_dqn.py affine avec le RL

Avantage : l'agent part d'une politique raisonnable,
pas d'une politique aléatoire → convergence 3x plus rapide.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import os
from dqn_agent import DQNAgent

# ── Mapping action texte → index (aligné avec ACTION_NAMES de l'env) ─────────
ACTION_MAP = {
    'reduire_5_pct':        0,
    'proposer_hotels':      1,
    'proposer_transport':   2,
    'retirer_excursion':    3,
    'retirer_assurance':    4,
    'refuser_negociation':  5,
    'aucune':               None,   # tour où le client accepte → skip
}

SAISON_MAP = {'basse': 0.0, 'moyenne': 0.5, 'haute': 1.0}

def build_state_from_row(row, tour_norm=None):
    """
    Reconstruit le state (8 dim) depuis une ligne du CSV.
    Même logique que NegotiationEnv.get_state()
    """
    prix_actuel   = float(row['prix_propose'])
    prix_plancher = float(row['prix_plancher'])
    budget_client = float(row['budget_client'])
    tour          = int(row['tour'])

    marge_pct    = (prix_actuel - prix_plancher) / prix_actuel if prix_actuel > 0 else 0.0
    budget_ratio = budget_client / prix_actuel if prix_actuel > 0 else 0.0
    reaction_enc = 0.0 if row['reaction_client'] == 'trop_cher' else 1.0
    saison_enc   = SAISON_MAP.get(row['saison'], 0.5)
    nb_personnes = float(row['nb_personnes'])

    return np.array([
        tour / 10.0,
        prix_actuel / prix_plancher,
        marge_pct,
        budget_ratio,
        float(row['popularite_dest']),
        saison_enc / 2.0,
        reaction_enc,
        nb_personnes / 4.0,
    ], dtype=np.float32)


def load_demonstrations(csv_path='data/tours_negociation.csv'):
    """Charge toutes les paires (state, action) exploitables du CSV"""
    df = pd.read_csv(csv_path)

    states  = []
    actions = []
    skipped = 0

    for _, row in df.iterrows():
        action_idx = ACTION_MAP.get(row['action_agent'])
        if action_idx is None:   # 'aucune' = client qui accepte → pas utile
            skipped += 1
            continue

        state = build_state_from_row(row)
        states.append(state)
        actions.append(action_idx)

    print(f"[PRETRAIN] Démonstrations chargées : {len(states)} tours")
    print(f"[PRETRAIN] Tours ignorés (acceptation) : {skipped}")
    return np.array(states, dtype=np.float32), np.array(actions, dtype=np.int64)


def pretrain(agent, states, actions, epochs=20, batch_size=128, lr=0.001):
    """
    Behavioral Cloning : entraîne le réseau à reproduire les actions du CSV.
    C'est de la classification supervisée sur le réseau Q de l'agent.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(agent.model.parameters(), lr=lr)

    n = len(states)
    print(f"\n[PRETRAIN] Début — {epochs} epochs, {n} exemples, batch={batch_size}")
    print("-" * 55)

    for epoch in range(epochs):
        # Shuffle
        idx = np.random.permutation(n)
        states_shuf  = states[idx]
        actions_shuf = actions[idx]

        total_loss   = 0.0
        correct      = 0
        nb_batches   = 0

        for start in range(0, n, batch_size):
            end = start + batch_size
            s_batch = torch.FloatTensor(states_shuf[start:end])
            a_batch = torch.LongTensor(actions_shuf[start:end])

            # Forward
            q_values = agent.model(s_batch)   # shape [B, 6]
            loss     = criterion(q_values, a_batch)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.model.parameters(), 1.0)
            optimizer.step()

            # Stats
            preds   = q_values.argmax(dim=1)
            correct += (preds == a_batch).sum().item()
            total_loss  += loss.item()
            nb_batches  += 1

        avg_loss = total_loss / nb_batches
        accuracy = correct / n * 100
        print(f"Epoch {epoch+1:>2}/{epochs} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.1f}%")

    # Sync target model avec les poids pré-entraînés
    agent.update_target_model()
    print("\n[PRETRAIN] ✅ Pré-entraînement terminé — target model synchronisé")


if __name__ == "__main__":
    # 1. Charger les démos
    states, actions = load_demonstrations('data/tours_negociation.csv')

    # 2. Créer l'agent
    agent = DQNAgent(state_size=8, action_size=6)
    # Epsilon réduit : l'agent part avec une bonne politique, moins d'exploration nécessaire
    agent.epsilon = 0.5

    # 3. Pré-entraîner par imitation
    pretrain(agent, states, actions, epochs=25, batch_size=128)

    # 4. Sauvegarder le checkpoint pré-entraîné
    os.makedirs("models", exist_ok=True)
    agent.save("models/dqn_agent_pretrained.pth")
    print("Checkpoint sauvegardé : models/dqn_agent_pretrained.pth")

    print("\n" + "=" * 55)
    print("Prochaine étape : lancer train_dqn.py")
    print("Il chargera ce checkpoint et affinera avec le RL.")
    print("=" * 55)