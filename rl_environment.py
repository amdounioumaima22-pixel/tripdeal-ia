import numpy as np
import pandas as pd
import random

class NegotiationEnv:
    """
    Environnement de négociation — VERSION PRO
    Initialisé depuis les vraies données CSV (2000 négociations réelles)
    
    AMÉLIORATIONS vs v1 :
    - Actions étendues de 3 → 6 (alignées avec le dataset)
    - State étendu de 6 → 8 (reaction_client + nb_personnes)
    - Initialisation par échantillonnage du CSV réel
    - Reward function calée sur les rewards observés dans les données
    - Modèle client probabiliste (pas juste un seuil fixe)
    """

    # ── Mapping actions (aligné avec action_agent du CSV) ──────────────────
    ACTION_NAMES = {
        0: "reduire_5_pct",
        1: "proposer_hotels",
        2: "proposer_transport",
        3: "retirer_excursion",
        4: "retirer_assurance",
        5: "refuser_negociation",
    }

    # Effet sur le prix pour chaque action
    ACTION_PRICE_EFFECT = {
        0: 0.95,   # -5%
        1: 0.97,   # -3%  (hôtel moins cher)
        2: 0.98,   # -2%  (transport moins cher)
        3: 0.97,   # -3%  (retrait excursion)
        4: 0.98,   # -2%  (retrait assurance)
        5: 1.00,   # pas de changement → fin immédiate
    }

    SAISON_MAP  = {'basse': 0, 'moyenne': 1, 'haute': 2}
    REACTION_MAP = {'trop_cher': 0, 'accepte': 1}

    def __init__(self, tours_path='data/tours_negociation.csv'):
        # ── Chargement et pré-traitement du dataset ──────────────────────────
        self.df = pd.read_csv(tours_path)
        self.df['saison_enc']       = self.df['saison'].map(self.SAISON_MAP)
        self.df['reaction_enc']     = self.df['reaction_client'].map(self.REACTION_MAP)

        # On garde uniquement le PREMIER tour de chaque négociation
        # → état de départ réaliste pour chaque épisode
        self.start_states = (
            self.df[self.df['tour'] == 1]
            .drop_duplicates(subset='negociation_id')
            .reset_index(drop=True)
        )

        print(f"[ENV] Dataset chargé : {len(self.start_states)} négociations de départ")
        print(f"[ENV] Destinations   : {self.df['destination'].unique().tolist()}")
        print(f"[ENV] Saisons        : {self.df['saison'].unique().tolist()}")

        self.action_space = len(self.ACTION_NAMES)  # 6
        self.state_size   = 8  # voir get_state()

        self.reset()

    # ── Reset : échantillonnage d'une vraie négociation ──────────────────────
    def reset(self):
        """Démarre un épisode depuis une négociation réelle du CSV"""
        row = self.start_states.sample(1).iloc[0]

        self.negociation_id  = row['negociation_id']
        self.destination     = row['destination']
        self.prix_actuel     = float(row['prix_propose'])
        self.prix_plancher   = float(row['prix_plancher'])
        self.budget_client   = float(row['budget_client'])
        self.popularite      = float(row['popularite_dest'])
        self.saison_enc      = float(row['saison_enc'])
        self.nb_personnes    = float(row['nb_personnes'])
        self.reaction_client = 0.0   # trop_cher par défaut au 1er tour
        self.tour            = 0

        return self.get_state()

    # ── State : 8 features ───────────────────────────────────────────────────
    def get_state(self):
        """
        State vector (8 dimensions) :
          0 → tour normalisé (0-1)
          1 → prix actuel normalisé (/ prix_plancher)
          2 → marge_pct : (prix_actuel - plancher) / prix_actuel
          3 → budget_ratio : budget_client / prix_actuel
          4 → popularite destination (0-1)
          5 → saison encodée (0=basse, 1=moyenne, 2=haute) / 2
          6 → reaction_client (0=trop_cher, 1=accepte)
          7 → nb_personnes normalisé (/ 4)
        """
        marge_pct    = (self.prix_actuel - self.prix_plancher) / self.prix_actuel \
                       if self.prix_actuel > 0 else 0.0
        budget_ratio = self.budget_client / self.prix_actuel \
                       if self.prix_actuel > 0 else 0.0

        return np.array([
            self.tour / 10.0,
            self.prix_actuel / self.prix_plancher,
            marge_pct,
            budget_ratio,
            self.popularite,
            self.saison_enc / 2.0,
            self.reaction_client,
            self.nb_personnes / 4.0,
        ], dtype=np.float32)

    # ── Modèle client probabiliste ────────────────────────────────────────────
    def _client_accepts(self):
        """
        Le client accepte si le prix est proche de son budget.
        On ajoute un bruit gaussien pour rendre le comportement probabiliste
        (comme dans la vraie vie — un client n'est pas un seuil fixe).

        Observé dans le CSV : acceptation dès que prix ≈ budget_client ± 5%
        """
        gap = (self.prix_actuel - self.budget_client) / self.budget_client
        # Sigmoid inversée : plus le prix s'approche du budget, plus l'acceptation est probable
        prob_accept = 1 / (1 + np.exp(15 * gap))   # sigmoid sur le gap
        return np.random.rand() < prob_accept

    # ── Step : exécution d'une action ─────────────────────────────────────────
    def step(self, action):
        """
        Exécute une action et retourne (next_state, reward, done, info)

        Reward calée sur les valeurs observées dans le CSV :
          - Chaque concession : pénalité croissante selon le tour (-5.5 → -9.5)
          - Deal conclu avec marge > 0  : +12.5 à +3.0 selon marge préservée
          - Deal conclu sous le plancher : -2.0 (perte acceptée)
          - Refus immédiat              : -9.5
        """
        self.tour += 1
        reward = 0.0
        done   = False
        info   = {'action_name': self.ACTION_NAMES[action], 'deal_conclu': False}

        # ── Action 5 : Refuser ────────────────────────────────────────────────
        if action == 5:
            reward = -9.5
            done   = True
            info['deal_conclu'] = False
            return self.get_state(), reward, done, info

        # ── Actions 0-4 : concessions ─────────────────────────────────────────
        # Pénalité de base croissante avec le tour (tirée du CSV)
        base_penalty = -(5.5 + 0.5 * (self.tour - 1))   # -5.5, -6.0, -6.5 …
        reward       = base_penalty

        # Application de l'effet prix
        effect = self.ACTION_PRICE_EFFECT[action]
        self.prix_actuel = max(self.prix_actuel * effect, self.prix_plancher * 0.85)

        # ── Réaction client ────────────────────────────────────────────────────
        if self._client_accepts():
            self.reaction_client = 1.0
            done = True
            info['deal_conclu'] = True

            # Bonus selon la marge préservée (calé sur les rewards du CSV)
            marge_finale = (self.prix_actuel - self.prix_plancher) / self.prix_actuel
            if marge_finale >= 0.14:
                reward += 25.0   # 12.5 deal bonus + 12.5 marge excellente
            elif marge_finale >= 0.05:
                reward += 18.5   # deal rapide, bonne marge
            elif marge_finale >= 0.01:
                reward += 10.0   # deal avec petite marge
            else:
                reward += 5.5    # deal sous le plancher (-2.0 net)
        else:
            self.reaction_client = 0.0

        # ── Fin par timeout ────────────────────────────────────────────────────
        if self.tour >= 10:
            done = True
            if not info['deal_conclu']:
                reward -= 5.0   # pénalité négociation sans issue

        return self.get_state(), reward, done, info


# ── Test rapide ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    env = NegotiationEnv()
    print("\n" + "="*60)

    for episode in range(3):
        state = env.reset()
        print(f"\n🧳 Épisode {episode+1} | Destination: {env.destination}")
        print(f"   Prix départ: {env.prix_actuel:.0f}€ | Plancher: {env.prix_plancher:.0f}€ | Budget client: {env.budget_client:.0f}€")

        total_reward = 0
        done = False
        while not done:
            action = random.randint(0, 5)
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            print(f"   Tour {env.tour} → {info['action_name']:<25} | reward: {reward:+.1f} | prix: {env.prix_actuel:.0f}€")

        status = "✅ DEAL" if info['deal_conclu'] else "❌ ÉCHEC"
        print(f"   {status} | Reward total: {total_reward:.2f}")