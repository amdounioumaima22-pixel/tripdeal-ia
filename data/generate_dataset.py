import pandas as pd
import numpy as np
import random
import json

np.random.seed(42)
random.seed(42)

# ── Prix alignés avec Salesforce ──
DESTINATIONS = {
    'Paris, France':              {'prix_affiche': 2699,  'prix_plancher': 2100, 'popularite': 0.9},
    'Dubai, EAU':                 {'prix_affiche': 4800,  'prix_plancher': 3800, 'popularite': 0.85},
    'Istanbul, Turquie':          {'prix_affiche': 2800,  'prix_plancher': 2200, 'popularite': 0.8},
    'Marrakech, Maroc':           {'prix_affiche': 2499,  'prix_plancher': 1900, 'popularite': 0.75},
    'La Mecque, Arabie Saoudite': {'prix_affiche': 4500,  'prix_plancher': 3500, 'popularite': 0.95},
    'Rome, Italie':               {'prix_affiche': 2800,  'prix_plancher': 2200, 'popularite': 0.78},
    'Barcelone, Espagne':         {'prix_affiche': 2750,  'prix_plancher': 2150, 'popularite': 0.76},
    'Thailande':                  {'prix_affiche': 4000,  'prix_plancher': 3100, 'popularite': 0.82},
}

SAISONS = {
    'haute':   {'multiplicateur': 1.15, 'mois': [6, 7, 8, 12]},
    'moyenne': {'multiplicateur': 1.0,  'mois': [3, 4, 5, 9, 10]},
    'basse':   {'multiplicateur': 0.90, 'mois': [1, 2, 11]},
}

# Remise max 15%
REMISE_MAX_PCT = 0.15

# Phase 2 : alternatives services
SERVICES_ORDRE = [
    'proposer_hotels',
    'proposer_transport',
    'retirer_excursion',
    'retirer_assurance',
]

SERVICES_ECO = {
    'proposer_hotels':    0.08,
    'proposer_transport': 0.03,
    'retirer_excursion':  0.04,
    'retirer_assurance':  0.02,
}

def get_saison(mois):
    for saison, info in SAISONS.items():
        if mois in info['mois']:
            return saison
    return 'moyenne'

def calculer_reward(prix_actuel, prix_plancher, prix_affiche, deal_conclu, tour):
    if prix_actuel < prix_plancher:
        return -100
    if deal_conclu:
        marge_initiale = prix_affiche - prix_plancher
        marge_finale   = prix_actuel  - prix_plancher
        pct = marge_finale / marge_initiale if marge_initiale > 0 else 0
        if pct > 0.70:   reward = 10
        elif pct > 0.50: reward = 7
        elif pct > 0.30: reward = 5
        elif pct > 0.10: reward = 3
        else:            reward = 1
        if tour <= 2:    reward += 3
        elif tour <= 3:  reward += 1
    else:
        reward = -5
    reward -= tour * 0.5
    return round(reward, 2)

def simuler_negociation(prix_affiche, prix_plancher, budget_client,
                        destination_popularite, saison):

    historique    = []
    prix_actuel   = prix_affiche
    tour          = 0
    max_tours     = 9
    deal_conclu   = False
    prix_final    = 0

    # Limite 15% : jamais descendre sous ce prix
    prix_limite   = max(round(prix_affiche * (1 - REMISE_MAX_PCT)), prix_plancher)
    phase         = 1
    service_index = 0

    while tour < max_tours and not deal_conclu:
        tour += 1
        tolerance = random.uniform(0.02, 0.08)

        if prix_actuel <= budget_client * (1 + tolerance):
            deal_conclu     = True
            prix_final      = prix_actuel
            reaction_client = 'accepte'
            action_agent    = 'aucune'
            reward          = calculer_reward(prix_final, prix_plancher, prix_affiche, True, tour)

        else:
            reaction_client = 'trop_cher'

            # ── PHASE 1 : Baisse directe max 15% ──
            if phase == 1:
                nouveau_prix = round(prix_actuel * 0.95)
                if nouveau_prix <= prix_limite:
                    nouveau_prix = prix_limite
                    phase        = 2
                action_agent = 'reduire_5_pct'

            # ── PHASE 2 : Alternatives services ──
            elif phase == 2 and service_index < len(SERVICES_ORDRE):
                action_agent   = SERVICES_ORDRE[service_index]
                eco            = SERVICES_ECO[action_agent]
                nouveau_prix   = max(round(prix_actuel * (1 - eco)), prix_plancher)
                service_index += 1

            # ── PHASE 3 : Refus final ──
            else:
                action_agent = 'refuser_negociation'
                reward       = calculer_reward(prix_actuel, prix_plancher, prix_affiche, False, tour)
                historique.append({
                    'tour':            tour,
                    'prix_propose':    prix_actuel,
                    'prix_plancher':   prix_limite,
                    'budget_client':   budget_client,
                    'marge_actuelle':  round(prix_actuel - prix_limite),
                    'marge_pct':       round((prix_actuel - prix_limite) / prix_actuel * 100, 2),
                    'reaction_client': reaction_client,
                    'action_agent':    action_agent,
                    'reward':          reward,
                    'deal_conclu':     False,
                    'popularite_dest': destination_popularite,
                    'saison':          saison,
                    'phase':           3,
                })
                break

            reward      = calculer_reward(round(nouveau_prix), prix_plancher, prix_affiche, False, tour)
            prix_actuel = nouveau_prix

        historique.append({
            'tour':            tour,
            'prix_propose':    prix_actuel,
            'prix_plancher':   prix_limite,
            'budget_client':   budget_client,
            'marge_actuelle':  round(prix_actuel - prix_limite),
            'marge_pct':       round((prix_actuel - prix_limite) / prix_actuel * 100, 2),
            'reaction_client': reaction_client,
            'action_agent':    action_agent,
            'reward':          reward,
            'deal_conclu':     deal_conclu,
            'popularite_dest': destination_popularite,
            'saison':          saison,
            'phase':           phase if not deal_conclu else 1,
        })

    return historique, deal_conclu, prix_final if deal_conclu else 0

def generer_dataset(nb_negociations=2000):
    print(f"Generation de {nb_negociations} negociations...")
    all_tours        = []
    all_negociations = []

    for i in range(nb_negociations):
        destination  = random.choice(list(DESTINATIONS.keys()))
        dest_info    = DESTINATIONS[destination]
        nb_personnes = random.choice([1, 2, 2, 2, 3, 4, 4])
        nb_nuits     = random.choice([3, 5, 7, 7, 10, 14])
        mois         = random.randint(1, 12)
        saison       = get_saison(mois)
        mult         = SAISONS[saison]['multiplicateur']

        # Prix basés sur les vrais prix Salesforce avec variation saison
        prix_affiche  = round(dest_info['prix_affiche'] * mult)
        prix_plancher = round(dest_info['prix_plancher'] * mult)

        if random.random() < 0.60:
            budget_client = round(random.uniform(prix_plancher * 0.85, prix_affiche * 0.95))
        else:
            budget_client = round(random.uniform(prix_affiche * 0.95, prix_affiche * 1.20))

        historique, deal_conclu, prix_final = simuler_negociation(
            prix_affiche, prix_plancher, budget_client,
            dest_info['popularite'], saison
        )

        negociation = {
            'negociation_id':  i + 1,
            'destination':     destination,
            'nb_personnes':    nb_personnes,
            'nb_nuits':        nb_nuits,
            'mois':            mois,
            'saison':          saison,
            'nb_options':      3,
            'prix_plancher':   prix_plancher,
            'prix_affiche':    prix_affiche,
            'budget_client':   budget_client,
            'ratio_budget':    round(budget_client / prix_affiche, 3),
            'nb_tours':        len(historique),
            'deal_conclu':     deal_conclu,
            'prix_final':      prix_final,
            'marge_finale':    prix_final - prix_plancher if deal_conclu else 0,
            'popularite_dest': dest_info['popularite'],
            'reward_total':    sum(t['reward'] for t in historique),
        }
        all_negociations.append(negociation)

        for tour in historique:
            tour_copy = tour.copy()
            tour_copy['negociation_id'] = i + 1
            tour_copy['destination']    = destination
            tour_copy['nb_personnes']   = nb_personnes
            tour_copy['saison']         = saison
            all_tours.append(tour_copy)

        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{nb_negociations} negociations generees")

    df_negociations = pd.DataFrame(all_negociations)
    df_tours        = pd.DataFrame(all_tours)
    return df_negociations, df_tours

if __name__ == '__main__':
    df_nego, df_tours = generer_dataset(2000)
    df_nego.to_csv('data/negociations.csv',       index=False)
    df_tours.to_csv('data/tours_negociation.csv', index=False)

    print("\n" + "=" * 55)
    print("STATISTIQUES DU DATASET")
    print("=" * 55)
    print(f"Total negociations : {len(df_nego)}")
    print(f"Deals conclus      : {df_nego['deal_conclu'].sum()} ({df_nego['deal_conclu'].mean()*100:.1f}%)")
    print(f"Prix moyen affiche : {df_nego['prix_affiche'].mean():.0f} TND")
    deals = df_nego[df_nego['deal_conclu']]
    print(f"Prix moyen final   : {deals['prix_final'].mean():.0f} TND")
    print(f"Marge moyenne      : {deals['marge_finale'].mean():.0f} TND")
    print(f"Nb tours moyen     : {df_nego['nb_tours'].mean():.1f}")
    print(f"Total tours        : {len(df_tours)}")
    print("\nActions utilisees :")
    print(df_tours['action_agent'].value_counts().to_string())