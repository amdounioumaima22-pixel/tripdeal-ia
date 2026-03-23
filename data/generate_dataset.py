import pandas as pd
import numpy as np
import random
import json

# ══════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════

np.random.seed(42)
random.seed(42)

DESTINATIONS = {
    'Paris, France':              {'prix_base': 2500, 'prix_plancher': 1800, 'popularite': 0.9},
    'Dubai, EAU':                 {'prix_base': 3500, 'prix_plancher': 2600, 'popularite': 0.85},
    'Istanbul, Turquie':          {'prix_base': 1800, 'prix_plancher': 1200, 'popularite': 0.8},
    'Marrakech, Maroc':           {'prix_base': 1200, 'prix_plancher': 850,  'popularite': 0.75},
    'La Mecque, Arabie Saoudite': {'prix_base': 4500, 'prix_plancher': 3500, 'popularite': 0.95},
    'Rome, Italie':               {'prix_base': 2200, 'prix_plancher': 1600, 'popularite': 0.78},
    'Barcelone, Espagne':         {'prix_base': 2000, 'prix_plancher': 1450, 'popularite': 0.76},
    'Thaïlande':                  {'prix_base': 3200, 'prix_plancher': 2400, 'popularite': 0.82},
}

SAISONS = {
    'haute':   {'multiplicateur': 1.3,  'mois': [6, 7, 8, 12]},
    'moyenne': {'multiplicateur': 1.0,  'mois': [3, 4, 5, 9, 10]},
    'basse':   {'multiplicateur': 0.75, 'mois': [1, 2, 11]},
}

OPTIONS = {
    'vol':       {'prix_moyen': 850},
    'hotel_3':   {'prix_moyen': 60},
    'hotel_4':   {'prix_moyen': 120},
    'hotel_5':   {'prix_moyen': 250},
    'transport': {'prix_moyen': 45},
    'excursion': {'prix_moyen': 80},
    'assurance': {'prix_moyen': 45},
}

# ══════════════════════════════════════════
# FONCTIONS UTILITAIRES
# ══════════════════════════════════════════

def get_saison(mois):
    """Retourne la saison selon le mois"""
    for saison, info in SAISONS.items():
        if mois in info['mois']:
            return saison
    return 'moyenne'


def calculer_prix_package(destination, nb_personnes, nb_nuits, options, saison):
    """
    Calcule le prix total d'un package voyage
    Retourne : (prix_reel, prix_plancher, prix_affiche)
    
    - prix_reel    = coût réel pour l'agence
    - prix_plancher = prix_reel + marge minimale 15%
    - prix_affiche  = prix de départ négociation (marge 20-35%)
    """
    mult = SAISONS[saison]['multiplicateur']
    prix_total = 0

    # Vol obligatoire × nb_personnes
    prix_vol = OPTIONS['vol']['prix_moyen'] * mult * nb_personnes
    prix_total += prix_vol

    # Hébergement × nb_nuits
    if 'hotel_5' in options:
        prix_hotel = OPTIONS['hotel_5']['prix_moyen'] * nb_nuits * (nb_personnes / 2)
    elif 'hotel_4' in options:
        prix_hotel = OPTIONS['hotel_4']['prix_moyen'] * nb_nuits * (nb_personnes / 2)
    else:
        prix_hotel = OPTIONS['hotel_3']['prix_moyen'] * nb_nuits * (nb_personnes / 2)
    prix_total += prix_hotel

    # Transport local
    if 'transport' in options:
        prix_total += OPTIONS['transport']['prix_moyen'] * nb_personnes

    # Excursions
    if 'excursion' in options:
        prix_total += OPTIONS['excursion']['prix_moyen'] * nb_personnes

    # Assurance
    if 'assurance' in options:
        prix_total += OPTIONS['assurance']['prix_moyen'] * nb_personnes

    # Marge minimale agence = 15%
    prix_plancher = prix_total * 1.15

    # Prix affiché = prix_plancher + marge négociation (20-35%)
    marge_nego = random.uniform(0.20, 0.35)
    prix_affiche = prix_plancher * (1 + marge_nego)

    return round(prix_total), round(prix_plancher), round(prix_affiche)


def calculer_reward(prix_actuel, prix_plancher, prix_affiche, deal_conclu, tour):
    """
    Calcule la récompense pour l'agent RL
    
    RÈGLES :
    +10  = excellent deal (conservé 70%+ de la marge)
    +7   = très bon deal
    +5   = bon deal
    +3   = deal acceptable
    +1   = deal minimal
    -100 = INTERDIT (prix sous plancher = perte pour l'agence)
    -5   = client parti sans deal
    -0.5 = pénalité par tour (négociation longue)
    """

    # RÈGLE ABSOLUE : jamais sous le prix plancher
    if prix_actuel < prix_plancher:
        return -100

    if deal_conclu:
        marge_initiale = prix_affiche - prix_plancher
        marge_finale   = prix_actuel  - prix_plancher
        
        # % de marge conservée par rapport au départ
        pct = marge_finale / marge_initiale if marge_initiale > 0 else 0

        if pct > 0.70:
            reward = 10
        elif pct > 0.50:
            reward = 7
        elif pct > 0.30:
            reward = 5
        elif pct > 0.10:
            reward = 3
        else:
            reward = 1

        # Bonus si deal rapide
        if tour <= 2:
            reward += 3
        elif tour <= 3:
            reward += 1

    else:
        # Client parti sans deal = mauvais
        reward = -5

    # Pénalité par tour pour encourager efficacité
    reward -= tour * 0.5

    return round(reward, 2)


def simuler_negociation(prix_affiche, prix_plancher, budget_client,
                         destination_popularite, saison):

    historique  = []
    prix_actuel = prix_affiche
    tour        = 0
    max_tours   = 8
    deal_conclu = False
    prix_final  = 0

    alternatives_ordre = [
        'changer_hotel_5_4',
        'changer_hotel_4_3',
        'retirer_excursion',
        'changer_transport',
        'retirer_assurance',
    ]
    ALTERNATIVES_ECO = {
        'changer_hotel_5_4': 0.10,
        'changer_hotel_4_3': 0.08,
        'retirer_excursion':  0.05,
        'changer_transport':  0.03,
        'retirer_assurance':  0.02,
    }
    alt_index = 0

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
            marge_restante  = prix_actuel - prix_plancher
            marge_pct       = marge_restante / prix_actuel

            if tour <= 2 and marge_pct > 0.15:
                action_agent = 'reduire_5_pct'
                nouveau_prix = prix_actuel * 0.95

            elif tour == 3 and marge_pct > 0.10:
                action_agent = 'reduire_3_pct'
                nouveau_prix = prix_actuel * 0.97

            elif alt_index < len(alternatives_ordre) and marge_pct > 0.02:
                action_agent = alternatives_ordre[alt_index]
                nouveau_prix = prix_actuel * (1 - ALTERNATIVES_ECO[action_agent])
                alt_index   += 1

            elif marge_pct > 0.02:
                action_agent = 'reduire_2_pct'
                nouveau_prix = prix_actuel * 0.98

            else:
                action_agent = 'refuser_negociation'
                reward       = calculer_reward(prix_actuel, prix_plancher, prix_affiche, False, tour)
                historique.append({
                    'tour':            tour,
                    'prix_propose':    prix_actuel,
                    'prix_plancher':   prix_plancher,
                    'budget_client':   budget_client,
                    'marge_actuelle':  round(prix_actuel - prix_plancher),
                    'marge_pct':       round(marge_pct * 100, 2),
                    'reaction_client': reaction_client,
                    'action_agent':    action_agent,
                    'reward':          reward,
                    'deal_conclu':     False,
                    'popularite_dest': destination_popularite,
                    'saison':          saison,
                })
                break

            nouveau_prix = max(nouveau_prix, prix_plancher)
            reward       = calculer_reward(round(nouveau_prix), prix_plancher, prix_affiche, False, tour)
            prix_actuel  = round(nouveau_prix)

        historique.append({
            'tour':            tour,
            'prix_propose':    prix_actuel,
            'prix_plancher':   prix_plancher,
            'budget_client':   budget_client,
            'marge_actuelle':  round(prix_actuel - prix_plancher),
            'marge_pct':       round((prix_actuel - prix_plancher) / prix_actuel * 100, 2),
            'reaction_client': reaction_client,
            'action_agent':    action_agent,
            'reward':          reward,
            'deal_conclu':     deal_conclu,
            'popularite_dest': destination_popularite,
            'saison':          saison,
        })

    return historique, deal_conclu, prix_final if deal_conclu else 0
# ══════════════════════════════════════════
# GÉNÉRATION DU DATASET
# ══════════════════════════════════════════

def generer_dataset(nb_negociations=2000):
    """
    Génère nb_negociations négociations complètes.
    
    Retourne 2 DataFrames :
    ┌─────────────────────────────────────────────┐
    │ df_negociations → 1 ligne par négociation   │
    │ df_tours        → 1 ligne par tour          │
    └─────────────────────────────────────────────┘
    """

    print(f"🚀 Génération de {nb_negociations} négociations...")

    all_tours        = []
    all_negociations = []

    for i in range(nb_negociations):

        # ── Paramètres aléatoires ──
        destination = random.choice(list(DESTINATIONS.keys()))
        dest_info   = DESTINATIONS[destination]

        nb_personnes = random.choice([1, 2, 2, 2, 3, 4, 4])
        nb_nuits     = random.choice([3, 5, 7, 7, 10, 14])
        mois         = random.randint(1, 12)
        saison       = get_saison(mois)

        # Options choisies
        options = ['hotel_4']
        if random.random() > 0.4: options.append('transport')
        if random.random() > 0.5: options.append('excursion')
        if random.random() > 0.6: options.append('assurance')
        if random.random() > 0.7:
            options = [o.replace('hotel_4', 'hotel_5') for o in options]

        # Calculer les prix
        prix_reel, prix_plancher, prix_affiche = calculer_prix_package(
            destination, nb_personnes, nb_nuits, options, saison
        )

        # Budget client
        # 60% ont un budget insuffisant → ils négocient
        if random.random() < 0.60:
            budget_client = round(random.uniform(
                prix_plancher * 0.85,
                prix_affiche  * 0.95
            ))
        else:
            budget_client = round(random.uniform(
                prix_affiche * 0.95,
                prix_affiche * 1.20
            ))

        # Simuler la négociation
        historique, deal_conclu, prix_final = simuler_negociation(
            prix_affiche, prix_plancher, budget_client,
            dest_info['popularite'], saison
        )

        # Résumé de la négociation
        negociation = {
            'negociation_id':  i + 1,
            'destination':     destination,
            'nb_personnes':    nb_personnes,
            'nb_nuits':        nb_nuits,
            'mois':            mois,
            'saison':          saison,
            'options':         json.dumps(options),
            'nb_options':      len(options),
            'prix_reel':       prix_reel,
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

        # Détail de chaque tour
        for tour in historique:
            tour_copy = tour.copy()
            tour_copy['negociation_id'] = i + 1
            tour_copy['destination']    = destination
            tour_copy['nb_personnes']   = nb_personnes
            tour_copy['saison']         = saison
            all_tours.append(tour_copy)

        if (i + 1) % 500 == 0:
            print(f"  ✅ {i + 1}/{nb_negociations} négociations générées")

    df_negociations = pd.DataFrame(all_negociations)
    df_tours        = pd.DataFrame(all_tours)

    return df_negociations, df_tours


# ══════════════════════════════════════════
# LANCER LA GÉNÉRATION
# ══════════════════════════════════════════

if __name__ == '__main__':

    # Générer le dataset
    df_nego, df_tours = generer_dataset(2000)

    # Sauvegarder les 2 fichiers CSV
    df_nego.to_csv('data/negociations.csv',       index=False)
    df_tours.to_csv('data/tours_negociation.csv', index=False)

    # ── STATISTIQUES ──
    print("\n" + "=" * 55)
    print("📊  STATISTIQUES DU DATASET")
    print("=" * 55)
    print(f"Total négociations       : {len(df_nego)}")
    print(f"Deals conclus            : {df_nego['deal_conclu'].sum()} "
          f"({df_nego['deal_conclu'].mean()*100:.1f}%)")
    print(f"Prix moyen affiché       : {df_nego['prix_affiche'].mean():.0f} TND")
    
    deals = df_nego[df_nego['deal_conclu']]
    print(f"Prix moyen final         : {deals['prix_final'].mean():.0f} TND")
    print(f"Marge moyenne conservée  : {deals['marge_finale'].mean():.0f} TND")
    print(f"Nb tours moyen           : {df_nego['nb_tours'].mean():.1f}")
    print(f"\nTotal tours enregistrés  : {len(df_tours)}")
    print(f"\n✅ Fichiers sauvegardés dans data/")
    print(f"   → data/negociations.csv")
    print(f"   → data/tours_negociation.csv")