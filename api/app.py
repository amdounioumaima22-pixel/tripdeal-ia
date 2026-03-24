from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd
import numpy as np
import datetime
import random
import os

app = Flask(__name__)
CORS(app)

# ══════════════════════════════════════════
# CHARGEMENT DES MODELES
# ══════════════════════════════════════════

print("Chargement des modeles...")

with open('models/model_prix.pkl',    'rb') as f: model_prix    = pickle.load(f)
with open('models/model_action.pkl',  'rb') as f: model_action  = pickle.load(f)
with open('models/le_dest.pkl',       'rb') as f: le_dest       = pickle.load(f)
with open('models/le_saison.pkl',     'rb') as f: le_saison     = pickle.load(f)
with open('models/le_action.pkl',     'rb') as f: le_action     = pickle.load(f)
with open('models/features_prix.pkl', 'rb') as f: features_prix = pickle.load(f)

print("OK - Modeles charges !")

# ══════════════════════════════════════════
# PRIX PLANCHERS = prix minimum agence
# ══════════════════════════════════════════

PRIX_PLANCHERS = {
    'Paris, France':              2100,
    'Dubai, EAU':                 3800,
    'Istanbul, Turquie':          2200,
    'Marrakech, Maroc':           1900,
    'La Mecque, Arabie Saoudite': 3500,
    'Rome, Italie':               2200,
    'Barcelone, Espagne':         2150,
    'Thailande':                  3100,
    'Thaïlande':                  3100,
}

POPULARITES = {
    'Paris, France':              0.9,
    'Dubai, EAU':                 0.85,
    'Istanbul, Turquie':          0.8,
    'Marrakech, Maroc':           0.75,
    'La Mecque, Arabie Saoudite': 0.95,
    'Rome, Italie':               0.78,
    'Barcelone, Espagne':         0.76,
    'Thailande':                  0.82,
    'Thaïlande':                  0.82,
}

SAISONS_MOIS = {
    1: 'basse', 2: 'basse', 11: 'basse',
    3: 'moyenne', 4: 'moyenne', 5: 'moyenne',
    9: 'moyenne', 10: 'moyenne',
    6: 'haute', 7: 'haute', 8: 'haute', 12: 'haute',
}

# ══════════════════════════════════════════
# ALTERNATIVES DE SERVICES
# ══════════════════════════════════════════

ALTERNATIVES_INFO = {
    'proposer_hotels': {
        'label':       'Choix hebergement',
        'description': "Je peux vous proposer differentes options d hebergement pour reduire le prix.",
    },
    'proposer_transport': {
        'label':       'Choix transport',
        'description': "Je peux changer le type de transport pour reduire le cout.",
    },
    'retirer_excursion': {
        'label':       'Excursion retiree',
        'description': "Je retire l excursion du package — vous pouvez la reserver sur place.",
    },
    'retirer_assurance': {
        'label':       'Assurance retiree',
        'description': "Je retire l assurance voyage du package — a souscrire separement.",
    },
    'changer_hotel_5_4': {
        'label':       'Hotel 5 etoiles vers 4 etoiles',
        'description': "Je remplace l hotel 5 etoiles par un 4 etoiles excellent.",
    },
    'changer_hotel_4_3': {
        'label':       'Hotel 4 etoiles vers 3 etoiles',
        'description': "Je propose un hotel 3 etoiles tres bien note.",
    },
    'changer_transport': {
        'label':       'Van vers Voiture standard',
        'description': "Je remplace le van par une voiture standard.",
    },
}

# ══════════════════════════════════════════
# FONCTIONS UTILITAIRES
# ══════════════════════════════════════════

def get_saison():
    mois = datetime.datetime.now().month
    return SAISONS_MOIS.get(mois, 'moyenne')


def normaliser_destination(destination):
    mapping = {'Thaïlande': 'Thailande'}
    return mapping.get(destination, destination)


def generer_message(action, prix_actuel, prix_precedent, destination, tour, prix_plancher):
    reduction = round(prix_precedent - prix_actuel) if prix_precedent else 0

    if action == 'reduire_5_pct':
        msgs = [
            f"Je comprends votre budget. Je peux faire un effort et vous proposer {destination} pour {prix_actuel} TND — une reduction de {reduction} TND !",
            f"Bonne nouvelle ! Je viens de revoir notre offre : {prix_actuel} TND. Economisez {reduction} TND !",
        ]
    elif action in ('reduire_3_pct', 'reduire_2_pct'):
        msgs = [
            f"Je fais un geste commercial pour vous : {prix_actuel} TND. C est vraiment notre meilleure offre.",
            f"Derniere concession possible sur le prix : {prix_actuel} TND.",
        ]
    elif action == 'proposer_hotels':
        msgs = [
            f"Nous avons atteint notre limite de remise. Mais je peux vous proposer differentes options d hebergement. Nouveau prix : {prix_actuel} TND (economie de {reduction} TND) !",
        ]
    elif action == 'proposer_transport':
        msgs = [
            f"Je peux changer le type de transport pour vous faire economiser davantage. Avec transport standard : {prix_actuel} TND (economie de {reduction} TND) !",
        ]
    elif action == 'retirer_excursion':
        msgs = [
            f"Je retire l excursion du package — vous pouvez la reserver sur place si vous le souhaitez. Nouveau prix : {prix_actuel} TND (economie de {reduction} TND) !",
        ]
    elif action == 'retirer_assurance':
        msgs = [
            f"Je retire l assurance voyage du package. Nouveau prix : {prix_actuel} TND (economie de {reduction} TND). Pensez a souscrire une assurance separement !",
        ]
    elif action in ALTERNATIVES_INFO:
        alt = ALTERNATIVES_INFO[action]
        msgs = [
            f"Pour respecter votre budget, je vous propose : {alt['label']}. Le prix passe a {prix_actuel} TND — economie de {reduction} TND !",
        ]
    elif action == 'refuser_negociation':
        msgs = [
            f"Je suis vraiment desole, {prix_actuel} TND est notre prix final. En dessous, nous ne pouvons garantir la qualite du service.",
        ]
    else:
        msgs = [f"Notre offre est de {prix_actuel} TND."]

    return random.choice(msgs)


# ══════════════════════════════════════════
# ROUTES API
# ══════════════════════════════════════════

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'message': 'TripDeal IA API est operationnelle'})


@app.route('/predict-prix', methods=['POST'])
def predict_prix():
    try:
        data = request.get_json()

        destination  = data.get('destination', 'Paris, France')
        nb_personnes = int(data.get('nb_personnes', 2))
        nb_nuits     = int(data.get('nb_nuits', 7))
        budget       = float(data.get('budget_client', 5000))
        prix_affiche = float(data.get('prix_affiche', 6000))
        nb_options   = int(data.get('nb_options', 3))

        prix_plancher = PRIX_PLANCHERS.get(destination, 2000)
        popularite    = POPULARITES.get(destination, 0.8)
        saison        = get_saison()

        dest_clean = normaliser_destination(destination)
        try:
            dest_enc = le_dest.transform([dest_clean])[0]
        except Exception:
            dest_enc = le_dest.transform(['Paris, France'])[0]

        saison_enc = le_saison.transform([saison])[0]

        features = pd.DataFrame([{
            'destination_enc': dest_enc,
            'nb_personnes':    nb_personnes,
            'nb_nuits':        nb_nuits,
            'saison_enc':      saison_enc,
            'nb_options':      nb_options,
            'prix_affiche':    prix_affiche,
            'prix_plancher':   prix_plancher,
            'budget_client':   budget,
            'ratio_budget':    budget / prix_affiche,
            'popularite_dest': popularite,
        }])

        prix_optimal = float(model_prix.predict(features)[0])
        prix_optimal = max(prix_optimal, prix_plancher * 1.05)

        return jsonify({
            'success':        True,
            'prix_optimal':   round(prix_optimal),
            'prix_plancher':  prix_plancher,
            'prix_affiche':   prix_affiche,
            'marge_initiale': round(prix_affiche - prix_plancher),
            'economie':       round(prix_affiche - prix_optimal),
            'saison':         saison,
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/negotiate', methods=['POST'])
def negotiate():
    try:
        data = request.get_json()

        destination           = data.get('destination', 'Paris, France')
        prix_actuel           = float(data.get('prix_actuel', 6000))
        prix_plancher         = float(data.get('prix_plancher', 4500))
        budget_client         = float(data.get('budget_client', 5000))
        marge_actuelle        = float(data.get('marge_actuelle', 1500))
        tour                  = int(data.get('tour', 1))
        prix_affiche_original = float(data.get('prix_affiche_original', prix_actuel))
        saison                = get_saison()
        popularite            = POPULARITES.get(destination, 0.8)

        marge_pct  = (marge_actuelle / prix_actuel * 100) if prix_actuel > 0 else 0
        saison_enc = le_saison.transform([saison])[0]

        features_action = pd.DataFrame([{
            'tour':            tour,
            'prix_propose':    prix_actuel,
            'prix_plancher':   prix_plancher,
            'budget_client':   budget_client,
            'marge_actuelle':  marge_actuelle,
            'marge_pct':       marge_pct,
            'popularite_dest': popularite,
            'saison_enc':      saison_enc,
        }])

        action_enc = model_action.predict(features_action)[0]
        action     = le_action.inverse_transform([action_enc])[0]

        if action == 'aucune':
            if marge_pct > 15:
                action = 'reduire_5_pct'
            elif marge_pct > 8:
                action = 'proposer_hotels'
            elif marge_pct > 5:
                action = 'proposer_transport'
            elif marge_pct > 3:
                action = 'retirer_excursion'
            elif marge_pct > 1:
                action = 'retirer_assurance'
            else:
                action = 'refuser_negociation'

        prix_precedent = prix_actuel

        reductions = {
            'reduire_5_pct':       0.95,
            'reduire_3_pct':       0.97,
            'reduire_2_pct':       0.98,
            'proposer_hotels':     0.92,
            'proposer_transport':  0.97,
            'retirer_excursion':   0.96,
            'retirer_assurance':   0.98,
            'changer_hotel_5_4':   0.92,
            'changer_hotel_4_3':   0.94,
            'changer_transport':   0.97,
            'refuser_negociation': 1.0,
        }

        ratio        = reductions.get(action, 1.0)
        nouveau_prix = round(prix_actuel * ratio)

        # LIMITE 1 : jamais sous le prix plancher
        nouveau_prix = max(nouveau_prix, round(prix_plancher))

        # LIMITE 2 : reduction totale max 15% du prix original
        prix_min_15pct = round(prix_affiche_original * 0.85)
        nouveau_prix   = max(nouveau_prix, prix_min_15pct)

        # Si prix bloque -> refus final
        if nouveau_prix >= round(prix_actuel) and action != 'refuser_negociation':
            action       = 'refuser_negociation'
            nouveau_prix = round(prix_actuel)

        message = generer_message(action, nouveau_prix, prix_precedent, destination, tour, prix_plancher)

        alternative_info = None
        if action in ALTERNATIVES_INFO:
            alternative_info = {
                'label':       ALTERNATIVES_INFO[action]['label'],
                'description': ALTERNATIVES_INFO[action]['description'],
                'economie':    round(prix_precedent - nouveau_prix),
            }

        tolerance     = 0.05
        deal_possible = nouveau_prix <= budget_client * (1 + tolerance)

        return jsonify({
            'success':          True,
            'action':           action,
            'prix_precedent':   round(prix_precedent),
            'nouveau_prix':     nouveau_prix,
            'prix_plancher':    round(prix_plancher),
            'message':          message,
            'deal_possible':    deal_possible,
            'reduction':        round(prix_precedent - nouveau_prix),
            'marge_restante':   round(nouveau_prix - prix_plancher),
            'tour':             tour,
            'alternative_info': alternative_info,
            'is_alternative':   action in ALTERNATIVES_INFO,
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/destinations', methods=['GET'])
def get_destinations():
    return jsonify({'success': True, 'destinations': list(PRIX_PLANCHERS.keys())})


@app.route('/generate-email', methods=['POST'])
def generate_email():
    try:
        data         = request.get_json()
        destination  = data.get('destination', '')
        client_name  = data.get('client_name', 'Client')
        prix_final   = data.get('prix_final', 0)
        prix_affiche = data.get('prix_affiche', 0)
        nb_personnes = data.get('nb_personnes', 2)
        nb_nuits     = data.get('nb_nuits', 7)
        nb_tours     = data.get('nb_tours', 1)
        economie     = round(prix_affiche - prix_final)

        email = (
            f"Bonjour {client_name},\n\n"
            f"Nous avons le plaisir de confirmer votre reservation de voyage avec TripDeal.\n\n"
            f"Details de votre voyage :\n"
            f"- Destination : {destination}\n"
            f"- Nombre de personnes : {nb_personnes}\n"
            f"- Nombre de nuits : {nb_nuits}\n"
            f"- Prix negocie : {prix_final} TND\n"
            f"- Economie realisee : {economie} TND (negociation en {nb_tours} tour(s))\n\n"
            f"Notre equipe commerciale vous contactera dans les 24 heures.\n\n"
            f"Merci de votre confiance !\n\n"
            f"Cordialement,\n"
            f"L equipe TripDeal"
        )

        return jsonify({'success': True, 'email': email})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)