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
# CONFIGURATION
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

# Reductions baisse directe
REDUCTIONS_PRIX = {
    'reduire_5_pct': 0.95,
    'reduire_3_pct': 0.97,
    'reduire_2_pct': 0.98,
}

# Services proposes apres limite 15% (par tour)
SERVICES_PAR_TOUR = {
    4: 'proposer_hotels',
    5: 'proposer_transport',
    6: 'retirer_excursion',
    7: 'retirer_assurance',
}

# ══════════════════════════════════════════
# CHOIX PROPOSES AU CLIENT PAR SERVICE
# ══════════════════════════════════════════

CHOICES_DEFINITION = {
    'proposer_hotels': {
        'titre': "Choisissez votre type d hebergement :",
        'message': "Nous avons atteint notre limite de remise sur le prix. Mais je peux adapter votre hebergement ! Quel type preferez-vous ?",
        'choices': [
            {'id': 'hotel_5', 'label': 'Hotel 5 etoiles (actuel)', 'icon': 'H5', 'reduction': 0.0},
            {'id': 'hotel_4', 'label': 'Hotel 4 etoiles',          'icon': 'H4', 'reduction': 0.08},
            {'id': 'hotel_3', 'label': 'Hotel 3 etoiles',          'icon': 'H3', 'reduction': 0.15},
            {'id': 'airbnb',  'label': 'Airbnb / Appartement',     'icon': 'AB', 'reduction': 0.18},
        ]
    },
    'proposer_transport': {
        'titre': "Choisissez votre type de transport :",
        'message': "Je peux egalement adapter le transport pour reduire le cout. Quelle option vous convient ?",
        'choices': [
            {'id': 'van_vip', 'label': 'Van VIP (actuel)',    'icon': 'VIP', 'reduction': 0.0},
            {'id': 'voiture', 'label': 'Voiture standard',   'icon': 'VT',  'reduction': 0.03},
            {'id': 'bus',     'label': 'Bus / Taxi local',   'icon': 'BUS', 'reduction': 0.05},
        ]
    },
    'retirer_excursion': {
        'titre': "Souhaitez-vous retirer l excursion ?",
        'message': "Je peux retirer l excursion du package pour reduire le prix. Vous pourrez la reserver sur place si vous le souhaitez.",
        'choices': [
            {'id': 'garder_excursion',  'label': "Garder l excursion",  'icon': 'OUI', 'reduction': 0.0},
            {'id': 'retirer_excursion', 'label': "Retirer l excursion", 'icon': 'NON', 'reduction': 0.04},
        ]
    },
    'retirer_assurance': {
        'titre': "Souhaitez-vous retirer l assurance voyage ?",
        'message': "Derniere option : je peux retirer l assurance voyage. Attention, pensez a en souscrire une separement !",
        'choices': [
            {'id': 'garder_assurance',  'label': "Garder l assurance",  'icon': 'OUI', 'reduction': 0.0},
            {'id': 'retirer_assurance', 'label': "Retirer l assurance", 'icon': 'NON', 'reduction': 0.02},
        ]
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


def generer_message_baisse(action, prix_actuel, prix_precedent, destination):
    reduction = round(prix_precedent - prix_actuel)
    if action == 'reduire_5_pct':
        msgs = [
            f"Je comprends votre budget. Je peux faire un effort et vous proposer {destination} pour {prix_actuel} TND — une reduction de {reduction} TND !",
            f"Bonne nouvelle ! Je viens de revoir notre offre : {prix_actuel} TND. Economisez {reduction} TND !",
        ]
    else:
        msgs = [
            f"Je fais un geste commercial pour vous : {prix_actuel} TND. C est vraiment notre meilleure offre sur le prix.",
            f"Derniere concession possible sur le prix : {prix_actuel} TND. Economie de {reduction} TND !",
        ]
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

        # Predire l action via le modele ML
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

        # Correction action aucune
        if action == 'aucune':
            if marge_pct > 15:
                action = 'reduire_5_pct'
            elif tour in SERVICES_PAR_TOUR:
                action = SERVICES_PAR_TOUR[tour]
            else:
                action = 'refuser_negociation'

        # Forcer service apres limite 15%
        prix_min_15pct = round(prix_affiche_original * 0.85)
        if action in REDUCTIONS_PRIX:
            nouveau_prix_test = round(prix_actuel * REDUCTIONS_PRIX[action])
            nouveau_prix_test = max(nouveau_prix_test, round(prix_plancher), prix_min_15pct)
            if nouveau_prix_test >= round(prix_actuel):
                if tour in SERVICES_PAR_TOUR:
                    action = SERVICES_PAR_TOUR[tour]
                else:
                    action = 'refuser_negociation'

        # ── CAS 1 : Baisse directe du prix ──
        if action in REDUCTIONS_PRIX:
            prix_precedent = prix_actuel
            ratio          = REDUCTIONS_PRIX[action]
            nouveau_prix   = round(prix_actuel * ratio)
            nouveau_prix   = max(nouveau_prix, round(prix_plancher), prix_min_15pct)

            message       = generer_message_baisse(action, nouveau_prix, prix_precedent, destination)
            deal_possible = nouveau_prix <= budget_client * 1.05

            return jsonify({
                'success':        True,
                'action':         action,
                'has_choices':    False,
                'prix_precedent': round(prix_precedent),
                'nouveau_prix':   nouveau_prix,
                'prix_plancher':  round(prix_plancher),
                'message':        message,
                'deal_possible':  deal_possible,
                'reduction':      round(prix_precedent - nouveau_prix),
                'marge_restante': round(nouveau_prix - prix_plancher),
                'tour':           tour,
                'alternative_info': None,
                'is_alternative': False,
            })

        # ── CAS 2 : Proposer des choix de service ──
        if action in CHOICES_DEFINITION:
            choice_def = CHOICES_DEFINITION[action]
            choices    = []
            for c in choice_def['choices']:
                c_prix = round(prix_actuel * (1 - c['reduction']))
                c_prix = max(c_prix, round(prix_plancher))
                choices.append({
                    'id':          c['id'],
                    'label':       c['label'],
                    'icon':        c['icon'],
                    'economie':    round(prix_actuel - c_prix),
                    'nouveau_prix': c_prix,
                })

            return jsonify({
                'success':      True,
                'action':       action,
                'has_choices':  True,
                'choice_type':  action,
                'choix_titre':  choice_def['titre'],
                'choices':      choices,
                'prix_precedent': round(prix_actuel),
                'nouveau_prix': round(prix_actuel),
                'message':      choice_def['message'],
                'deal_possible': False,
                'reduction':    0,
                'marge_restante': round(prix_actuel - prix_plancher),
                'tour':         tour,
                'alternative_info': None,
                'is_alternative': False,
            })

        # ── CAS 3 : Refus final ──
        return jsonify({
            'success':        True,
            'action':         'refuser_negociation',
            'has_choices':    False,
            'prix_precedent': round(prix_actuel),
            'nouveau_prix':   round(prix_actuel),
            'prix_plancher':  round(prix_plancher),
            'message':        f"C est notre meilleure offre a {round(prix_actuel)} TND. Nous avons fait le maximum pour vous ! Souhaitez-vous confirmer votre reservation ?",
            'deal_possible':  False,
            'reduction':      0,
            'marge_restante': round(prix_actuel - prix_plancher),
            'tour':           tour,
            'alternative_info': None,
            'is_alternative': False,
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