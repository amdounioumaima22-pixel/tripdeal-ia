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
# CHARGEMENT DES MODÈLES
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
    'Paris, France':              1800,
    'Dubai, EAU':                 2600,
    'Istanbul, Turquie':          1200,
    'Marrakech, Maroc':           850,
    'La Mecque, Arabie Saoudite': 3500,
    'Rome, Italie':               1600,
    'Barcelone, Espagne':         1450,
    'Thailande':                  2400,
    
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
}

SAISONS_MOIS = {
    1: 'basse', 2: 'basse', 11: 'basse',
    3: 'moyenne', 4: 'moyenne', 5: 'moyenne',
    9: 'moyenne', 10: 'moyenne',
    6: 'haute', 7: 'haute', 8: 'haute', 12: 'haute',
}

# ══════════════════════════════════════════
# FONCTIONS UTILITAIRES
# ══════════════════════════════════════════

def get_saison():
    mois = datetime.datetime.now().month
    return SAISONS_MOIS.get(mois, 'moyenne')


# Messages pour les nouvelles actions de service
ALTERNATIVES_INFO = {
    'changer_hotel_5_4':  {'label': 'Hôtel 5★ → 4★',          'description': "Je remplace l'hôtel 5 étoiles par un 4 étoiles excellent — même confort, prix réduit."},
    'changer_hotel_4_3':  {'label': 'Hôtel 4★ → 3★',          'description': "Je propose un hôtel 3 étoiles très bien noté à la place du 4 étoiles."},
    'retirer_excursion':  {'label': 'Excursion retirée',        'description': "Je retire l'excursion du package — vous pouvez la réserver sur place si vous le souhaitez."},
    'changer_transport':  {'label': 'Van → Voiture standard',   'description': "Je remplace le van par une voiture standard — tout aussi confortable pour 2-3 personnes."},
    'retirer_assurance':  {'label': 'Assurance retirée',        'description': "Je retire l'assurance voyage du package — à souscrire séparément si souhaitée."},
}

def generer_message(action, prix_actuel, prix_precedent, destination, tour, prix_plancher):
    reduction = round(prix_precedent - prix_actuel) if prix_precedent else 0

    # Actions de baisse directe
    if action == 'reduire_5_pct':
        msgs = [
            f"Je comprends votre budget. Je peux faire un effort et vous proposer {destination} pour {prix_actuel} TND — une réduction de {reduction} TND !",
            f"Bonne nouvelle ! Je viens de revoir notre offre : {prix_actuel} TND. Économisez {reduction} TND !",
        ]
    elif action == 'reduire_3_pct':
        msgs = [
            f"Je fais un geste commercial pour vous : {prix_actuel} TND. C'est vraiment notre meilleure offre pour cette qualité.",
            f"Pour vous aider, je descends à {prix_actuel} TND. Difficile d'aller plus bas avec ce niveau de services !",
        ]
    elif action == 'reduire_2_pct':
        msgs = [
            f"Nous approchons de notre limite, mais je peux encore vous proposer {prix_actuel} TND. C'est exceptionnel !",
            f"Dernière concession possible sur le prix : {prix_actuel} TND.",
        ]
    # Actions alternatives de service
    elif action in ALTERNATIVES_INFO:
        alt = ALTERNATIVES_INFO[action]
        msgs = [
            f"Je comprends que le budget reste serré. Voici une alternative : {alt['description']} "
            f"Nouveau prix : {prix_actuel} TND (économie de {reduction} TND). Qu'en pensez-vous ?",
            f"Pour respecter votre budget, je vous propose : {alt['label']}. "
            f"Le prix passe à {prix_actuel} TND — une économie de {reduction} TND !",
        ]
    elif action == 'refuser_negociation':
        msgs = [
            f"Je suis vraiment désolé, {prix_actuel} TND est notre prix plancher absolu. "
            f"En dessous, nous ne pouvons garantir la qualité du service. C'est notre offre finale.",
        ]
    else:
        msgs = [f"Notre offre est de {prix_actuel} TND."]

    return random.choice(msgs)


@app.route('/negotiate', methods=['POST'])
def negotiate():
    try:
        data = request.get_json()

        destination    = data.get('destination', 'Paris, France')
        prix_actuel    = float(data.get('prix_actuel', 6000))
        prix_plancher  = float(data.get('prix_plancher', 4500))
        budget_client  = float(data.get('budget_client', 5000))
        marge_actuelle = float(data.get('marge_actuelle', 1500))
        tour           = int(data.get('tour', 1))
        saison         = get_saison()
        popularite     = POPULARITES.get(destination, 0.8)

        marge_pct  = (marge_actuelle / prix_actuel * 100) if prix_actuel > 0 else 0
        saison_enc = le_saison.transform([saison])[0]

        # Prédire l'action via le modèle ML
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

        # ── Correction : "aucune" impossible quand client dit "trop cher" ──
        if action == 'aucune':
            if   marge_pct > 20: action = 'reduire_5_pct'
            elif marge_pct > 12: action = 'reduire_3_pct'
            elif marge_pct > 8:  action = 'reduire_2_pct'
            elif marge_pct > 5:  action = 'changer_hotel_4_3'
            elif marge_pct > 3:  action = 'retirer_excursion'
            elif marge_pct > 2:  action = 'changer_transport'
            else:                action = 'refuser_negociation'

        # ── Calcul du nouveau prix selon l'action ──
        prix_precedent = prix_actuel
        reductions = {
            'reduire_5_pct':      0.95,
            'reduire_3_pct':      0.97,
            'reduire_2_pct':      0.98,
            'changer_hotel_5_4':  0.90,
            'changer_hotel_4_3':  0.92,
            'retirer_excursion':  0.95,
            'changer_transport':  0.97,
            'retirer_assurance':  0.98,
            'refuser_negociation': 1.0,
        }
        ratio = reductions.get(action, 1.0)
        nouveau_prix = max(round(prix_actuel * ratio), round(prix_plancher))

        # Message
        message = generer_message(action, nouveau_prix, prix_precedent, destination, tour, prix_plancher)

        # Info alternative si c'est une action de service
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
            'alternative_info': alternative_info,   # ← NOUVEAU
            'is_alternative':   action in ALTERNATIVES_INFO,  # ← NOUVEAU
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

# ══════════════════════════════════════════
# ROUTES API
# ══════════════════════════════════════════

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'message': 'TripDeal IA API est operationnelle'
    })


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

        dest_enc   = le_dest.transform([destination])[0]
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


@app.route('/destinations', methods=['GET'])
def get_destinations():
    return jsonify({
        'success':      True,
        'destinations': list(PRIX_PLANCHERS.keys()),
    })
'Thailande':   
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

        email = f"""Bonjour {client_name},

Nous avons le plaisir de confirmer votre reservation de voyage avec TripDeal.

Details de votre voyage :
- Destination : {destination}
- Nombre de personnes : {nb_personnes}
- Nombre de nuits : {nb_nuits}
- Prix negocie : {prix_final} TND
- Economie realisee : {economie} TND (negociation en {nb_tours} tour(s))

Notre equipe commerciale vous contactera dans les 24 heures pour finaliser les details.

Merci de votre confiance !

Cordialement,
L'equipe TripDeal"""

        return jsonify({ 'success': True, 'email': email })

    except Exception as e:
        return jsonify({ 'success': False, 'error': str(e) }), 400

if __name__ == '__main__':
    print("\nTripDeal IA API")
    print("URL : http://localhost:5000")
    print("Routes :")
    print("  GET  /health")
    print("  POST /predict-prix")
    print("  POST /negotiate")
    print("  GET  /destinations")
    print("-" * 40)
    port = int(os.environ.get('PORT', 10000))
    app.run(debug=False, host='0.0.0.0', port=5000)