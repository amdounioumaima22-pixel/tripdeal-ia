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


def generer_message(action, prix_actuel, prix_precedent,
                    destination, tour, prix_plancher):
    reduction = prix_precedent - prix_actuel if prix_precedent else 0

    messages = {
        'reduire_5_pct': [
            f"Je comprends votre budget. Je peux faire un effort "
            f"et vous proposer {destination} pour {prix_actuel} TND. "
            f"C'est une reduction de {reduction:.0f} TND !",
            f"Bonne nouvelle ! Je viens de revoir notre offre. "
            f"Je vous propose {prix_actuel} TND pour ce voyage — "
            f"economisez {reduction:.0f} TND !",
        ],
        'reduire_3_pct': [
            f"Je fais un geste commercial pour vous : {prix_actuel} TND. "
            f"C'est vraiment notre meilleure offre pour cette qualite.",
            f"Pour vous aider, je descends a {prix_actuel} TND. "
            f"Difficile d'aller plus bas avec ce niveau de services !",
        ],
        'reduire_2_pct': [
            f"Nous approchons de notre limite, mais je peux encore "
            f"vous proposer {prix_actuel} TND. C'est exceptionnel !",
            f"Derniere concession possible : {prix_actuel} TND. "
            f"Au-dela, je ne peux garantir la meme qualite.",
        ],
        'offre_alternative': [
            f"Je suis a ma limite, mais j'ai une idee : "
            f"en retirant une excursion, je peux vous faire {prix_actuel} TND. "
            f"Qu'en pensez-vous ?",
            f"Pour respecter votre budget, je propose {prix_actuel} TND "
            f"avec un hebergement legerement revu. Cela reste excellent !",
        ],
        'refuser_negociation': [
            f"Je suis vraiment desole, {prix_actuel} TND est notre prix "
            f"plancher. En dessous, nous ne pouvons garantir la qualite "
            f"du service. C'est notre meilleure offre finale.",
            f"Nous avons atteint notre limite absolue a {prix_actuel} TND. "
            f"Ce prix inclut tous vos services. C'est notre offre finale.",
        ],
        'aucune': [
            f"Excellente nouvelle ! Je peux vous confirmer ce voyage "
            f"pour {prix_actuel} TND. C'est un excellent choix !",
        ]
    }

    msgs = messages.get(action, [f"Notre offre est de {prix_actuel} TND."])
    return random.choice(msgs)


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

        # Prédire l'action
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

        # ── FIX DU BUG ──
        # Si client dit "trop_cher", "aucune" est impossible
        if action == 'aucune':
            if marge_pct > 20:
                action = 'reduire_5_pct'
            elif marge_pct > 12:
                action = 'reduire_3_pct'
            elif marge_pct > 6:
                action = 'reduire_2_pct'
            elif marge_pct > 2:
                action = 'offre_alternative'
            else:
                action = 'refuser_negociation'

        # Calculer le nouveau prix
        prix_precedent = prix_actuel

        if action == 'reduire_5_pct':
            nouveau_prix = prix_actuel * 0.95
        elif action == 'reduire_3_pct':
            nouveau_prix = prix_actuel * 0.97
        elif action == 'reduire_2_pct':
            nouveau_prix = prix_actuel * 0.98
        elif action == 'offre_alternative':
            nouveau_prix = prix_actuel * 0.99
        elif action == 'refuser_negociation':
            nouveau_prix = prix_actuel
        else:
            nouveau_prix = prix_actuel

        # Sécurité : jamais sous le plancher
        nouveau_prix = max(nouveau_prix, prix_plancher)
        nouveau_prix = round(nouveau_prix)

        # Générer le message
        message = generer_message(
            action, nouveau_prix, prix_precedent,
            destination, tour, prix_plancher
        )

        # Deal possible ?
        tolerance     = 0.05
        deal_possible = nouveau_prix <= budget_client * (1 + tolerance)

        return jsonify({
            'success':        True,
            'action':         action,
            'prix_precedent': round(prix_precedent),
            'nouveau_prix':   nouveau_prix,
            'prix_plancher':  round(prix_plancher),
            'message':        message,
            'deal_possible':  deal_possible,
            'reduction':      round(prix_precedent - nouveau_prix),
            'marge_restante': round(nouveau_prix - prix_plancher),
            'tour':           tour,
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/destinations', methods=['GET'])
def get_destinations():
    return jsonify({
        'success':      True,
        'destinations': list(PRIX_PLANCHERS.keys()),
    })


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
app.run(debug=False, host='0.0.0.0', port=port)