from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os
import pandas as pd
from groq import Groq

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl_environment import NegotiationEnv
from dqn_agent import DQNAgent

app = Flask(__name__)
CORS(app)

# ====================== CHARGEMENT ======================
base_dir    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
models_path = os.path.join(base_dir, 'models')
data_path   = os.path.join(base_dir, 'data')
csv_path    = os.path.join(data_path, 'tours_negociation.csv')

print("Chargement de l'Agent RL (DQN)...")
rl_agent = DQNAgent(state_size=8, action_size=6)
rl_agent.load(os.path.join(models_path, 'dqn_agent.pth'))
rl_agent.epsilon = 0.0
print("✅ Agent DQN chargé — mode décision autonome")

df_tours = pd.read_csv(csv_path)
SAISON_MAP = {'basse': 0, 'moyenne': 1, 'haute': 2}
df_tours['saison_enc'] = df_tours['saison'].map(SAISON_MAP)
print(f"✅ CSV chargé — {len(df_tours)} négociations disponibles")

GROQ_API_KEY = os.environ.get('GROQ_API_KEY', 'mets_ta_cle_ici')
groq_client  = Groq(api_key=GROQ_API_KEY)
print("✅ Client Groq initialisé")


# ====================== LABELS ACTIONS ======================
ACTION_LABELS = {
    0: "baisser le prix de 5%",
    1: "proposer un hôtel alternatif moins cher",
    2: "proposer un transport moins cher",
    3: "retirer l'excursion du forfait",
    4: "retirer l'assurance du forfait",
    5: "refuser de continuer la négociation",
}


# ====================== CALCUL DÉTAIL CHANGEMENT ======================
def calculer_detail_changement(action, prix_avant, prix_apres,
                                hebergement=None, transport=None,
                                excursions=None, extras=None):
    """
    Retourne une liste de lignes expliquant ce qui a changé selon l'action DQN.
    Chaque ligne = { 'icone': '🏨', 'texte': 'Hôtel réduit : ...', 'montant': '-80 TND' }
    """
    reduction = round(prix_avant - prix_apres)
    details   = []

    if action == 0:
        # Réduction globale de 5%
        details.append({
            'icone':   '💰',
            'texte':   'Remise commerciale appliquée sur le forfait',
            'montant': f'-{reduction} TND'
        })

    elif action == 1:
        # Hôtel alternatif moins cher
        nom_heb = hebergement if hebergement else "l'hébergement"
        details.append({
            'icone':   '🏨',
            'texte':   f'Hébergement optimisé : {nom_heb}',
            'montant': f'-{reduction} TND'
        })

    elif action == 2:
        # Transport moins cher
        nom_tr = transport if transport else "le transport"
        details.append({
            'icone':   '🚌',
            'texte':   f'Transport optimisé : {nom_tr}',
            'montant': f'-{reduction} TND'
        })

    elif action == 3:
        # Retrait excursion
        if excursions:
            noms = excursions if isinstance(excursions, str) else ', '.join(excursions)
            details.append({
                'icone':   '🗺️',
                'texte':   f'Excursion retirée : {noms}',
                'montant': f'-{reduction} TND'
            })
        else:
            details.append({
                'icone':   '🗺️',
                'texte':   'Excursion optionnelle retirée du forfait',
                'montant': f'-{reduction} TND'
            })

    elif action == 4:
        # Retrait assurance / extra
        if extras:
            noms = extras if isinstance(extras, str) else ', '.join(extras)
            details.append({
                'icone':   '🛡️',
                'texte':   f'Service retiré : {noms}',
                'montant': f'-{reduction} TND'
            })
        else:
            details.append({
                'icone':   '🛡️',
                'texte':   'Service optionnel retiré du forfait',
                'montant': f'-{reduction} TND'
            })

    elif action == 5:
        # Refus — pas de changement
        details.append({
            'icone':   '⚠️',
            'texte':   'Prix plancher atteint — aucune réduction possible',
            'montant': '0 TND'
        })

    return details


# ====================== GÉNÉRATION MESSAGE LLM ======================
def generer_message_llm(action, prix_actuel, budget_client,
                         destination, tour, deal_conclu,
                         hebergement=None, transport=None,
                         excursions=None, extras=None):
    try:
        action_label = ACTION_LABELS.get(action, "faire une offre")

        # Contexte éléments choisis pour Groq
        contexte_elements = ""
        if hebergement: contexte_elements += f"\n- Hébergement choisi : {hebergement}"
        if transport:   contexte_elements += f"\n- Transport choisi : {transport}"
        if excursions:  contexte_elements += f"\n- Excursions : {excursions}"
        if extras:      contexte_elements += f"\n- Services additionnels : {extras}"

        if deal_conclu:
            situation = f"Le client a accepté l'offre à {round(prix_actuel)} TND."
        else:
            situation = f"Le client n'a pas encore accepté. Tour numéro {tour}."

        prompt = f"""Tu es un agent commercial de l'agence de voyage TripDeal en Tunisie.
Tu négocies le prix d'un voyage pour la destination : {destination}.

Éléments du forfait :{contexte_elements if contexte_elements else ' forfait standard'}

Situation actuelle :
- Prix proposé : {round(prix_actuel)} TND
- Budget du client : {round(budget_client)} TND
- Tour de négociation : {tour}
- Action décidée : {action_label}
- {situation}

Génère UN SEUL message commercial court (2 phrases maximum), chaleureux et professionnel en français.
Le message doit correspondre exactement à l'action décidée et mentionner l'élément concerné si pertinent.
Ne mentionne pas de chiffres autres que le prix proposé.
Réponds uniquement avec le message, sans introduction ni explication."""

        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"[GROQ] Erreur : {e} — utilisation du message par défaut")
        fallback = {
            0: f"J'ai pu réduire le prix à {round(prix_actuel)} TND. Qu'en pensez-vous ?",
            1: "Je peux vous proposer un hébergement plus adapté à votre budget.",
            2: "J'ai une option transport moins chère qui peut réduire le coût total.",
            3: f"En retirant l'excursion, je descends à {round(prix_actuel)} TND.",
            4: f"En retirant le service additionnel, je vous propose {round(prix_actuel)} TND.",
            5: f"Je suis désolé, {round(prix_actuel)} TND est notre offre finale.",
        }
        return fallback.get(action, f"Notre offre est à {round(prix_actuel)} TND.")


# ====================== ROUTE PRINCIPALE ======================
@app.route('/negotiate', methods=['POST'])
def negotiate():
    try:
        data        = request.get_json()
        destination = data.get('destination', None)

        # ── Éléments choisis par le client (nouveaux paramètres) ──
        hebergement = data.get('hebergement', None)   # ex: "Djerba Palace ★★★★"
        transport   = data.get('transport',   None)   # ex: "Navette privée"
        excursions  = data.get('excursions',  None)   # ex: "Safari désert, Medina Tour"
        extras      = data.get('extras',      None)   # ex: "Assurance annulation"

        if destination:
            rows = df_tours[df_tours['destination'] == destination]
        else:
            rows = df_tours

        if rows.empty:
            return jsonify({'success': False, 'error': f'Destination inconnue : {destination}'}), 400

        row = rows.iloc[0]

        env = NegotiationEnv(tours_path=csv_path)
        env.budget_client   = float(data.get('budget_client',   row['budget_client']))
        env.nb_personnes    = float(data.get('nb_personnes',    row['nb_personnes']))
        env.saison_enc      = float(data.get('saison_enc',      row['saison_enc']))
        env.tour            = int(data.get('tour',              0))
        env.reaction_client = float(data.get('reaction_client', 0.0))
        env.prix_actuel     = float(data.get('prix_actuel',     row['prix_propose']))
        env.prix_plancher   = float(row['prix_plancher'])
        env.popularite      = float(row['popularite_dest'])

        prix_avant = env.prix_actuel  # ← on sauvegarde avant le step

        # ── Deal immédiat si budget suffisant ──
        if env.budget_client >= env.prix_actuel:
            message = generer_message_llm(
                action=-1, prix_actuel=env.prix_actuel,
                budget_client=env.budget_client,
                destination=destination or row['destination'],
                tour=1, deal_conclu=True,
                hebergement=hebergement, transport=transport,
                excursions=excursions, extras=extras
            )
            return jsonify({
                'success': True, 'action': -1,
                'action_name': 'deal_immediat',
                'message': message,
                'nouveau_prix': round(env.prix_actuel),
                'prix_plancher': round(env.prix_plancher),
                'detail_changement': [],
                'has_choices': False,
                'tour': 1, 'reward': 25.0,
                'done': True, 'deal_conclu': True,
            })

        # ── Agent DQN décide ──
        state  = env.get_state()
        action = rl_agent.act(state)
        next_state, reward, done, info = env.step(action)

        prix_apres = env.prix_actuel  # ← après le step

        # ── Calcul détail changement ──
        detail_changement = calculer_detail_changement(
            action=action,
            prix_avant=prix_avant,
            prix_apres=prix_apres,
            hebergement=hebergement,
            transport=transport,
            excursions=excursions,
            extras=extras
        )

        # ── Message Groq enrichi ──
        message = generer_message_llm(
            action=action, prix_actuel=env.prix_actuel,
            budget_client=env.budget_client,
            destination=destination or row['destination'],
            tour=env.tour,
            deal_conclu=info.get('deal_conclu', False),
            hebergement=hebergement, transport=transport,
            excursions=excursions, extras=extras
        )

        return jsonify({
            'success':            True,
            'action':             int(action),
            'action_name':        info.get('action_name', ''),
            'message':            message,
            'nouveau_prix':       round(env.prix_actuel),
            'prix_plancher':      round(env.prix_plancher),
            'detail_changement':  detail_changement,  # ← NOUVEAU
            'has_choices':        action in [1, 2],
            'tour':               int(env.tour),
            'reward':             float(reward),
            'done':               bool(done),
            'deal_conclu':        bool(info.get('deal_conclu', False)),
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'message': 'TripDeal IA — Agent DQN + Groq LLM opérationnel'})


@app.route('/destinations', methods=['GET'])
def get_destinations():
    destinations = df_tours['destination'].unique().tolist()
    return jsonify({'success': True, 'destinations': destinations})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)