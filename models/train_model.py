import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (mean_absolute_error, r2_score,
                              classification_report, confusion_matrix,
                              ConfusionMatrixDisplay)

# ══════════════════════════════════════════
# CHARGEMENT ET PRÉPARATION DES DONNÉES
# ══════════════════════════════════════════

print("=" * 55)
print("🤖 ENTRAÎNEMENT DES MODÈLES ML")
print("=" * 55)

df = pd.read_csv('data/negociations.csv')
df_tours = pd.read_csv('data/tours_negociation.csv')

print(f"\n✅ Dataset chargé : {len(df)} négociations")

# ── Encodage des variables catégorielles ──
le_dest   = LabelEncoder()
le_saison = LabelEncoder()

df['destination_enc'] = le_dest.fit_transform(df['destination'])
df['saison_enc']      = le_saison.fit_transform(df['saison'])

print(f"✅ Encodage des variables catégorielles")
print(f"   Destinations : {list(le_dest.classes_)}")
print(f"   Saisons      : {list(le_saison.classes_)}")

# ══════════════════════════════════════════
# MODÈLE 1 : PRÉDICTION DU PRIX OPTIMAL
# But : prédire le prix final accepté
# ══════════════════════════════════════════

print("\n" + "─" * 55)
print("📈 MODÈLE 1 : Prédiction du Prix Optimal")
print("─" * 55)

# Garder seulement les deals conclus pour ce modèle
df_deals = df[df['deal_conclu'] == True].copy()
print(f"Négociations avec deal : {len(df_deals)}")

# Features (variables d'entrée)
features_prix = [
    'destination_enc',   # Quelle destination
    'nb_personnes',      # Combien de personnes
    'nb_nuits',          # Combien de nuits
    'saison_enc',        # Quelle saison
    'nb_options',        # Nombre d'options choisies
    'prix_affiche',      # Prix de départ
    'prix_plancher',     # Prix minimum
    'budget_client',     # Budget du client
    'ratio_budget',      # Ratio budget/prix affiché
    'popularite_dest',   # Popularité de la destination
]

# Target (variable à prédire)
target_prix = 'prix_final'

X_prix = df_deals[features_prix]
y_prix = df_deals[target_prix]

# Split train/test (80% train, 20% test)
X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(
    X_prix, y_prix, test_size=0.2, random_state=42
)

print(f"\nDonnées d'entraînement : {len(X_train_p)} exemples")
print(f"Données de test        : {len(X_test_p)} exemples")

# ── Entraînement ──
print("\n⏳ Entraînement en cours...")

model_prix = GradientBoostingRegressor(
    n_estimators   = 200,
    learning_rate  = 0.1,
    max_depth      = 4,
    min_samples_split = 5,
    random_state   = 42
)
model_prix.fit(X_train_p, y_train_p)

# ── Évaluation ──
y_pred_p = model_prix.predict(X_test_p)

mae = mean_absolute_error(y_test_p, y_pred_p)
r2  = r2_score(y_test_p, y_pred_p)

# Cross-validation
cv_scores = cross_val_score(model_prix, X_prix, y_prix,
                             cv=5, scoring='r2')

print(f"\n📊 RÉSULTATS MODÈLE 1 :")
print(f"   MAE  (erreur moyenne)  : {mae:.0f} TND")
print(f"   R²   (précision)       : {r2:.4f} ({r2*100:.1f}%)")
print(f"   R²   (cross-val 5)     : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ── Importance des features ──
feat_imp = pd.DataFrame({
    'feature':   features_prix,
    'importance': model_prix.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n🎯 Importance des variables :")
for _, row in feat_imp.iterrows():
    bar = '█' * int(row['importance'] * 50)
    print(f"   {row['feature']:<20} {bar} {row['importance']:.4f}")

# ══════════════════════════════════════════
# MODÈLE 2 : PRÉDICTION DE L'ACTION
# But : prédire la meilleure action à chaque tour
# ══════════════════════════════════════════

print("\n" + "─" * 55)
print("🎮 MODÈLE 2 : Prédiction de l'Action Optimale")
print("─" * 55)

# Encodage de la saison dans df_tours
df_tours['saison_enc'] = le_saison.transform(df_tours['saison'])

le_action = LabelEncoder()
df_tours['action_enc'] = le_action.fit_transform(df_tours['action_agent'])

print(f"Actions possibles : {list(le_action.classes_)}")

# Features pour prédire l'action
features_action = [
    'tour',              # Numéro du tour
    'prix_propose',      # Prix actuel proposé
    'prix_plancher',     # Prix minimum autorisé
    'budget_client',     # Budget du client
    'marge_actuelle',    # Marge restante en TND
    'marge_pct',         # Marge restante en %
    'popularite_dest',   # Popularité destination
    'saison_enc',        # Saison
]

X_action = df_tours[features_action]
y_action = df_tours['action_enc']

X_train_a, X_test_a, y_train_a, y_test_a = train_test_split(
    X_action, y_action, test_size=0.2, random_state=42
)

print(f"\nDonnées d'entraînement : {len(X_train_a)} tours")
print(f"Données de test        : {len(X_test_a)} tours")

# ── Entraînement ──
print("\n⏳ Entraînement en cours...")

model_action = RandomForestClassifier(
    n_estimators = 200,
    max_depth    = 8,
    random_state = 42,
    n_jobs       = -1
)
model_action.fit(X_train_a, y_train_a)

# ── Évaluation ──
y_pred_a = model_action.predict(X_test_a)
accuracy = (y_pred_a == y_test_a).mean()

print(f"\n📊 RÉSULTATS MODÈLE 2 :")
print(f"   Accuracy : {accuracy:.4f} ({accuracy*100:.1f}%)")
print(f"\n{classification_report(y_test_a, y_pred_a, target_names=le_action.classes_)}")

# ══════════════════════════════════════════
# VISUALISATIONS
# ══════════════════════════════════════════

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('🤖 Résultats des Modèles ML', fontsize=14, fontweight='bold')

# ── Graphique 1 : Prédictions vs Réalité ──
ax1 = axes[0, 0]
ax1.scatter(y_test_p, y_pred_p, alpha=0.4, color='#3b82f6', s=20)
min_val = min(y_test_p.min(), y_pred_p.min())
max_val = max(y_test_p.max(), y_pred_p.max())
ax1.plot([min_val, max_val], [min_val, max_val],
         'r--', linewidth=2, label='Prédiction parfaite')
ax1.set_xlabel('Prix réel (TND)')
ax1.set_ylabel('Prix prédit (TND)')
ax1.set_title(f'Modèle 1 : Prix Réel vs Prédit\nR² = {r2:.3f}',
              fontweight='bold')
ax1.legend()

# ── Graphique 2 : Distribution des erreurs ──
ax2 = axes[0, 1]
erreurs = y_pred_p - y_test_p
ax2.hist(erreurs, bins=40, color='#e8720c', alpha=0.8, edgecolor='white')
ax2.axvline(x=0, color='red', linestyle='--', linewidth=2)
ax2.axvline(x=mae,  color='green', linestyle='--', label=f'MAE = {mae:.0f} TND')
ax2.axvline(x=-mae, color='green', linestyle='--')
ax2.set_xlabel('Erreur de prédiction (TND)')
ax2.set_ylabel('Fréquence')
ax2.set_title('Distribution des erreurs\nModèle 1', fontweight='bold')
ax2.legend()

# ── Graphique 3 : Importance des features ──
ax3 = axes[1, 0]
feat_imp_sorted = feat_imp.sort_values('importance')
colors = ['#ef4444' if i >= len(feat_imp)-3 else '#3b82f6'
          for i in range(len(feat_imp))]
ax3.barh(feat_imp_sorted['feature'], feat_imp_sorted['importance'],
         color=colors, alpha=0.85)
ax3.set_xlabel('Importance')
ax3.set_title('Importance des variables\nModèle 1', fontweight='bold')

# ── Graphique 4 : Matrice de confusion ──
ax4 = axes[1, 1]
cm = confusion_matrix(y_test_a, y_pred_a)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=le_action.classes_
)
disp.plot(ax=ax4, colorbar=False, cmap='Blues')
ax4.set_title(f'Matrice de confusion\nModèle 2 (Accuracy: {accuracy*100:.1f}%)',
              fontweight='bold')
plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=7)
plt.setp(ax4.yaxis.get_majorticklabels(), fontsize=7)

plt.tight_layout()
plt.savefig('data/resultats_modeles.png', dpi=150, bbox_inches='tight')
plt.show()

# ══════════════════════════════════════════
# SAUVEGARDER LES MODÈLES
# ══════════════════════════════════════════

print("\n" + "─" * 55)
print("💾 SAUVEGARDE DES MODÈLES")
print("─" * 55)

with open('models/model_prix.pkl',    'wb') as f: pickle.dump(model_prix,   f)
with open('models/model_action.pkl',  'wb') as f: pickle.dump(model_action, f)
with open('models/le_dest.pkl',       'wb') as f: pickle.dump(le_dest,      f)
with open('models/le_saison.pkl',     'wb') as f: pickle.dump(le_saison,    f)
with open('models/le_action.pkl',     'wb') as f: pickle.dump(le_action,    f)
with open('models/features_prix.pkl', 'wb') as f: pickle.dump(features_prix, f)

print("✅ Modèles sauvegardés :")
print("   → models/model_prix.pkl")
print("   → models/model_action.pkl")
print("   → models/le_dest.pkl")
print("   → models/le_saison.pkl")
print("   → models/le_action.pkl")

# ── TEST RAPIDE ──
print("\n" + "─" * 55)
print("🧪 TEST RAPIDE DU MODÈLE")
print("─" * 55)

# Simuler un client qui veut aller à Paris
test_client = pd.DataFrame([{
    'destination_enc': le_dest.transform(['Paris, France'])[0],
    'nb_personnes':    2,
    'nb_nuits':        7,
    'saison_enc':      le_saison.transform(['haute'])[0],
    'nb_options':      3,
    'prix_affiche':    6500,
    'prix_plancher':   4800,
    'budget_client':   5200,
    'ratio_budget':    5200 / 6500,
    'popularite_dest': 0.9,
}])

prix_predit = model_prix.predict(test_client)[0]

print(f"\nClient : Paris, 2 personnes, 7 nuits, saison haute")
print(f"Prix affiché    : 6500 TND")
print(f"Prix plancher   : 4800 TND")
print(f"Budget client   : 5200 TND")
print(f"Prix prédit     : {prix_predit:.0f} TND  ← suggestion du modèle")
print(f"Marge conservée : {prix_predit - 4800:.0f} TND")