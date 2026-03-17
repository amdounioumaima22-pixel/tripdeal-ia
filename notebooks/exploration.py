import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# ══════════════════════════════════════════
# CHARGEMENT DES DONNÉES
# ══════════════════════════════════════════

df_nego  = pd.read_csv('data/negociations.csv')
df_tours = pd.read_csv('data/tours_negociation.csv')

print("=" * 55)
print("📊 APERÇU DU DATASET NÉGOCIATIONS")
print("=" * 55)
print(df_nego.head())
print(f"\nColonnes : {list(df_nego.columns)}")
print(f"\nTypes :\n{df_nego.dtypes}")
print(f"\nValeurs manquantes :\n{df_nego.isnull().sum()}")

# ══════════════════════════════════════════
# VISUALISATIONS
# ══════════════════════════════════════════

plt.style.use('seaborn-v0_8-whitegrid')
fig = plt.figure(figsize=(18, 14))
fig.suptitle('📊 Analyse du Dataset de Négociation TripDeal', 
             fontsize=16, fontweight='bold', y=0.98)

gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.35)

# ── GRAPHIQUE 1 : Taux de deals conclus ──
ax1 = fig.add_subplot(gs[0, 0])
deals = df_nego['deal_conclu'].value_counts()
colors = ['#22c55e', '#ef4444']
ax1.pie(deals.values, 
        labels=['Deal conclu ✅', 'Pas de deal ❌'],
        colors=colors,
        autopct='%1.1f%%',
        startangle=90,
        textprops={'fontsize': 10})
ax1.set_title('Taux de deals conclus', fontweight='bold')

# ── GRAPHIQUE 2 : Distribution des prix affiché vs final ──
ax2 = fig.add_subplot(gs[0, 1:])
deals_only = df_nego[df_nego['deal_conclu'] == True]
ax2.hist(df_nego['prix_affiche'], bins=30, alpha=0.6, 
         color='#3b82f6', label='Prix affiché')
ax2.hist(deals_only['prix_final'], bins=30, alpha=0.6, 
         color='#e8720c', label='Prix final négocié')
ax2.set_xlabel('Prix (TND)')
ax2.set_ylabel('Nombre de négociations')
ax2.set_title('Distribution Prix Affiché vs Prix Final', fontweight='bold')
ax2.legend()

# ── GRAPHIQUE 3 : Deals par destination ──
ax3 = fig.add_subplot(gs[1, :2])
dest_deals = df_nego.groupby('destination')['deal_conclu'].agg(['sum', 'count'])
dest_deals['taux'] = dest_deals['sum'] / dest_deals['count'] * 100
dest_deals = dest_deals.sort_values('taux', ascending=True)

colors_bar = ['#ef4444' if t < 60 else '#f97316' if t < 70 else '#22c55e' 
              for t in dest_deals['taux']]
bars = ax3.barh(dest_deals.index, dest_deals['taux'], color=colors_bar)
ax3.set_xlabel('Taux de deals conclus (%)')
ax3.set_title('Taux de succès par destination', fontweight='bold')
ax3.axvline(x=70, color='gray', linestyle='--', alpha=0.5, label='Moyenne 70%')

for bar, val in zip(bars, dest_deals['taux']):
    ax3.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
             f'{val:.1f}%', va='center', fontsize=9)

# ── GRAPHIQUE 4 : Nombre de tours ──
ax4 = fig.add_subplot(gs[1, 2])
tours_count = df_nego['nb_tours'].value_counts().sort_index()
ax4.bar(tours_count.index, tours_count.values, 
        color='#8b5cf6', alpha=0.8, edgecolor='white')
ax4.set_xlabel('Nombre de tours')
ax4.set_ylabel('Fréquence')
ax4.set_title('Distribution du nombre\nde tours', fontweight='bold')

for i, (x, y) in enumerate(zip(tours_count.index, tours_count.values)):
    ax4.text(x, y + 5, str(y), ha='center', fontsize=9)

# ── GRAPHIQUE 5 : Actions de l'agent ──
ax5 = fig.add_subplot(gs[2, 0])
actions = df_tours['action_agent'].value_counts()
action_labels = {
    'aucune':              'Accepté ✅',
    'reduire_5_pct':       'Baisse 5%',
    'reduire_3_pct':       'Baisse 3%',
    'reduire_2_pct':       'Baisse 2%',
    'offre_alternative':   'Alternative',
    'refuser_negociation': 'Refus final',
}
labels = [action_labels.get(a, a) for a in actions.index]
colors_pie = ['#22c55e', '#3b82f6', '#f97316', '#e8720c', '#8b5cf6', '#ef4444']
ax5.pie(actions.values, labels=labels, autopct='%1.1f%%',
        colors=colors_pie[:len(actions)],
        textprops={'fontsize': 8})
ax5.set_title('Actions de l\'agent', fontweight='bold')

# ── GRAPHIQUE 6 : Marge conservée par saison ──
ax6 = fig.add_subplot(gs[2, 1])
marge_saison = deals_only.groupby('saison')['marge_finale'].mean()
colors_saison = {'haute': '#ef4444', 'moyenne': '#f97316', 'basse': '#3b82f6'}
bars2 = ax6.bar(marge_saison.index, 
                marge_saison.values,
                color=[colors_saison.get(s, '#gray') for s in marge_saison.index],
                alpha=0.85, edgecolor='white')
ax6.set_ylabel('Marge moyenne (TND)')
ax6.set_title('Marge conservée\npar saison', fontweight='bold')

for bar, val in zip(bars2, marge_saison.values):
    ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
             f'{val:.0f}', ha='center', fontsize=10, fontweight='bold')

# ── GRAPHIQUE 7 : Reward moyen par action ──
ax7 = fig.add_subplot(gs[2, 2])
reward_action = df_tours.groupby('action_agent')['reward'].mean().sort_values()
colors_reward = ['#ef4444' if r < 0 else '#22c55e' for r in reward_action.values]
bars3 = ax7.barh(
    [action_labels.get(a, a) for a in reward_action.index],
    reward_action.values,
    color=colors_reward, alpha=0.85
)
ax7.axvline(x=0, color='black', linewidth=0.8)
ax7.set_xlabel('Reward moyen')
ax7.set_title('Reward moyen\npar action', fontweight='bold')

plt.savefig('data/analyse_dataset.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n✅ Graphiques sauvegardés dans data/analyse_dataset.png")

# ══════════════════════════════════════════
# STATISTIQUES DÉTAILLÉES
# ══════════════════════════════════════════

print("\n" + "=" * 55)
print("📈 STATISTIQUES DÉTAILLÉES")
print("=" * 55)

print("\n🎯 Par destination :")
dest_stats = df_nego.groupby('destination').agg(
    nb_nego   = ('negociation_id', 'count'),
    taux_deal = ('deal_conclu',    'mean'),
    prix_moy  = ('prix_affiche',   'mean'),
    marge_moy = ('marge_finale',   'mean'),
    tours_moy = ('nb_tours',       'mean'),
).round(2)
dest_stats['taux_deal'] = (dest_stats['taux_deal'] * 100).round(1)
print(dest_stats.to_string())

print("\n🌤️ Par saison :")
saison_stats = df_nego.groupby('saison').agg(
    nb_nego   = ('negociation_id', 'count'),
    taux_deal = ('deal_conclu',    'mean'),
    prix_moy  = ('prix_affiche',   'mean'),
).round(2)
saison_stats['taux_deal'] = (saison_stats['taux_deal'] * 100).round(1)
print(saison_stats.to_string())

print("\n🎬 Actions les plus fréquentes :")
print(df_tours['action_agent'].value_counts().to_string())