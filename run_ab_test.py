import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("TP4 - A/B TESTING RETAILROCKET - EXECUTION AUTOMATIQUE")
print("="*70)

# Configuration
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')
pd.set_option('display.max_columns', None)

# PARTIE 1 - CHARGEMENT DES DONNEES
print("\n[1/7] Chargement des données...")
df = pd.read_csv('events.csv')
print(f"✓ Dataset chargé: {df.shape[0]:,} lignes x {df.shape[1]} colonnes")
print(f"✓ Colonnes: {list(df.columns)}")

print("\nRépartition des événements:")
print(df['event'].value_counts())

# PARTIE 2 - NETTOYAGE
print("\n[2/7] Nettoyage des données...")
df_filtered = df[df['event'].isin(['view', 'addtocart'])].copy()
print(f"✓ Dataset filtré: {len(df_filtered):,} lignes (view + addtocart)")
print(f"✓ Visiteurs uniques: {df_filtered['visitorid'].nunique():,}")
print(f"✓ Produits uniques: {df_filtered['itemid'].nunique():,}")

# Échantillonnage si trop volumineux
if len(df_filtered) > 500000:
    print("  Dataset volumineux - échantillonnage 30%...")
    sample_visitors = df_filtered['visitorid'].unique()
    np.random.seed(42)
    sample_visitors = np.random.choice(sample_visitors, size=int(len(sample_visitors)*0.3), replace=False)
    df_filtered = df_filtered[df_filtered['visitorid'].isin(sample_visitors)]
    print(f"  Nouveau dataset: {len(df_filtered):,} lignes")

# PARTIE 3 - SIMULATION A/B TEST
print("\n[3/7] Simulation A/B test...")
unique_visitors = df_filtered['visitorid'].unique()
np.random.seed(42)
visitor_groups = pd.DataFrame({
    'visitorid': unique_visitors,
    'group': np.random.choice(['A', 'B'], size=len(unique_visitors), p=[0.5, 0.5])
})

print(f"✓ Randomisation: {len(unique_visitors):,} visiteurs")
print("  Répartition:")
for group, count in visitor_groups['group'].value_counts().items():
    pct = (count / len(visitor_groups)) * 100
    print(f"    Groupe {group}: {count:,} ({pct:.2f}%)")

df_ab = df_filtered.merge(visitor_groups, on='visitorid', how='left')

# PARTIE 4 - CALCUL KPI
print("\n[4/7] Calcul du KPI Add-to-Cart Rate...")

# Groupe A
df_group_a = df_ab[df_ab['group'] == 'A']
views_a = len(df_group_a[df_group_a['event'] == 'view'])
addtocart_a = len(df_group_a[df_group_a['event'] == 'addtocart'])
rate_a = (addtocart_a / views_a) * 100

# Groupe B
df_group_b = df_ab[df_ab['group'] == 'B']
views_b = len(df_group_b[df_group_b['event'] == 'view'])
addtocart_b = len(df_group_b[df_group_b['event'] == 'addtocart'])
rate_b = (addtocart_b / views_b) * 100

print(f"\nGROUPE A:")
print(f"  Views: {views_a:,}")
print(f"  Add-to-cart: {addtocart_a:,}")
print(f"  Taux: {rate_a:.4f}%")

print(f"\nGROUPE B:")
print(f"  Views: {views_b:,}")
print(f"  Add-to-cart: {addtocart_b:,}")
print(f"  Taux: {rate_b:.4f}%")

diff_absolute = rate_b - rate_a
diff_relative = ((rate_b - rate_a) / rate_a) * 100

print(f"\nDIFFÉRENCE:")
print(f"  Absolue: {diff_absolute:.4f} points")
print(f"  Relative: {diff_relative:.2f}%")
print(f"  Meilleur groupe: {'B' if rate_b > rate_a else 'A'}")

# PARTIE 5 - TEST STATISTIQUE
print("\n[5/7] Test statistique de proportions...")

n_a = views_a
n_b = views_b
x_a = addtocart_a
x_b = addtocart_b

p_a = x_a / n_a
p_b = x_b / n_b
p_pooled = (x_a + x_b) / (n_a + n_b)

se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n_a + 1/n_b))
z_score = (p_b - p_a) / se
p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

alpha = 0.05

print(f"✓ Proportion pooled: {p_pooled:.6f}")
print(f"✓ Erreur standard: {se:.6f}")
print(f"✓ Z-score: {z_score:.4f}")
print(f"✓ P-value: {p_value:.6f}")
print(f"✓ Seuil α: {alpha}")

print("\n" + "="*70)
print("RÉSULTAT DU TEST:")
print("="*70)

if p_value < alpha:
    print(f"✓ DIFFÉRENCE STATISTIQUEMENT SIGNIFICATIVE (p={p_value:.6f} < {alpha})")
    print(f"✓ On REJETTE l'hypothèse nulle H0")
    print(f"✓ Confiance: {(1-p_value)*100:.2f}%")

    if p_b > p_a:
        print(f"\n➜ Le groupe B est MEILLEUR que A")
        print(f"➜ Amélioration: +{diff_relative:.2f}%")
    else:
        print(f"\n➜ Le groupe A est MEILLEUR que B")
        print(f"➜ Amélioration: +{abs(diff_relative):.2f}%")
else:
    print(f"✗ DIFFÉRENCE NON SIGNIFICATIVE (p={p_value:.6f} >= {alpha})")
    print(f"✗ On NE PEUT PAS REJETER l'hypothèse nulle H0")
    print(f"✗ Les groupes peuvent être considérés équivalents")

# PARTIE 6 - ANALYSE BUSINESS
print("\n[6/7] Analyse Business...")

print("\n" + "="*70)
print("DÉCISION BUSINESS:")
print("="*70)

if p_value < alpha:
    if p_b > p_a:
        print("✓ RECOMMANDATION: DÉPLOYER LA VARIANTE B EN PRODUCTION")
        print(f"  • Impact attendu: +{diff_relative:.2f}% d'ajouts au panier")
        print(f"  • Gain absolu: +{diff_absolute:.4f} points")
        if views_b > 0:
            gain_per_1000 = (diff_absolute / 100) * 1000
            print(f"  • Pour 1000 vues: ~{gain_per_1000:.1f} ajouts supplémentaires")
        print("\n  Actions recommandées:")
        print("  1. Déploiement progressif (10% → 50% → 100%)")
        print("  2. Monitoring pendant 2 semaines")
        print("  3. Vérifier impact sur transactions finales")
    else:
        print("✓ RECOMMANDATION: CONSERVER LA VARIANTE A")
        print(f"  • B causerait une baisse de {abs(diff_relative):.2f}%")
        print("\n  Actions recommandées:")
        print("  1. Ne pas déployer B")
        print("  2. Analyser pourquoi B performe moins bien")
        print("  3. Itérer sur le design pour variante C")
else:
    print("⚠ RECOMMANDATION: AUCUN CHANGEMENT")
    print("  • Différence non significative")
    print("\n  Actions recommandées:")
    print("  1. Prolonger le test (2+ semaines)")
    print("  2. Augmenter taille échantillon")
    print("  3. Tester variante C avec changements plus marqués")

# PARTIE 7 - GÉNÉRATION DES FICHIERS
print("\n[7/7] Génération des fichiers de sortie...")

# Graphiques
fig, ax = plt.subplots(1, 2, figsize=(14, 5))

# Graphique 1: Taux de conversion
groups = ['Groupe A', 'Groupe B']
rates = [rate_a, rate_b]
colors = ['#3498db', '#e74c3c']

ax[0].bar(groups, rates, color=colors, alpha=0.7, edgecolor='black')
ax[0].set_ylabel('Taux d\'ajout au panier (%)', fontsize=12)
ax[0].set_title('Comparaison des taux de conversion', fontsize=14, fontweight='bold')
ax[0].grid(axis='y', alpha=0.3)
for i, v in enumerate(rates):
    ax[0].text(i, v + 0.01, f'{v:.4f}%', ha='center', va='bottom', fontweight='bold')

# Graphique 2: Volume d'événements
events_data = pd.DataFrame({
    'Groupe': ['A', 'A', 'B', 'B'],
    'Type': ['Views', 'Add-to-cart', 'Views', 'Add-to-cart'],
    'Count': [views_a, addtocart_a, views_b, addtocart_b]
})

events_pivot = events_data.pivot(index='Type', columns='Groupe', values='Count')
events_pivot.plot(kind='bar', ax=ax[1], color=colors, alpha=0.7, edgecolor='black')
ax[1].set_ylabel('Nombre d\'événements', fontsize=12)
ax[1].set_title('Volume d\'événements par groupe', fontsize=14, fontweight='bold')
ax[1].legend(title='Groupe')
ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=0)
ax[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('ab_test_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Graphique sauvegardé: ab_test_comparison.png")
plt.close()

# CSV résultats
summary_df = pd.DataFrame({
    'Métrique': ['Nombre de views', 'Nombre d\'add-to-cart', 'Taux de conversion (%)',
                 'Différence absolue', 'Différence relative (%)', 'Z-score', 'P-value', 'Significatif?'],
    'Groupe A': [f"{views_a:,}", f"{addtocart_a:,}", f"{rate_a:.4f}%", '-', '-', '-', '-', '-'],
    'Groupe B': [f"{views_b:,}", f"{addtocart_b:,}", f"{rate_b:.4f}%",
                 f"{diff_absolute:.4f} pts", f"{diff_relative:.2f}%",
                 f"{z_score:.4f}", f"{p_value:.6f}",
                 'OUI' if p_value < alpha else 'NON']
})

summary_df.to_csv('ab_test_results.csv', index=False)
print("✓ Résultats sauvegardés: ab_test_results.csv")

# Rapport PDF
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime

pdf_filename = 'Rapport_AB_Test_RetailRocket.pdf'

with PdfPages(pdf_filename) as pdf:
    # Page 1: Rapport textuel
    fig = plt.figure(figsize=(8.5, 11))
    fig.text(0.5, 0.95, 'RAPPORT A/B TEST - RETAILROCKET',
             ha='center', fontsize=16, fontweight='bold')
    fig.text(0.5, 0.92, f'Master AIA02-1 - {datetime.now().strftime("%d/%m/%Y")}',
             ha='center', fontsize=10)

    y_pos = 0.88
    fig.text(0.1, y_pos, '1. CONTEXTE DU TEST', fontsize=12, fontweight='bold')
    y_pos -= 0.03
    context_text = f"""Objectif: Comparer deux variantes (A et B) de l'interface e-commerce
KPI mesuré: Taux d'ajout au panier (Add-to-Cart Rate)
Dataset: RetailRocket events.csv
Population: {len(unique_visitors):,} visiteurs uniques
Événements analysés: {len(df_filtered):,} (views et add-to-cart)"""
    fig.text(0.1, y_pos, context_text, fontsize=9, verticalalignment='top')

    y_pos -= 0.15
    fig.text(0.1, y_pos, '2. MÉTHODOLOGIE', fontsize=12, fontweight='bold')
    y_pos -= 0.03
    method_text = f"""• Randomisation: Attribution aléatoire 50/50 des visiteurs aux groupes A et B
• Test statistique: Test Z de comparaison de proportions
• Seuil de significativité: α = 0.05 (95% de confiance)
• Hypothèses:
  - H0: Taux de conversion A = Taux de conversion B
  - H1: Taux de conversion A ≠ Taux de conversion B"""
    fig.text(0.1, y_pos, method_text, fontsize=9, verticalalignment='top')

    y_pos -= 0.18
    fig.text(0.1, y_pos, '3. RÉSULTATS - KPI', fontsize=12, fontweight='bold')
    y_pos -= 0.03

    kpi_text = f"""┌───────────────────────────────────────────────────┐
│  GROUPE A              │  GROUPE B               │
├───────────────────────────────────────────────────┤
│  Views: {views_a:>12,}    │  Views: {views_b:>12,}   │
│  Add-to-cart: {addtocart_a:>7,}    │  Add-to-cart: {addtocart_b:>7,}   │
│  Taux: {rate_a:>7.4f}%       │  Taux: {rate_b:>7.4f}%      │
└───────────────────────────────────────────────────┘

Différence absolue: {abs(diff_absolute):.4f} points
Différence relative: {abs(diff_relative):.2f}%
Meilleur groupe: {'B' if p_b > p_a else 'A'}"""
    fig.text(0.1, y_pos, kpi_text, fontsize=8, verticalalignment='top', family='monospace')

    y_pos -= 0.22
    fig.text(0.1, y_pos, '4. TEST STATISTIQUE', fontsize=12, fontweight='bold')
    y_pos -= 0.03
    stat_text = f"""Z-score: {z_score:.4f}
P-value: {p_value:.6f}
Seuil α: {alpha}

Conclusion: {'DIFFÉRENCE SIGNIFICATIVE' if p_value < alpha else 'DIFFÉRENCE NON SIGNIFICATIVE'}
➜ {'On REJETTE H0' if p_value < alpha else 'On NE PEUT PAS REJETER H0'}
➜ Confiance: {(1-p_value)*100:.2f}%"""
    fig.text(0.1, y_pos, stat_text, fontsize=9, verticalalignment='top')

    y_pos -= 0.18
    fig.text(0.1, y_pos, '5. DÉCISION FINALE', fontsize=12, fontweight='bold')
    y_pos -= 0.03

    if p_value < alpha and p_b > p_a:
        decision = f"""✓ DÉPLOYER LA VARIANTE B EN PRODUCTION
  Justification: Amélioration significative de +{diff_relative:.2f}%
  Impact: +{diff_absolute:.4f} points de taux de conversion"""
    elif p_value < alpha and p_a > p_b:
        decision = f"""✓ CONSERVER LA VARIANTE A
  Justification: A performe significativement mieux
  Impact: B causerait une baisse de {abs(diff_relative):.2f}%"""
    else:
        decision = """⚠ AUCUN CHANGEMENT RECOMMANDÉ
  Justification: Différence non significative
  Action: Prolonger le test ou tester une variante C"""

    fig.text(0.1, y_pos, decision, fontsize=10, verticalalignment='top')

    y_pos -= 0.15
    fig.text(0.1, y_pos, '6. RECOMMANDATIONS PRODUIT', fontsize=12, fontweight='bold')
    y_pos -= 0.03

    if p_value < alpha and p_b > p_a:
        reco = """• Déploiement progressif: 10% → 50% → 100% du trafic
• Monitoring post-déploiement sur 2 semaines
• Analyse de l'impact sur les transactions complètes
• Segmentation: impact sur nouveaux vs anciens clients"""
    else:
        reco = """• Prolonger le test pour augmenter la puissance statistique
• Analyser les données qualitatives (heatmaps, recordings)
• Tester une variante C avec des changements plus marqués
• Prioriser d'autres optimisations à plus fort impact"""

    fig.text(0.1, y_pos, reco, fontsize=9, verticalalignment='top')

    plt.axis('off')
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

    # Page 2: Graphiques
    fig2, axes = plt.subplots(2, 1, figsize=(8.5, 11))
    fig2.suptitle('VISUALISATIONS - A/B TEST', fontsize=14, fontweight='bold', y=0.98)

    # Graphique 1: Taux de conversion
    axes[0].bar(groups, rates, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    axes[0].set_ylabel('Taux d\'ajout au panier (%)', fontsize=11)
    axes[0].set_title('Comparaison des taux de conversion', fontsize=12, fontweight='bold', pad=15)
    axes[0].grid(axis='y', alpha=0.3, linestyle='--')
    for i, v in enumerate(rates):
        axes[0].text(i, v + max(rates)*0.01, f'{v:.4f}%',
                     ha='center', va='bottom', fontweight='bold', fontsize=11)

    # Graphique 2: Distribution normale
    x = np.linspace(-4, 4, 1000)
    y = stats.norm.pdf(x, 0, 1)

    axes[1].plot(x, y, 'k-', linewidth=2, label='Distribution normale standard')
    axes[1].fill_between(x, y, where=(x <= -1.96) | (x >= 1.96),
                          alpha=0.3, color='red', label='Zone de rejet (α=0.05)')
    axes[1].axvline(z_score, color='blue', linestyle='--', linewidth=2,
                    label=f'Z-score observé = {z_score:.4f}')
    axes[1].axvline(-1.96, color='red', linestyle=':', linewidth=1)
    axes[1].axvline(1.96, color='red', linestyle=':', linewidth=1)
    axes[1].set_xlabel('Z-score', fontsize=11)
    axes[1].set_ylabel('Densité de probabilité', fontsize=11)
    axes[1].set_title('Test Z bilatéral - Visualisation', fontsize=12, fontweight='bold', pad=15)
    axes[1].legend(loc='upper right', fontsize=9)
    axes[1].grid(True, alpha=0.3, linestyle='--')
    axes[1].text(0, max(y)*0.5, f'p-value = {p_value:.6f}',
                 ha='center', fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    pdf.savefig(fig2, bbox_inches='tight')
    plt.close()

print(f"✓ Rapport PDF généré: {pdf_filename}")

print("\n" + "="*70)
print("FICHIERS GÉNÉRÉS:")
print("="*70)
print("✓ ab_test_comparison.png - Graphiques de comparaison")
print("✓ ab_test_results.csv - Tableau récapitulatif")
print("✓ Rapport_AB_Test_RetailRocket.pdf - Rapport complet 2 pages")
print("\n✓✓✓ ANALYSE TERMINÉE AVEC SUCCÈS! ✓✓✓")
print("="*70)
