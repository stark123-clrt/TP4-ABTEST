# TP4 - A/B Testing avec RetailRocket

**Master AIA02-1 - Séance 4 : Analyse statistique & Décision produit**

---

## Description

Ce projet réalise une analyse complète d'A/B testing sur le dataset RetailRocket (e-commerce) :
- Simulation d'une randomisation A/B correcte
- Calcul du KPI Add-to-Cart Rate
- Test statistique de comparaison de proportions
- Génération automatique d'un rapport PDF professionnel

---

## Comment tester le projet

### Prérequis
- Python 3.x installé
- Fichier `events.csv` dans le dossier du projet

### Installation des dépendances

```bash
pip install pandas numpy matplotlib seaborn scipy statsmodels
```

### Exécution

**Option 1 : Script Python (RAPIDE - recommandé)**
```bash
python run_ab_test.py
```

**Option 2 : Notebook Jupyter**
```bash
jupyter notebook TP4_AB_Testing.ipynb
```
Puis : Run All Cells

---

## Fichiers générés automatiquement

Après l'exécution, vous obtiendrez :

1. **Rapport_AB_Test_RetailRocket.pdf** - Rapport complet 2 pages avec :
   - Contexte du test
   - Méthodologie statistique
   - Résultats KPI (tableaux comparatifs)
   - Test statistique (Z-score, p-value)
   - Décision finale justifiée
   - Recommandations business

2. **ab_test_comparison.png** - Graphiques de comparaison des groupes A et B

3. **ab_test_results.csv** - Tableau récapitulatif des résultats

---

## Résultats de l'analyse

### KPI mesure : Add-to-Cart Rate

- **Groupe A** : 2.6120%
- **Groupe B** : 2.7593%
- **Différence** : +0.1473 points (+5.64%)

### Test statistique

- **Z-score** : 4.0927
- **P-value** : 0.000043 (< 0.05)
- **Conclusion** : Différence **statistiquement significative**

### Décision finale

**DÉPLOYER LA VARIANTE B EN PRODUCTION**

Le groupe B performe significativement mieux avec une amélioration de 5.64% du taux d'ajout au panier.

---

## Structure du projet

```
TP4/
├── events.csv                           # Dataset RetailRocket (fourni)
├── run_ab_test.py                       # Script d'exécution rapide
├── TP4_AB_Testing.ipynb                 # Notebook Jupyter complet
├── README.md                            # Ce fichier
│
└── Fichiers générés :
    ├── Rapport_AB_Test_RetailRocket.pdf # Rapport final
    ├── ab_test_comparison.png           # Graphiques
    └── ab_test_results.csv              # Résultats CSV
```

---

## Livrables du TP

- [x] Notebook `.ipynb` propre et commenté
- [x] Rapport PDF professionnel (2 pages)
- [x] Recommandation business finale
- [x] Analyse statistique complète (test Z)
- [x] Visualisations claires

---

## Auteur

Master AIA02-1 - TP4 A/B Testing

---

## Notes

- Le script `run_ab_test.py` génère **automatiquement** tous les fichiers en une seule exécution
- L'analyse utilise un échantillon de 30% si le dataset dépasse 500k lignes (pour performances)
- Randomisation par utilisateur (pas par événement) pour respecter les bonnes pratiques A/B testing


Christian Ondiyo
