# AgriPredict — Classification de cultures agricoles

## Objectif du projet

Ce projet a pour but de prédire la **classe d'une culture agricole** (Céréales, Légumineuses, Maraîchage ou Cultures industrielles) à partir de caractéristiques d'une parcelle : propriétés du sol, conditions climatiques et pratiques agricoles.

Il s'agit d'un problème de **classification multi-classe** (4 classes, équilibrées) en apprentissage supervisé.

## Le dataset — `dataset_cultures.csv`

600 parcelles, 14 variables descriptives + 1 variable cible.

| Variable | Type | Description |
|---|---|---|
| `pH_sol` | Numérique | pH du sol |
| `azote_N_kg_ha` | Numérique | Azote disponible (kg/ha) |
| `phosphore_P_kg_ha` | Numérique | Phosphore disponible (kg/ha) |
| `potassium_K_kg_ha` | Numérique | Potassium disponible (kg/ha) |
| `pluviometrie_mm` | Numérique | Pluviométrie moyenne (mm) |
| `temperature_moy_C` | Numérique | Température moyenne (°C) |
| `humidite_air_pct` | Numérique | Humidité de l'air (%) |
| `type_sol` | Catégorielle nominale | Argileux, Sableux, Limoneux, Argilo-limoneux, Sablo-limoneux |
| `region` | Catégorielle nominale | Nord, Centre, Sud, Est, Ouest |
| `methode_irrigation` | Catégorielle nominale | Goutte-à-goutte, Aspersion, Gravitaire, Pluvial |
| `niveau_ensoleillement` | Catégorielle **ordinale** | Faible < Moyen < Élevé |
| `qualite_drainage` | Catégorielle **ordinale** | Mauvais < Moyen < Bon < Excellent |
| `niveau_fertilite_sol` | Catégorielle **ordinale** | Bas < Moyen < Élevé |
| `classe_culture` | **Cible** | Cereales, Legumineuses, Maraichage, Cultures_industrielles |

Le dataset est synthétique mais construit avec une logique agronomique cohérente (ex. les légumineuses nécessitent moins d'azote car elles le fixent elles-mêmes ; les cultures industrielles tolèrent des températures plus élevées et une humidité plus faible), avec un chevauchement volontaire entre classes pour que le problème reste réaliste et non trivial.

## Démarche (notebook `AgriPredict.ipynb`)

1. **Exploration** : visualisation des relations entre variables (`pairplot` coloré par classe).
2. **Préparation des données** :
   - Séparation de la variable cible (`classe_culture`) des variables explicatives.
   - Distinction claire entre variables **numériques**, **catégorielles ordinales** et **catégorielles nominales**, chacune traitée différemment.
3. **Prétraitement** (`ColumnTransformer`) :
   - `StandardScaler` sur les variables numériques.
   - `OrdinalEncoder` (avec un ordre explicite) sur les variables ordinales.
   - `OneHotEncoder` sur les variables nominales.
4. **Modélisation** : pipeline `StandardScaler/OrdinalEncoder/OneHotEncoder + KNeighborsClassifier`.
5. **Évaluation** :
   - Validation croisée à 5 plis (`KFold`).
   - Comparaison avec un `DummyClassifier` pour vérifier que le modèle apprend un signal réel.
6. **Test de sensibilité** : vérification de l'effet d'une variable (`region`) sur la prédiction d'une parcelle donnée.

## Résultats

| Modèle | Accuracy (validation croisée) |
|---|---|
| `DummyClassifier` (référence) | ~0,21 |
| `KNeighborsClassifier` (k=5) | ~0,96 |

Le modèle dépasse largement la référence aléatoire. L'analyse par classe (`classification_report`) montre des performances homogènes, avec une légère confusion entre "Céréales" et les deux autres classes de cultures pluviales.

## Structure du dépôt

```
.
├── AgriPredict.ipynb        # Notebook principal (exploration, modélisation, évaluation)
├── dataset_cultures.csv     # Jeu de données
└── README.md                 # Ce fichier
```

## Pistes d'amélioration

- Ajustement des hyperparamètres (`GridSearchCV`) sur le nombre de voisins.
- Comparaison avec d'autres modèles (RandomForest, LogisticRegression).
- Matrice de confusion normalisée pour affiner l'analyse des erreurs.

## Exécution

```bash
pip install pandas numpy scikit-learn seaborn matplotlib
jupyter notebook AgriPredict.ipynb
```
