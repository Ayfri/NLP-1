# Discord Messages NLP Pipeline

Un pipeline NLP complet pour analyser les messages Discord avec des fonctionnalités avancées de traitement de texte et d'analyse conversationnelle.

## 🎯 Fonctionnalités

### Traitement NLP
- **Nettoyage de texte avancé** : Suppression des mentions Discord, emojis, URLs, caractères spéciaux
- **Support des contractions françaises** : Expansion automatique (c'est → ce est, qu'il → que il, etc.)
- **Tokenisation intelligente** : Découpage en tokens avec filtrage par longueur
- **Suppression des mots vides** : Support français et anglais avec préservation contextuelle
- **Lemmatisation** : Réduction aux formes canoniques avec spaCy
- **Traitement par lots optimisé** : Processing efficace de grandes quantités de messages

### Analyse de sentiment
- **Score de sentiment** : Analyse avec VADER (de -1 à +1)
- **Classification émotionnelle** : Positif, négatif, neutre
- **Support multilingue** : Français et anglais

### Analyse conversationnelle
- **Détection automatique de conversations** : Basée sur les intervalles de temps
- **Métriques par conversation** :
  - Durée et nombre de messages
  - Participants actifs
  - Émotion dominante
  - Sujets clés (mots les plus fréquents)
- **Analyse des participants** : Classement par activité

### Analyses statistiques
- **Fréquence des lemmes** : Top 100 mots avec pourcentages
- **Statistiques globales** : Tokens moyens, distribution des sentiments
- **Résumé conversationnel** : Vue d'ensemble de l'activité

## 📋 Prérequis

- Python 3.8+
- Modèle spaCy français : `python -m spacy download fr_core_news_sm`
- Dépendances listées dans `requirements.txt`

## 🚀 Installation

1. Clonez le dépôt
2. Installez les dépendances :
   ```bash
   pip install -r requirements.txt
   ```
3. Téléchargez le modèle spaCy :
   ```bash
   python -m spacy download fr_core_news_sm
   ```

## 📤 Export des messages Discord

Pour obtenir vos messages Discord au format CSV, utilisez [DiscordChatExporter](https://github.com/Tyrrrz/DiscordChatExporter) :

1. Téléchargez DiscordChatExporter
2. Exportez vos conversations au format **CSV**
3. Placez les fichiers CSV dans le dossier `data/`

## 🔧 Utilisation

1. Placez vos fichiers CSV Discord dans le dossier `data/`
2. Lancez le traitement :
   ```bash
   python main.py
   ```
3. Les résultats seront sauvegardés dans le dossier `output/`

## 📊 Format d'entrée

Les fichiers CSV doivent contenir les colonnes suivantes :
- `AuthorID` : ID utilisateur Discord
- `Author` : Nom d'utilisateur
- `Date` : Horodatage du message
- `Content` : Contenu du message
- `Attachments` : Pièces jointes (optionnel)
- `Reactions` : Réactions au message (optionnel)

## 📁 Fichiers de sortie

- `messages_processed.csv` : Messages traités avec résultats NLP
- `conversations_analysis.csv` : Analyse détaillée des conversations
- `conversation_summary.csv` : Résumé global des conversations
- `lemma_frequency.csv` : Fréquence des lemmes (top 100)
- Logs détaillés dans la console

## 🏗️ Structure du projet

```
nlp-2/
├── data/              # Fichiers CSV d'entrée
├── output/            # Résultats de traitement
├── models/            # Modèles (vide)
├── utils/             # Utilitaires (vide)
├── main.py            # Script principal
├── requirements.txt   # Dépendances Python
└── README.md          # Ce fichier
```

## ⚙️ Configuration

Les paramètres peuvent être ajustés dans `main.py` :
- `BATCH_SIZE` : Taille des lots de traitement (défaut: 2000)
- `MIN_TOKEN_LENGTH` / `MAX_TOKEN_LENGTH` : Filtrage des tokens
- `FREQ_ANALYSIS_TOP_N` : Nombre de lemmes dans l'analyse de fréquence
- `gap_minutes` : Seuil de détection des conversations (défaut: 30 min)

## 🔍 Pipeline de traitement

```
Message brut → Nettoyage → Tokenisation → Suppression mots vides → Lemmatisation → Analyse sentiment
```

Chaque étape est optimisée pour le traitement de grandes quantités de données avec support multilingue français/anglais.
