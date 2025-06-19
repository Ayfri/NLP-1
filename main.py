#!/usr/bin/env python3
"""
Discord Messages NLP Pipeline
Enhanced NLP processing: cleaning → tokenization → stopwords removal → lemmatization
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import re
from datetime import datetime
from collections import Counter, defaultdict

# NLP imports
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import sklearn.feature_extraction.text
from sklearn.feature_extraction.text import TfidfVectorizer

# Visualization imports (from TP1)
import matplotlib.pyplot as plt
import seaborn as sns
import json

# ================================
# CONSTANTS - Configuration
# ================================

# Logging configuration
LOG_LEVEL = logging.INFO
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'

# Processing parameters
MIN_TOKEN_LENGTH = 2
MAX_TOKEN_LENGTH = 50
BATCH_SIZE = 100

# French contractions mapping for better preprocessing
FRENCH_CONTRACTIONS = {
    r"c'est": "ce est",
    r"qu'": "que ",
    r"d'": "de ",
    r"n'": "ne ",
    r"s'": "se ",
    r"j'": "je ",
    r"t'": "te ",
    r"m'": "me ",
    r"l'": "le ",
    r"jusqu'": "jusque ",
    r"aujourd'hui": "aujourd hui",
    r"quelqu'": "quelque ",
}

# Important short words to preserve (common in French)
IMPORTANT_SHORT_WORDS = {
    'ça', 'si', 'ou', 'et', 'en', 'un', 'le', 'la', 'de', 'du', 'ce', 'se',
    'me', 'te', 'ne', 'je', 'tu', 'il', 'on', 'au', 'ai', 'as', 'va', 'eu'
}

# Contextually important words to keep (less aggressive stopword filtering)
CONTEXTUAL_WORDS_TO_KEEP = {
    'pas', 'non', 'oui', 'bien', 'tout', 'tous', 'faire', 'dit', 'voir',
    'très', 'plus', 'moins', 'beaucoup', 'peu', 'assez', 'trop'
}

# Common lemmatization fixes for French
LEMMA_FIXES = {
    'pourer': {'pourrait', 'pourait', 'pourrai', 'pourrais'},
    'lavai': {'lavais'},
    'donner': {'donnerais', 'donnerai', 'donneraient'},
    'faire': {'ferais', 'ferait', 'feraient'},
    'avoir': {'aurais', 'aurait', 'auraient'},
    'être': {'serais', 'serait', 'seraient'},
}

# Output configuration
OUTPUT_DIR = "output"
FREQ_ANALYSIS_TOP_N = 100
STATS_TOP_LEMMAS = 20

# Emotion analysis configuration
EMOTION_LEXICON_FR = {
	'positif': [
		# Mots formels
		'heureux', 'content', 'joie', 'génial', 'super', 'cool', 'bien', 'bon', 'excellent',
		'parfait', 'magnifique', 'formidable', 'merveilleux', 'fantastique', 'extraordinaire',
		# Expressions Discord/Internet
		'mdr', 'lol', 'ptdr', 'xD', 'hilarant', 'marrant', 'rigolo', 'drôle', 'fun',
		'top', 'nickel', 'stylé', 'classe', 'ouf', 'trop bien', 'grave cool', 'bg',
		'gg', 'wp', 'nice', 'ez', 'poggers', 'pog', 'lets go', 'yay', 'yes', 'yess',
		# Emojis textuels
		':)', '^^', ':D', '=)', ':3', 'x)'
	],
	'négatif': [
		# Mots formels
		'triste', 'énervé', 'colère', 'mal', 'mauvais', 'nul', 'problème', 'erreur', 'bug',
		'pénible', 'agaçant', 'frustrant', 'horrible', 'terrible', 'affreux', 'catastrophique',
		# Expressions familières
		'merde', 'putain', 'chiant', 'relou', 'galère', 'bordel', 'fait chier', 'ras le bol',
		'saoulé', 'soûle', 'gavé', 'blasé', 'dégoûté', 'deg', 'la flemme', 'chaud',
		'rip', 'dead', 'mort', 'tué', 'fini', 'cramé', 'naze', 'pourri',
		# Emojis textuels
		':(', ':/', ':|', 'T_T', '-_-', '>_<'
	],
	'surprise': [
		'wow', 'waouh', 'woah', 'omg', 'wtf', 'quoi', 'sérieux', 'vraiment',
		'incroyable', 'surprenant', 'étonnant', 'impressionnant', 'ouf',
		'dingue', 'fou', 'malade', 'bizarre', 'étrange', 'chelou', 'louche',
		'jamais vu', 'première fois', 'ah bon', 'oh', 'ah', 'hein', 'pardon',
		'o_O', 'O_o', ':O', 'D:'
	],
	'peur': [
		'peur', 'angoisse', 'stress', 'inquiet', 'crainte', 'effrayer', 'terroriser',
		'paniquer', 'flipper', 'psychoter', 'bad', 'tendu', 'chaud', 'risqué',
		'dangereux', 'attention', 'careful', 'méfie', 'gaffe'
	],
	'colère': [
		'rage', 'rageux', 'tilt', 'tilté', 'salé', 'salty', 'vénère', 'vnr',
		'furieux', 'furax', 'enragé', 'péter un câble', 'péter un plomb',
		'exploser', 'craquer', 'foutre', 'casser', 'détruire', 'taper'
	],
	'dégoût': [
		'dégoût', 'beurk', 'berk', 'dégueulasse', 'dégueu', 'écœurant', 'répugnant',
		'horrible', 'ignoble', 'immonde', 'crade', 'sale', 'pourri', 'moisi',
		'gerber', 'vomir', 'nausée', 'bleh', 'eurk', 'yuck'
	],
	'confiance': [
		'confiance', 'sûr', 'certain', 'évident', 'clair', 'précis', 'exact',
		'fiable', 'crédible', 'garanti', 'promis', 'juré', 'validé', 'confirmé',
		'approuvé', 'ok', 'okay', 'ça marche', 'nickel', 'parfait', 'impec'
	],
	'espoir': [
		'espoir', 'espérer', 'souhaiter', 'vouloir', 'rêver', 'imaginer',
		'peut-être', 'possible', 'potentiel', 'chance', 'opportunité',
		'bientôt', 'prochainement', 'futur', 'avenir', 'projet'
	]
}

# Conversation grouping parameters (messages within X minutes = same conversation)
CONVERSATION_GAP_MINUTES = 15
MIN_MESSAGES_PER_CONVERSATION = 3
MAX_TITLE_WORDS = 5

# ================================
# LOGGING SETUP
# ================================

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# ================================
# MAIN CLASS
# ================================

class DiscordNLPProcessor:
	"""Enhanced Discord message processor with improved NLP pipeline"""
	nlp: spacy.Language
	french_stopwords: set[str]
	english_stopwords: set[str]
	contextual_stopwords: set[str]
	sentiment_analyzer: SentimentIntensityAnalyzer

	def __init__(self):
		"""Initialize the enhanced processor"""
		logger.info("🔧 Initializing enhanced NLP processor...")

		# Download NLTK resources
		self._download_nltk_data()

		# Load spaCy model
		self._load_spacy_model()

		# Setup enhanced stopwords
		self.french_stopwords = set(stopwords.words('french'))
		self.english_stopwords = set(stopwords.words('english'))

		# Create contextual stopwords (remove less aggressive ones)
		all_stopwords = self.french_stopwords.union(self.english_stopwords)
		self.contextual_stopwords = all_stopwords - CONTEXTUAL_WORDS_TO_KEEP

		# Initialize sentiment analyzer
		self.sentiment_analyzer = SentimentIntensityAnalyzer()

		logger.info("✅ Enhanced NLP processor initialized")

	def _download_nltk_data(self):
		"""Download required NLTK data"""
		try:
			nltk.data.find('tokenizers/punkt')
		except LookupError:
			nltk.download('punkt')

		try:
			nltk.data.find('corpora/stopwords')
		except LookupError:
			nltk.download('stopwords')

		try:
			nltk.data.find('vader_lexicon')
		except LookupError:
			nltk.download('vader_lexicon')

	def _load_spacy_model(self):
		"""Load spaCy model (French preferred, English fallback)"""
		try:
			self.nlp = spacy.load("fr_core_news_sm")
			logger.info("📝 Using French spaCy model")
		except OSError:
			try:
				self.nlp = spacy.load("en_core_web_sm")
				logger.info("📝 Using English spaCy model")
			except OSError:
				logger.error("❌ No spaCy model found. Install with: python -m spacy download fr_core_news_sm")
				raise

	def extract_discord_features(self, text: str) -> dict:
		"""Extract Discord-specific features from text (inspired by TP1)"""
		features = {}

		default_features = {
			'caps_ratio': 0,
			'exclamation_count': 0,
			'question_count': 0,
			'mention_count': 0,
			'emoji_count': 0,
			'code_block_count': 0,
			'url_count': 0,
			'message_length': 0,
			'word_count': 0,
			'avg_word_length': 0
		}

		# Handle NaN, None, or non-string values
		if pd.isna(text) or text is None or not isinstance(text, str):
			return default_features

		# Convert to string if not already
		text = str(text)

		if not text:
			return default_features

		# Ratio of capital letters (excitement/shouting indicator)
		caps_count = sum(1 for char in text if char.isupper())
		features['caps_ratio'] = caps_count / len(text) if len(text) > 0 else 0

		# Punctuation counts
		features['exclamation_count'] = text.count('!')
		features['question_count'] = text.count('?')

		# Discord-specific patterns
		features['mention_count'] = len(re.findall(r'@\w+', text))
		features['emoji_count'] = len(re.findall(r':\w+:', text)) + len(re.findall(r'<a?:\w+:\d+>', text))
		features['code_block_count'] = len(re.findall(r'```', text))
		features['url_count'] = len(re.findall(r'https?://\S+', text))

		# Text statistics
		features['message_length'] = len(text)
		words = text.split()
		features['word_count'] = len(words)
		features['avg_word_length'] = sum(len(w) for w in words) / len(words) if words else 0

		return features

	def clean_text(self, text: str) -> str:
		"""Enhanced text cleaning with better French support"""
		if pd.isna(text) or text == "":
			return ""

		# Remove Discord mentions (@user and @user#1234)
		text = re.sub(r'<@!?\d+>', '', text)
		text = re.sub(r'@\w+(?:#\d{4})?', '', text)

		# Remove Discord emojis (:emoji:) and custom emojis but preserve text emoticons
		# First, preserve text emoticons by replacing them temporarily
		emoticon_mapping = {
			':)': '__HAPPY__',
			':D': '__VERYHAPPY__',
			':(': '__SAD__',
			':/': '__UNSURE__',
			':|': '__NEUTRAL__',
			'^^': '__HAPPY2__',
			'=)': '__HAPPY3__',
			':3': '__CUTE__',
			'T_T': '__CRY__',
			'-_-': '__ANNOYED__',
			'>_<': '__FRUSTRATED__',
			'o_O': '__SURPRISED__',
			'O_o': '__SURPRISED2__',
			':O': '__SHOCKED__',
			'D:': '__DISMAY__',
			'xD': '__LAUGH__',
			'XD': '__LAUGH2__',
			'x)': '__LAUGH3__',
		}

		for emoticon, placeholder in emoticon_mapping.items():
			text = text.replace(emoticon, placeholder)

		# Now remove Discord emojis
		text = re.sub(r':\w+:', '', text)
		text = re.sub(r'<a?:\w+:\d+>', '', text)

		# Remove URLs (improved regex)
		text = re.sub(r'https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:\w*))?)?', '', text)

		# Handle French contractions systematically
		for contraction, replacement in FRENCH_CONTRACTIONS.items():
			text = re.sub(contraction, replacement, text, flags=re.IGNORECASE)

		# Handle remaining apostrophes
		text = re.sub(r"([a-zA-Z])'([a-zA-Z])", r'\1 \2', text)

		# Remove special characters but keep basic punctuation and accents
		text = re.sub(r'[^\w\s.,!?;:\-àáâäçèéêëìíîïñòóôöùúûüÿæœÀÁÂÄÇÈÉÊËÌÍÎÏÑÒÓÔÖÙÚÛÜŸÆŒ]', '', text)

		# Restore emoticons
		for emoticon, placeholder in emoticon_mapping.items():
			text = text.replace(placeholder, emoticon)

		# Clean up multiple spaces and normalize whitespace
		text = re.sub(r'\s+', ' ', text)

		return text.strip()

	def tokenize_text(self, text: str) -> list[str]:
		"""Enhanced tokenization with better French support"""
		if not text:
			return []

		tokens = word_tokenize(text.lower(), language='french')

		# Enhanced filtering with length constraints and important short words
		filtered_tokens = []
		for token in tokens:
			if (token.isalpha() and
				(MIN_TOKEN_LENGTH <= len(token) <= MAX_TOKEN_LENGTH or
				 token in IMPORTANT_SHORT_WORDS)):
				filtered_tokens.append(token)

		return filtered_tokens

	def remove_stopwords(self, tokens: list[str]) -> list[str]:
		"""Enhanced stopword removal with contextual preservation"""
		return [token for token in tokens if token not in self.contextual_stopwords]

	def lemmatize_tokens(self, tokens: list[str]) -> list[str]:
		"""Enhanced lemmatization with French-specific fixes"""
		if not tokens:
			return []

		text = " ".join(tokens)
		doc = self.nlp(text)

		lemmas = []
		for token in doc:
			if token.is_alpha and len(token.lemma_) > 1:
				lemma = token.lemma_.lower()

				# Apply French-specific lemmatization fixes
				original_word = token.text.lower()
				for correct_lemma, word_set in LEMMA_FIXES.items():
					if original_word in word_set:
						lemma = correct_lemma
						break

				lemmas.append(lemma)

		return lemmas

	def analyze_emotions(self, text: str, lemmas: list[str]) -> dict:
		"""Analyze emotions in text using multiple approaches"""
		if not text:
			return {
				'sentiment_score': 0.0,
				'sentiment_label': 'neutral',
				'detected_emotions': [],
				'emotion_scores': {},
				'textblob_polarity': 0.0,
				'textblob_subjectivity': 0.0
			}

		# VADER sentiment analysis (works better with informal text)
		vader_scores = self.sentiment_analyzer.polarity_scores(text)

		# TextBlob sentiment analysis
		try:
			blob = TextBlob(text)
			textblob_polarity = blob.sentiment.polarity
			textblob_subjectivity = blob.sentiment.subjectivity
		except:
			textblob_polarity = 0.0
			textblob_subjectivity = 0.0

		# French emotion lexicon analysis
		emotion_scores = {}
		detected_emotions = []

		text_lower = text.lower()
		lemmas_lower = [lemma.lower() for lemma in lemmas]

		for emotion, keywords in EMOTION_LEXICON_FR.items():
			score = 0
			for keyword in keywords:
				# Check in original text
				if keyword in text_lower:
					score += text_lower.count(keyword)
				# Check in lemmas
				if keyword in lemmas_lower:
					score += lemmas_lower.count(keyword)

			emotion_scores[emotion] = score
			if score > 0:
				detected_emotions.append(emotion)

		# Determine overall sentiment
		compound_score = vader_scores['compound']
		if compound_score >= 0.05:
			sentiment_label = 'positive'
		elif compound_score <= -0.05:
			sentiment_label = 'negative'
		else:
			sentiment_label = 'neutral'

		return {
			'sentiment_score': compound_score,
			'sentiment_label': sentiment_label,
			'detected_emotions': detected_emotions,
			'emotion_scores': emotion_scores,
			'textblob_polarity': textblob_polarity,
			'textblob_subjectivity': textblob_subjectivity,
			'vader_scores': vader_scores
		}

	def process_message(self, text: str) -> dict:
		"""Process a single message through the enhanced NLP pipeline"""
		# Extract Discord features BEFORE cleaning (like in TP1)
		discord_features = self.extract_discord_features(text)

		cleaned_text = self.clean_text(text)
		tokens = self.tokenize_text(cleaned_text)
		filtered_tokens = self.remove_stopwords(tokens)
		lemmas = self.lemmatize_tokens(filtered_tokens)

		emotions = self.analyze_emotions(cleaned_text, lemmas)

		return {
			'original': text,
			'cleaned': cleaned_text,
			'tokens': tokens,
			'filtered_tokens': filtered_tokens,
			'lemmas': lemmas,
			'processed_text': ' '.join(lemmas),
			'emotions': emotions,
			'discord_features': discord_features
		}


def find_first_csv(data_dir: str = "data") -> Path:
	"""Find the first CSV file in data directory (alphabetical order)"""
	data_path = Path(data_dir)

	if not data_path.exists():
		raise FileNotFoundError(f"Data directory '{data_dir}' not found")

	csv_files = sorted(data_path.glob("*.csv"))

	if not csv_files:
		raise FileNotFoundError(f"No CSV files found in '{data_dir}' directory")

	return csv_files[0]


def load_discord_csv(file_path: Path) -> pd.DataFrame:
	"""Load Discord CSV file"""
	logger.info(f"📂 Loading {file_path.name}...")

	df = pd.read_csv(file_path)
	df['Date'] = pd.to_datetime(df['Date'], utc=True)

	logger.info(f"✅ {len(df)} messages loaded")
	return df


def process_messages(df: pd.DataFrame, processor: DiscordNLPProcessor) -> pd.DataFrame:
	"""Process all messages with enhanced progress tracking"""
	logger.info("🔄 Processing messages with enhanced pipeline...")

	processed_data = []
	batch_count = 0

	for idx, row in df.iterrows():
		if idx % BATCH_SIZE == 0:
			batch_count += 1
			logger.info(f"Processing batch {batch_count} (messages {idx}-{min(idx+BATCH_SIZE-1, len(df)-1)})")

		result = processor.process_message(row['Content'])

		processed_data.append({
			'AuthorID': row['AuthorID'],
			'Author': row['Author'],
			'Date': row['Date'],
			'original_content': result['original'],
			'cleaned_content': result['cleaned'],
			'processed_text': result['processed_text'],
			'token_count': len(result['tokens']),
			'lemma_count': len(result['lemmas']),
			'sentiment_score': result['emotions']['sentiment_score'],
			'sentiment_label': result['emotions']['sentiment_label'],
			'detected_emotions': ','.join(result['emotions']['detected_emotions']),
			'textblob_polarity': result['emotions']['textblob_polarity'],
			'textblob_subjectivity': result['emotions']['textblob_subjectivity'],
			'emotions': result['emotions']  # Keep full emotions data for conversation analysis
		})

	processed_df = pd.DataFrame(processed_data)
	logger.info("✅ Enhanced processing complete")

	return processed_df


def save_results(processed_df: pd.DataFrame, output_dir: str = OUTPUT_DIR):
	"""Save processing results with enhanced analysis"""
	output_path = Path(output_dir)
	output_path.mkdir(exist_ok=True)

	# Save processed messages
	output_file = output_path / 'messages_processed.csv'
	processed_df.to_csv(output_file, index=False)
	logger.info(f"✅ Results saved to {output_file}")

	# Enhanced statistics
	total_messages = len(processed_df)
	empty_messages = len(processed_df[processed_df['processed_text'] == ''])
	non_empty_df = processed_df[processed_df['processed_text'] != '']

	if len(non_empty_df) > 0:
		avg_tokens = non_empty_df['token_count'].mean()
		avg_lemmas = non_empty_df['lemma_count'].mean()
		median_tokens = non_empty_df['token_count'].median()
		median_lemmas = non_empty_df['lemma_count'].median()

		# Additional statistics
		max_tokens = non_empty_df['token_count'].max()
		max_lemmas = non_empty_df['lemma_count'].max()
		std_tokens = non_empty_df['token_count'].std()
		std_lemmas = non_empty_df['lemma_count'].std()
	else:
		avg_tokens = avg_lemmas = median_tokens = median_lemmas = 0
		max_tokens = max_lemmas = std_tokens = std_lemmas = 0

	logger.info("📊 Enhanced Processing Statistics:")
	logger.info(f"  Total messages: {total_messages:,}")
	logger.info(f"  Empty after processing: {empty_messages:,} ({empty_messages/total_messages*100:.1f}%)")
	logger.info(f"  Messages with content: {len(non_empty_df):,}")
	logger.info(f"  Token statistics - Avg: {avg_tokens:.1f}, Median: {median_tokens:.1f}, Max: {max_tokens}, Std: {std_tokens:.1f}")
	logger.info(f"  Lemma statistics - Avg: {avg_lemmas:.1f}, Median: {median_lemmas:.1f}, Max: {max_lemmas}, Std: {std_lemmas:.1f}")

	# Enhanced frequency analysis
	if len(non_empty_df) > 0:
		# Most common lemmas
		all_lemmas = []
		for text in non_empty_df['processed_text']:
			if text:
				all_lemmas.extend(text.split())

		if all_lemmas:
			lemma_counts = Counter(all_lemmas)
			common_lemmas = lemma_counts.most_common(STATS_TOP_LEMMAS)

			logger.info(f"🔤 Top {STATS_TOP_LEMMAS} most common lemmas:")
			for lemma, count in common_lemmas:
				percentage = (count / len(all_lemmas)) * 100
				logger.info(f"  {lemma}: {count} ({percentage:.2f}%)")

			# Save comprehensive word frequency analysis
			freq_df = pd.DataFrame(
				lemma_counts.most_common(FREQ_ANALYSIS_TOP_N),
				columns=['lemma', 'count']
			)
			freq_df['percentage'] = (freq_df['count'] / len(all_lemmas)) * 100
			freq_df['cumulative_percentage'] = freq_df['percentage'].cumsum()

			freq_file = output_path / 'lemma_frequency.csv'
			freq_df.to_csv(freq_file, index=False)
			logger.info(f"✅ Lemma frequency analysis saved to {freq_file}")

			# Additional insights
			unique_lemmas = len(lemma_counts)
			vocabulary_richness = unique_lemmas / len(all_lemmas) if all_lemmas else 0

			logger.info(f"📈 Vocabulary Analysis:")
			logger.info(f"  Total lemmas: {len(all_lemmas):,}")
			logger.info(f"  Unique lemmas: {unique_lemmas:,}")
			logger.info(f"  Vocabulary richness: {vocabulary_richness:.4f}")
			logger.info(f"  Top 10 lemmas cover: {freq_df.head(10)['percentage'].sum():.1f}% of text")


def group_conversations(df: pd.DataFrame) -> list[dict]:
	"""Group messages into conversations based on temporal proximity"""
	logger.info("🔗 Grouping messages into conversations...")

	# Sort by date
	df_sorted = df.sort_values('Date').reset_index(drop=True)

	conversations = []
	current_conversation = []
	last_timestamp = None

	for idx, row in df_sorted.iterrows():
		current_timestamp = row['Date']

		# If this is the first message or gap is too large, start new conversation
		if (last_timestamp is None or
			(current_timestamp - last_timestamp).total_seconds() > CONVERSATION_GAP_MINUTES * 60):

			# Save previous conversation if it has enough messages
			if len(current_conversation) >= MIN_MESSAGES_PER_CONVERSATION:
				conversations.append({
					'conversation_id': len(conversations),
					'start_time': current_conversation[0]['Date'],
					'end_time': current_conversation[-1]['Date'],
					'message_count': len(current_conversation),
					'messages': current_conversation,
					'participants': list(set(msg['Author'] for msg in current_conversation))
				})

			# Start new conversation
			current_conversation = []

		# Add message to current conversation
		current_conversation.append({
			'AuthorID': row['AuthorID'],
			'Author': row['Author'],
			'Date': row['Date'],
			'original_content': row['original_content'],
			'cleaned_content': row['cleaned_content'],
			'processed_text': row['processed_text'],
			'emotions': row['emotions']
		})

		last_timestamp = current_timestamp

	# Don't forget the last conversation
	if len(current_conversation) >= MIN_MESSAGES_PER_CONVERSATION:
		conversations.append({
			'conversation_id': len(conversations),
			'start_time': current_conversation[0]['Date'],
			'end_time': current_conversation[-1]['Date'],
			'message_count': len(current_conversation),
			'messages': current_conversation,
			'participants': list(set(msg['Author'] for msg in current_conversation))
		})

	logger.info(f"✅ Found {len(conversations)} conversations")
	return conversations


def analyze_conversation(conversation: dict) -> dict:
	"""Analyze a single conversation for emotions, topics, and generate summary"""
	# Use original cleaned content for better context
	messages_cleaned = [msg['cleaned_content'] for msg in conversation['messages'] if msg['cleaned_content']]
	messages_processed = [msg['processed_text'] for msg in conversation['messages'] if msg['processed_text']]

	if not messages_cleaned:
		return {
			'summary': 'Conversation vide',
			'title': 'Conversation sans contenu',
			'dominant_emotion': 'neutral',
			'emotion_distribution': {},
			'key_topics': [],
			'sentiment_trend': 'neutral'
		}

	# Combine all text for analysis
	combined_cleaned = ' '.join(messages_cleaned)
	combined_processed = ' '.join(messages_processed)

	# Extract emotions from all messages
	all_emotions = []
	emotion_counts = defaultdict(int)
	sentiment_scores = []

	for msg in conversation['messages']:
		if msg['emotions']:
			emotions_data = msg['emotions']
			sentiment_scores.append(emotions_data.get('sentiment_score', 0))

			# Count detected emotions
			for emotion in emotions_data.get('detected_emotions', []):
				emotion_counts[emotion] += 1
				all_emotions.append(emotion)

	# Determine dominant emotion
	if emotion_counts:
		dominant_emotion = max(emotion_counts.items(), key=lambda x: x[1])[0]
		emotion_distribution = dict(emotion_counts)
	else:
		dominant_emotion = 'neutral'
		emotion_distribution = {}

	# Calculate sentiment trend
	if sentiment_scores:
		avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
		if avg_sentiment > 0.1:
			sentiment_trend = 'positive'
		elif avg_sentiment < -0.1:
			sentiment_trend = 'negative'
		else:
			sentiment_trend = 'neutral'
	else:
		sentiment_trend = 'neutral'

	# Define French stopwords to filter out
	french_stopwords_extended = {
		'le', 'la', 'les', 'un', 'une', 'des', 'de', 'du', 'ce', 'cet', 'cette', 'ces',
		'mon', 'ma', 'mes', 'ton', 'ta', 'tes', 'son', 'sa', 'ses', 'notre', 'nos', 'votre', 'vos', 'leur', 'leurs',
		'je', 'tu', 'il', 'elle', 'on', 'nous', 'vous', 'ils', 'elles',
		'être', 'avoir', 'faire', 'dire', 'aller', 'voir', 'savoir', 'pouvoir', 'vouloir',
		'est', 'es', 'sommes', 'êtes', 'sont', 'était', 'étaient', 'été',
		'ai', 'as', 'a', 'avons', 'avez', 'ont', 'avait', 'avaient', 'eu',
		'fais', 'fait', 'faisons', 'faites', 'font', 'faisait', 'faisaient',
		'dis', 'dit', 'disons', 'dites', 'disent', 'disait', 'disaient',
		'vais', 'vas', 'va', 'allons', 'allez', 'vont', 'allait', 'allaient',
		'que', 'qui', 'quoi', 'où', 'quand', 'comment', 'pourquoi',
		'et', 'ou', 'mais', 'donc', 'or', 'ni', 'car', 'si',
		'à', 'de', 'par', 'pour', 'avec', 'sans', 'sous', 'sur', 'dans', 'en', 'vers', 'chez',
		'ne', 'pas', 'plus', 'moins', 'très', 'trop', 'assez', 'peu', 'beaucoup',
		'ce est', 'cela', 'ça', 'ceci', 'celui', 'celle', 'ceux', 'celles',
		'y', 'lui', 'se', 'me', 'te', 'nous', 'vous', 'leur',
		'alors', 'ainsi', 'après', 'avant', 'aussi', 'comme', 'depuis', 'encore', 'enfin',
		'même', 'peut', 'tout', 'tous', 'toute', 'toutes'
	}

	# Extract key topics using TF-IDF on cleaned content with better filtering
	try:
		# Custom analyzer to filter stopwords
		def custom_analyzer(text):
			words = re.findall(r'\b[a-zA-ZÀ-ÿ]+\b', text.lower())
			# Filter stopwords and short words
			return [w for w in words if w not in french_stopwords_extended and len(w) > 2]

		vectorizer = TfidfVectorizer(
			max_features=30,
			ngram_range=(1, 3),
			min_df=1,
			analyzer=custom_analyzer
		)

		# Analyze each message separately to find most relevant terms
		if len(messages_cleaned) > 1:
			tfidf_matrix = vectorizer.fit_transform(messages_cleaned)
			feature_names = vectorizer.get_feature_names_out()

			# Get average TF-IDF scores across all messages
			avg_scores = tfidf_matrix.mean(axis=0).A1
			term_scores = list(zip(feature_names, avg_scores))
			term_scores.sort(key=lambda x: x[1], reverse=True)

			# Filter to get meaningful topics
			key_topics = []
			for term, score in term_scores:
				# Additional filtering for quality
				if score > 0.05 and not any(word in french_stopwords_extended for word in term.split()):
					key_topics.append(term)
					if len(key_topics) >= 10:
						break
		else:
			# For single message, extract nouns and important words
			words = custom_analyzer(combined_cleaned)
			word_counts = Counter(words)
			key_topics = [word for word, count in word_counts.most_common(10)]

	except:
		# Fallback to noun extraction from processed text
		words = combined_processed.split()
		# Filter common words and keep only substantive terms
		substantive_words = []
		for word in words:
			if len(word) > 3 and word.lower() not in french_stopwords_extended:
				substantive_words.append(word)
		word_counts = Counter(substantive_words)
		key_topics = [word for word, count in word_counts.most_common(10)]

	# Generate better summary - extract most informative messages
	if len(messages_cleaned) <= 3:
		summary = ' | '.join(messages_cleaned[:3])
	else:
		# Find messages with highest information content
		message_scores = []
		for i, msg in enumerate(messages_cleaned):
			# Score based on length, keywords, and position
			words = msg.lower().split()
			score = len([w for w in words if len(w) > 3])  # Meaningful word count

			# Keyword relevance score
			for topic in key_topics[:5]:
				if topic.lower() in msg.lower():
					score += 10

			# Position score (first and last messages are important)
			if i == 0 or i == len(messages_cleaned) - 1:
				score += 5

			# Emotion content score
			if any(emotion_word in msg.lower() for emotion_list in EMOTION_LEXICON_FR.values() for emotion_word in emotion_list):
				score += 3

			message_scores.append((i, score, msg))

		# Sort by score and select top messages
		message_scores.sort(key=lambda x: x[1], reverse=True)
		selected_messages = sorted([msg for i, score, msg in message_scores[:3]],
								   key=lambda msg: messages_cleaned.index(msg))
		summary = ' | '.join(selected_messages)

	# Generate more descriptive title based on key topics and content
	if key_topics:
		# Prioritize multi-word phrases as they're more descriptive
		multiword_topics = [t for t in key_topics if ' ' in t]
		single_word_topics = [t for t in key_topics if ' ' not in t]

		# Build title from most relevant terms
		title_components = []

		# Add multi-word phrases first (more specific)
		for topic in multiword_topics[:2]:
			title_components.append(topic)

		# Add single words if needed
		if len(title_components) < 3:
			for topic in single_word_topics:
				if topic not in ' '.join(title_components):  # Avoid repetition
					title_components.append(topic)
					if len(title_components) >= 3:
						break

		if title_components:
			# Create a natural title
			title = ' '.join(title_components[:3])
			# Capitalize first letter
			title = title[0].upper() + title[1:] if title else title
		else:
			# Extract key subject from first message
			first_msg_words = messages_cleaned[0].split()
			meaningful = [w for w in first_msg_words if len(w) > 4 and w.lower() not in french_stopwords_extended]
			title = ' '.join(meaningful[:3]) if meaningful else f"Conv. {conversation['conversation_id']}"
	else:
		# Last resort - use beginning of first message
		first_words = messages_cleaned[0].split()[:8]
		title = ' '.join(first_words)

	# Add conversation metadata to title
	if len(conversation['participants']) > 3:
		title = f"[Groupe {len(conversation['participants'])}] {title}"
	elif len(conversation['participants']) > 2:
		title = f"[Groupe] {title}"

	# Add emotion indicator if strong
	if emotion_counts and max(emotion_counts.values()) >= 3:
		title = f"{title} [{dominant_emotion}]"

	# Trim summary if too long
	if len(summary) > 500:
		summary = summary[:497] + '...'

	return {
		'summary': summary,
		'title': title[:100],  # Limit title length
		'dominant_emotion': dominant_emotion,
		'emotion_distribution': emotion_distribution,
		'key_topics': key_topics[:MAX_TITLE_WORDS],
		'sentiment_trend': sentiment_trend,
		'participant_count': len(conversation['participants']),
		'duration_minutes': (conversation['end_time'] - conversation['start_time']).total_seconds() / 60
	}


def visualize_conversation_emotions(conversations: list[dict], output_dir: Path):
	"""Visualize emotion distribution across conversations"""
	logger.info("📊 Creating emotion visualizations...")

	# Collect emotion data
	all_emotions = []
	for conv in conversations:
		if 'dominant_emotion' in conv:
			all_emotions.append(conv['dominant_emotion'])

	if not all_emotions:
		logger.warning("No emotion data to visualize")
		return

	# Count emotions
	emotion_counts = Counter(all_emotions)

	# Create pie chart
	plt.figure(figsize=(10, 8))
	colors = {
		'positif': '#5865F2',  # Discord blue
		'négatif': '#ED4245',  # Discord red
		'surprise': '#FEE75C',  # Discord yellow
		'peur': '#EB459E',  # Discord pink
		'colère': '#F47B67',  # Discord orange
		'dégoût': '#5865F2',  # Discord purple
		'confiance': '#57F287',  # Discord green
		'espoir': '#3BA55C',  # Discord light green
		'neutral': '#747F8D'  # Discord gray
	}

	plt.pie(
		emotion_counts.values(),
		labels=emotion_counts.keys(),
		autopct='%1.1f%%',
		colors=[colors.get(e, '#747F8D') for e in emotion_counts.keys()],
		startangle=90
	)

	plt.title('Distribution des émotions dominantes dans les conversations', fontsize=16)

	output_path = output_dir / 'emotion_distribution.png'
	plt.savefig(output_path, bbox_inches='tight')
	plt.close()

	logger.info(f"✅ Emotion visualization saved to {output_path}")





def create_analysis_report(processed_df: pd.DataFrame, conversations: list[dict], output_dir: Path):
	"""Create a comprehensive analysis report with visualizations"""
	logger.info("📝 Creating comprehensive analysis report...")

	report_content = []
	report_content.append("# Rapport d'analyse des conversations Discord\n")
	report_content.append(f"Date de génération: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

	# General statistics
	report_content.append("## Statistiques générales\n")
	report_content.append(f"- Nombre total de messages: {len(processed_df)}\n")
	report_content.append(f"- Nombre de conversations: {len(conversations)}\n")
	report_content.append(f"- Nombre de participants uniques: {processed_df['Author'].nunique()}\n")

	# Top participants
	top_authors = processed_df['Author'].value_counts().head(10)
	report_content.append("\n## Top 10 des participants les plus actifs\n")
	for author, count in top_authors.items():
		report_content.append(f"- {author}: {count} messages\n")

	# Emotion analysis
	emotion_counts = Counter(conv.get('dominant_emotion', 'neutral') for conv in conversations)
	report_content.append("\n## Analyse des émotions\n")
	for emotion, count in emotion_counts.most_common():
		percentage = (count / len(conversations)) * 100
		report_content.append(f"- {emotion}: {count} conversations ({percentage:.1f}%)\n")

	# Most discussed topics
	all_topics = []
	for conv in conversations:
		all_topics.extend(conv.get('key_topics', []))
	topic_counts = Counter(all_topics)

	report_content.append("\n## Sujets les plus discutés\n")
	for topic, count in topic_counts.most_common(20):
		report_content.append(f"- {topic}: {count} occurrences\n")

	# Save report
	report_path = output_dir / 'analysis_report.md'
	with open(report_path, 'w', encoding='utf-8') as f:
		f.writelines(report_content)

	logger.info(f"✅ Analysis report saved to {report_path}")


def main():
	"""Main function"""
	logger.info("🚀 Starting Discord NLP processing")

	try:
		# Find first CSV file
		csv_file = find_first_csv()
		logger.info(f"📁 Found CSV file: {csv_file.name}")

		# Load data
		df = load_discord_csv(csv_file)

		# Initialize processor
		processor = DiscordNLPProcessor()

		# Process messages
		processed_df = process_messages(df, processor)

		# Save results
		save_results(processed_df)

		# Group conversations and analyze them
		logger.info("🔍 Starting conversation analysis...")
		conversations = group_conversations(processed_df)

		# Analyze conversations
		logger.info("📊 Analyzing conversations...")
		conversation_analysis = []
		for i, conversation in enumerate(conversations):
			if i % 10 == 0:
				logger.info(f"Analyzing conversation {i+1}/{len(conversations)}")
			analysis = analyze_conversation(conversation)
			analysis['conversation_id'] = conversation['conversation_id']
			analysis['start_time'] = conversation['start_time']
			analysis['end_time'] = conversation['end_time']
			analysis['message_count'] = conversation['message_count']
			analysis['participants'] = ', '.join(conversation['participants'])
			conversation_analysis.append(analysis)

		# Save conversation analysis
		output_path = Path(OUTPUT_DIR)
		conversation_analysis_df = pd.DataFrame(conversation_analysis)
		conversation_analysis_file = output_path / 'conversation_analysis.csv'
		conversation_analysis_df.to_csv(conversation_analysis_file, index=False)
		logger.info(f"✅ Conversation analysis saved to {conversation_analysis_file}")

		# Print summary statistics
		logger.info("📈 Conversation Analysis Summary:")
		logger.info(f"  Total conversations: {len(conversations)}")
		if conversation_analysis:
			emotions = [conv['dominant_emotion'] for conv in conversation_analysis]
			emotion_counts = Counter(emotions)
			logger.info(f"  Dominant emotions: {dict(emotion_counts)}")

			sentiments = [conv['sentiment_trend'] for conv in conversation_analysis]
			sentiment_counts = Counter(sentiments)
			logger.info(f"  Sentiment trends: {dict(sentiment_counts)}")

		# Generate visualizations
		logger.info("🎨 Generating visualizations...")
		output_path = Path(OUTPUT_DIR)

		# Visualize emotion distribution
		visualize_conversation_emotions(conversation_analysis, output_path)

		# Create analysis report
		create_analysis_report(processed_df, conversation_analysis, output_path)

		logger.info("🎉 Processing completed successfully!")

	except Exception as e:
		logger.error(f"❌ Error: {e}")
		raise


if __name__ == "__main__":
	main()
