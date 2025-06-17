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
from collections import Counter

# NLP imports
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import sklearn.feature_extraction.text
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict

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
	'positif': ['heureux', 'content', 'joie', 'génial', 'super', 'cool', 'bien', 'bon', 'excellent', 'parfait', 'magnifique', 'formidable', 'merveilleux', 'fantastique', 'extraordinaire', 'mdr', 'lol', 'hilarant', 'marrant', 'rigolo'],
	'négatif': ['triste', 'énervé', 'colère', 'mal', 'mauvais', 'nul', 'merde', 'chiant', 'pénible', 'agaçant', 'frustrant', 'horrible', 'terrible', 'affreux', 'catastrophique', 'problème', 'erreur', 'bug'],
	'surprise': ['wow', 'incroyable', 'surprenant', 'étonnant', 'impressionnant', 'ouf', 'waouh', 'dingue', 'fou', 'bizarre', 'étrange'],
	'peur': ['peur', 'angoisse', 'stress', 'inquiet', 'crainte', 'effrayer', 'terroriser', 'paniquer'],
	'dégoût': ['dégoût', 'beurk', 'dégueulasse', 'écœurant', 'répugnant', 'horrible', 'ignoble'],
	'confiance': ['confiance', 'sûr', 'certain', 'évident', 'clair', 'précis', 'exact', 'fiable', 'crédible']
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

	def clean_text(self, text: str) -> str:
		"""Enhanced text cleaning with better French support"""
		if pd.isna(text) or text == "":
			return ""

		# Remove Discord mentions (@user and @user#1234)
		text = re.sub(r'<@!?\d+>', '', text)
		text = re.sub(r'@\w+(?:#\d{4})?', '', text)

		# Remove Discord emojis (:emoji:) and custom emojis
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
			'emotions': emotions
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
	messages_text = [msg['processed_text'] for msg in conversation['messages'] if msg['processed_text']]

	if not messages_text:
		return {
			'summary': 'Conversation vide',
			'title': 'Conversation sans contenu',
			'dominant_emotion': 'neutral',
			'emotion_distribution': {},
			'key_topics': [],
			'sentiment_trend': 'neutral'
		}

	# Combine all processed text
	combined_text = ' '.join(messages_text)

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

	# Extract key topics using TF-IDF
	try:
		vectorizer = TfidfVectorizer(
			max_features=10,
			stop_words=None,  # We already processed the text
			ngram_range=(1, 2)
		)
		tfidf_matrix = vectorizer.fit_transform([combined_text])
		feature_names = vectorizer.get_feature_names_out()
		scores = tfidf_matrix.toarray()[0]

		# Get top scoring terms
		term_scores = list(zip(feature_names, scores))
		term_scores.sort(key=lambda x: x[1], reverse=True)
		key_topics = [term for term, score in term_scores[:MAX_TITLE_WORDS] if score > 0]
	except:
		# Fallback to most common words
		words = combined_text.split()
		word_counts = Counter(words)
		key_topics = [word for word, count in word_counts.most_common(MAX_TITLE_WORDS)]

	# Generate summary (extractive approach - take most representative sentences)
	original_messages = [msg['cleaned_content'] for msg in conversation['messages']
						if msg['cleaned_content'] and len(msg['cleaned_content']) > 10]

	if len(original_messages) <= 3:
		summary = ' | '.join(original_messages[:3])
	else:
		# Take first, middle, and last message as summary
		summary = f"{original_messages[0]} | {original_messages[len(original_messages)//2]} | {original_messages[-1]}"

	# Generate title from key topics and context
	if key_topics:
		title = f"Discussion sur {' '.join(key_topics[:3])}"
		if dominant_emotion != 'neutral':
			title += f" ({dominant_emotion})"
	else:
		title = f"Conversation {conversation['conversation_id']} ({dominant_emotion})"

	return {
		'summary': summary[:500] + '...' if len(summary) > 500 else summary,
		'title': title,
		'dominant_emotion': dominant_emotion,
		'emotion_distribution': emotion_distribution,
		'key_topics': key_topics,
		'sentiment_trend': sentiment_trend,
		'participant_count': len(conversation['participants']),
		'duration_minutes': (conversation['end_time'] - conversation['start_time']).total_seconds() / 60
	}


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

		# Save detailed conversations
		detailed_conversations_file = output_path / 'detailed_conversations.json'
		import json
		with open(detailed_conversations_file, 'w', encoding='utf-8') as f:
			# Convert datetime objects to strings for JSON serialization
			conversations_for_json = []
			for conv in conversations:
				conv_copy = conv.copy()
				conv_copy['start_time'] = conv_copy['start_time'].isoformat()
				conv_copy['end_time'] = conv_copy['end_time'].isoformat()
				for msg in conv_copy['messages']:
					msg['Date'] = msg['Date'].isoformat()
				conversations_for_json.append(conv_copy)

			json.dump(conversations_for_json, f, ensure_ascii=False, indent=2)
		logger.info(f"✅ Detailed conversations saved to {detailed_conversations_file}")

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

		logger.info("🎉 Processing completed successfully!")

	except Exception as e:
		logger.error(f"❌ Error: {e}")
		raise


if __name__ == "__main__":
	main()
