#!/usr/bin/env python3
"""
Discord Messages NLP Pipeline
Enhanced NLP processing: cleaning → tokenization → stopwords removal → lemmatization
"""

import pandas as pd
from pathlib import Path
import logging
import re
from collections import Counter

# NLP imports
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer

# ================================
# CONSTANTS - Configuration
# ================================

# Logging configuration
LOG_LEVEL = logging.INFO
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'

# Processing parameters
MIN_TOKEN_LENGTH = 2
MAX_TOKEN_LENGTH = 50
BATCH_SIZE = 500

# Compiled regex patterns for better performance
REGEX_PATTERNS = {
	'mentions': re.compile(r'<@!?\d+>'),
	'mentions_legacy': re.compile(r'@\w+(?:#\d{4})?'),
	'emojis_custom': re.compile(r'<a?:\w+:\d+>'),
	'emojis_standard': re.compile(r':\w+:'),
	'urls': re.compile(r'https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:\w*))?)?'),
	'apostrophes': re.compile(r"([a-zA-Z])'([a-zA-Z])"),
	'special_chars': re.compile(r'[^\w\s.,!?;:\-àáâäçèéêëìíîïñòóôöùúûüÿæœÀÁÂÄÇÈÉÊËÌÍÎÏÑÒÓÔÖÙÚÛÜŸÆŒ]'),
	'multiple_spaces': re.compile(r'\s+')
}

# French contractions mapping
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

# Compile contraction patterns
CONTRACTION_PATTERNS = {re.compile(pattern, re.IGNORECASE): replacement
						for pattern, replacement in FRENCH_CONTRACTIONS.items()}

# Important short words to preserve
IMPORTANT_SHORT_WORDS = {
	'ça', 'si', 'ou', 'et', 'en', 'un', 'le', 'la', 'de', 'du', 'ce', 'se',
	'me', 'te', 'ne', 'je', 'tu', 'il', 'on', 'au', 'ai', 'as', 'va', 'eu'
}

# Contextually important words to keep
CONTEXTUAL_WORDS_TO_KEEP = {
	'pas', 'non', 'oui', 'bien', 'tout', 'tous', 'faire', 'dit', 'voir',
	'très', 'plus', 'moins', 'beaucoup', 'peu', 'assez', 'trop'
}

# Output configuration
OUTPUT_DIR = "output"
FREQ_ANALYSIS_TOP_N = 100
STATS_TOP_LEMMAS = 20

# ================================
# LOGGING SETUP
# ================================

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# ================================
# MAIN CLASS
# ================================

class DiscordNLPProcessor:
	"""Enhanced Discord message processor with NLP pipeline"""
	nlp: spacy.Language
	french_stopwords: set[str]
	english_stopwords: set[str]
	contextual_stopwords: set[str]
	sentiment_analyzer: SentimentIntensityAnalyzer

	def __init__(self):
		"""Initialize the processor"""
		logger.info("🔧 Initializing NLP processor...")

		# Download NLTK resources
		self._download_nltk_data()

		# Load spaCy model
		self._load_spacy_model()

		# Setup stopwords
		self.french_stopwords = set(stopwords.words('french'))
		self.english_stopwords = set(stopwords.words('english'))
		all_stopwords = self.french_stopwords.union(self.english_stopwords)
		self.contextual_stopwords = all_stopwords - CONTEXTUAL_WORDS_TO_KEEP

		# Initialize sentiment analyzer
		self.sentiment_analyzer = SentimentIntensityAnalyzer()

		logger.info("✅ NLP processor initialized")

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
		"""Enhanced text cleaning with French support"""
		if pd.isna(text) or text == "":
			return ""

		# Remove Discord mentions and emojis
		text = REGEX_PATTERNS['mentions'].sub('', text)
		text = REGEX_PATTERNS['mentions_legacy'].sub('', text)
		text = REGEX_PATTERNS['emojis_standard'].sub('', text)
		text = REGEX_PATTERNS['emojis_custom'].sub('', text)
		text = REGEX_PATTERNS['urls'].sub('', text)

		# Handle French contractions
		for pattern, replacement in CONTRACTION_PATTERNS.items():
			text = pattern.sub(replacement, text)

		# Handle apostrophes
		text = REGEX_PATTERNS['apostrophes'].sub(r'\1 \2', text)

		# Remove special characters
		text = REGEX_PATTERNS['special_chars'].sub('', text)

		# Clean up spaces
		text = REGEX_PATTERNS['multiple_spaces'].sub(' ', text)

		return text.strip()

	def tokenize_text(self, text: str) -> list[str]:
		"""Enhanced tokenization with French support"""
		if not text:
			return []

		tokens = word_tokenize(text.lower(), language='french')

		# Filter tokens
		filtered_tokens = []
		for token in tokens:
			if (token.isalpha() and
				(MIN_TOKEN_LENGTH <= len(token) <= MAX_TOKEN_LENGTH or
				 token in IMPORTANT_SHORT_WORDS)):
				filtered_tokens.append(token)

		return filtered_tokens

	def remove_stopwords(self, tokens: list[str]) -> list[str]:
		"""Remove stopwords with contextual preservation"""
		return [token for token in tokens if token not in self.contextual_stopwords]

	def lemmatize_tokens(self, tokens: list[str]) -> list[str]:
		"""Lemmatization with spaCy"""
		if not tokens:
			return []

		text = " ".join(tokens)
		doc = self.nlp(text)

		lemmas = []
		for token in doc:
			if token.is_alpha and len(token.lemma_) > 1:
				lemmas.append(token.lemma_.lower())

		return lemmas

	def analyze_sentiment(self, text: str) -> dict:
		"""Simple sentiment analysis using VADER"""
		if not text:
			return {
				'sentiment_score': 0.0,
				'sentiment_label': 'neutral'
			}

		# VADER sentiment analysis
		scores = self.sentiment_analyzer.polarity_scores(text)
		compound_score = scores['compound']

		# Determine label
		if compound_score >= 0.05:
			sentiment_label = 'positive'
		elif compound_score <= -0.05:
			sentiment_label = 'negative'
		else:
			sentiment_label = 'neutral'

		return {
			'sentiment_score': compound_score,
			'sentiment_label': sentiment_label
		}

	def process_message(self, text: str) -> dict:
		"""Process a single message through the NLP pipeline"""
		# Step 1: Clean text
		cleaned = self.clean_text(text)

		# Step 2: Tokenize
		tokens = self.tokenize_text(cleaned)

		# Step 3: Remove stopwords
		filtered_tokens = self.remove_stopwords(tokens)

		# Step 4: Lemmatize
		lemmas = self.lemmatize_tokens(filtered_tokens)

		# Step 5: Sentiment analysis
		sentiment = self.analyze_sentiment(cleaned)

		return {
			'original': text,
			'cleaned': cleaned,
			'tokens': tokens,
			'filtered_tokens': filtered_tokens,
			'lemmas': lemmas,
			'processed_text': ' '.join(lemmas),
			'sentiment': sentiment
		}

	def process_batch(self, messages: list[str]) -> list[dict]:
		"""Process a batch of messages"""
		return [self.process_message(msg) for msg in messages]


def find_first_csv(data_dir: str = "data") -> Path:
	"""Find the first CSV file in data directory"""
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
	"""Process all messages"""
	logger.info("🔄 Processing messages...")

	processed_data = []
	total_messages = len(df)

	# Process in batches
	for batch_start in range(0, total_messages, BATCH_SIZE):
		batch_end = min(batch_start + BATCH_SIZE, total_messages)
		batch_num = (batch_start // BATCH_SIZE) + 1

		logger.info(f"Processing batch {batch_num} (messages {batch_start}-{batch_end-1})")

		# Extract batch data
		batch_rows = df.iloc[batch_start:batch_end]
		batch_messages = batch_rows['Content'].tolist()

		# Process batch
		batch_results = processor.process_batch(batch_messages)

		# Compile results
		for i, result in enumerate(batch_results):
			row = batch_rows.iloc[i]

			processed_data.append({
				'AuthorID': row['AuthorID'],
				'Author': row['Author'],
				'Date': row['Date'],
				'original_content': result['original'],
				'cleaned_content': result['cleaned'],
				'processed_text': result['processed_text'],
				'token_count': len(result['tokens']),
				'lemma_count': len(result['lemmas']),
				'sentiment_score': result['sentiment']['sentiment_score'],
				'sentiment_label': result['sentiment']['sentiment_label']
			})

	processed_df = pd.DataFrame(processed_data)
	logger.info("✅ Processing complete")

	return processed_df


def save_results(processed_df: pd.DataFrame, output_dir: str = OUTPUT_DIR):
	"""Save processing results"""
	output_path = Path(output_dir)
	output_path.mkdir(exist_ok=True)

	# Save processed messages
	output_file = output_path / 'messages_processed.csv'
	processed_df.to_csv(output_file, index=False)
	logger.info(f"✅ Results saved to {output_file}")

	# Statistics
	total_messages = len(processed_df)
	empty_messages = len(processed_df[processed_df['processed_text'] == ''])
	non_empty_df = processed_df[processed_df['processed_text'] != '']

	if len(non_empty_df) > 0:
		avg_tokens = non_empty_df['token_count'].mean()
		avg_lemmas = non_empty_df['lemma_count'].mean()
	else:
		avg_tokens = avg_lemmas = 0

	logger.info("📊 Processing Statistics:")
	logger.info(f"  Total messages: {total_messages:,}")
	logger.info(f"  Empty after processing: {empty_messages:,} ({empty_messages/total_messages*100:.1f}%)")
	logger.info(f"  Messages with content: {len(non_empty_df):,}")
	logger.info(f"  Average tokens per message: {avg_tokens:.1f}")
	logger.info(f"  Average lemmas per message: {avg_lemmas:.1f}")

	# Sentiment distribution
	sentiment_counts = processed_df['sentiment_label'].value_counts()
	logger.info("📈 Sentiment Distribution:")
	for sentiment, count in sentiment_counts.items():
		percentage = (count / total_messages) * 100
		logger.info(f"  {sentiment}: {count} ({percentage:.1f}%)")

	# Word frequency analysis
	if len(non_empty_df) > 0:
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

			# Save frequency analysis
			freq_df = pd.DataFrame(
				lemma_counts.most_common(FREQ_ANALYSIS_TOP_N),
				columns=['lemma', 'count']
			)
			freq_df['percentage'] = (freq_df['count'] / len(all_lemmas)) * 100

			freq_file = output_path / 'lemma_frequency.csv'
			freq_df.to_csv(freq_file, index=False)
			logger.info(f"✅ Lemma frequency analysis saved to {freq_file}")


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

		logger.info("🎉 Processing completed successfully!")

	except Exception as e:
		logger.error(f"❌ Error: {e}")
		raise


if __name__ == "__main__":
	main()
