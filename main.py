#!/usr/bin/env python3
"""
Discord Messages NLP Pipeline
Basic NLP processing: cleaning → tokenization → stopwords removal → lemmatization
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import re
from datetime import datetime

# NLP imports
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Logging configuration
logging.basicConfig(
	level=logging.INFO,
	format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DiscordNLPProcessor:
	"""Discord message processor with basic NLP pipeline"""

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
		self.all_stopwords = self.french_stopwords.union(self.english_stopwords)

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
		"""Clean Discord message text"""
		if pd.isna(text) or text == "":
			return ""

		# Remove Discord mentions (@user)
		text = re.sub(r'<@!?\d+>', '', text)

		# Remove Discord emojis (:emoji:)
		text = re.sub(r':\w+:', '', text)

		# Remove URLs
		text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)

		# Remove special characters but keep basic punctuation
		text = re.sub(r'[^\w\s.,!?;:-]', '', text)

		# Remove multiple spaces
		text = re.sub(r'\s+', ' ', text)

		return text.strip()

	def tokenize_text(self, text: str) -> list[str]:
		"""Tokenize text"""
		if not text:
			return []

		tokens = word_tokenize(text.lower(), language='french')
		tokens = [token for token in tokens if token.isalpha() and len(token) > 2]

		return tokens

	def remove_stopwords(self, tokens: list[str]) -> list[str]:
		"""Remove stopwords"""
		return [token for token in tokens if token not in self.all_stopwords]

	def lemmatize_tokens(self, tokens: list[str]) -> list[str]:
		"""Lemmatize tokens using spaCy"""
		if not tokens:
			return []

		text = " ".join(tokens)
		doc = self.nlp(text)
		lemmas = [token.lemma_.lower() for token in doc if not token.is_stop and token.is_alpha]

		return lemmas

	def process_message(self, text: str) -> dict:
		"""Process a single message through the NLP pipeline"""
		cleaned_text = self.clean_text(text)
		tokens = self.tokenize_text(cleaned_text)
		filtered_tokens = self.remove_stopwords(tokens)
		lemmas = self.lemmatize_tokens(filtered_tokens)

		return {
			'original': text,
			'cleaned': cleaned_text,
			'tokens': tokens,
			'filtered_tokens': filtered_tokens,
			'lemmas': lemmas,
			'processed_text': ' '.join(lemmas)
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
	df['Date'] = pd.to_datetime(df['Date'])

	logger.info(f"✅ {len(df)} messages loaded")
	return df


def process_messages(df: pd.DataFrame, processor: DiscordNLPProcessor) -> pd.DataFrame:
	"""Process all messages in the dataframe"""
	logger.info("🔄 Processing messages...")

	processed_data = []

	for idx, row in df.iterrows():
		if idx % 100 == 0:
			logger.info(f"Processing message {idx}/{len(df)}")

		result = processor.process_message(row['Content'])

		processed_data.append({
			'AuthorID': row['AuthorID'],
			'Author': row['Author'],
			'Date': row['Date'],
			'original_content': result['original'],
			'cleaned_content': result['cleaned'],
			'processed_text': result['processed_text'],
			'token_count': len(result['tokens']),
			'lemma_count': len(result['lemmas'])
		})

	processed_df = pd.DataFrame(processed_data)
	logger.info("✅ Processing complete")

	return processed_df


def save_results(processed_df: pd.DataFrame, output_dir: str = "output"):
	"""Save processing results"""
	output_path = Path(output_dir)
	output_path.mkdir(exist_ok=True)

	# Save processed messages
	output_file = output_path / 'messages_processed.csv'
	processed_df.to_csv(output_file, index=False)
	logger.info(f"✅ Results saved to {output_file}")

	# Print basic statistics
	total_messages = len(processed_df)
	empty_messages = len(processed_df[processed_df['processed_text'] == ''])
	avg_tokens = processed_df['token_count'].mean()
	avg_lemmas = processed_df['lemma_count'].mean()

	logger.info("📊 Processing Statistics:")
	logger.info(f"  Total messages: {total_messages:,}")
	logger.info(f"  Empty after processing: {empty_messages:,}")
	logger.info(f"  Average tokens per message: {avg_tokens:.1f}")
	logger.info(f"  Average lemmas per message: {avg_lemmas:.1f}")


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
