#!/usr/bin/env python3
"""
NLP processing module for Discord messages
"""

import pandas as pd
import logging
import nltk
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer

from .config import (
	REGEX_PATTERNS, CONTRACTION_PATTERNS, MIN_TOKEN_LENGTH, MAX_TOKEN_LENGTH,
	IMPORTANT_SHORT_WORDS, CONTEXTUAL_WORDS_TO_KEEP, BATCH_SIZE
)

logger = logging.getLogger(__name__)


class DiscordNLPProcessor:
	"""Enhanced Discord message processor with NLP pipeline"""

	def __init__(self):
		"""Initialize the processor"""
		logger.info("Initializing NLP processor...")

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

		logger.info("NLP processor ready")

	def _download_nltk_data(self):
		"""Download required NLTK data"""
		try:
			nltk.data.find('tokenizers/punkt_tab')
		except LookupError:
			nltk.download('punkt_tab')

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
			self.nlp = spacy.load("fr_core_news_sm", disable=["parser", "ner"])
			logger.info("Using French spaCy model")
		except OSError:
			try:
				self.nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
				logger.info("Using English spaCy model")
			except OSError:
				logger.error("No spaCy model found. Install with: python -m spacy download fr_core_news_sm")
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

	def lemmatize_tokens_batch(self, token_lists: list[list[str]]) -> list[list[str]]:
		"""Batch lemmatization with spaCy for better performance"""
		if not token_lists:
			return []

		# Prepare texts for batch processing
		texts = [" ".join(tokens) for tokens in token_lists if tokens]

		if not texts:
			return [[] for _ in token_lists]

		# Process all texts at once
		docs = list(self.nlp.pipe(texts, batch_size=50))

		# Extract lemmas
		results = []
		doc_idx = 0

		for tokens in token_lists:
			if not tokens:
				results.append([])
			else:
				doc = docs[doc_idx]
				lemmas = []
				for token in doc:
					if token.is_alpha and len(token.lemma_) > 1:
						lemmas.append(token.lemma_.lower())
				results.append(lemmas)
				doc_idx += 1

		return results

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
		"""Process a batch of messages with optimized batch processing"""
		# Step 1: Clean all texts
		cleaned_texts = [self.clean_text(msg) for msg in messages]

		# Step 2: Tokenize all texts
		token_lists = [self.tokenize_text(cleaned) for cleaned in cleaned_texts]

		# Step 3: Remove stopwords for all
		filtered_token_lists = [self.remove_stopwords(tokens) for tokens in token_lists]

		# Step 4: Batch lemmatize (optimized)
		lemma_lists = self.lemmatize_tokens_batch(filtered_token_lists)

		# Step 5: Sentiment analysis for all
		sentiments = [self.analyze_sentiment(cleaned) for cleaned in cleaned_texts]

		# Compile results
		results = []
		for i, msg in enumerate(messages):
			results.append({
				'original': msg,
				'cleaned': cleaned_texts[i],
				'tokens': token_lists[i],
				'filtered_tokens': filtered_token_lists[i],
				'lemmas': lemma_lists[i],
				'processed_text': ' '.join(lemma_lists[i]),
				'sentiment': sentiments[i]
			})

		return results
