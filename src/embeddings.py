#!/usr/bin/env python3
"""
Embeddings module using FastText for document representation
"""

import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import Optional
import pickle

from gensim.models import FastText
from gensim.models.fasttext import FastText as FastTextModel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

from .config import FASTTEXT_CONFIG, MIN_WORDS_FOR_EMBEDDINGS

logger = logging.getLogger(__name__)


class FastTextEmbedder:
	"""FastText-based document embedder"""

	def __init__(self, config: dict = FASTTEXT_CONFIG):
		"""Initialize the embedder with configuration"""
		self.config = config
		self.model: Optional[FastTextModel] = None
		self.embeddings: Optional[np.ndarray] = None
		self.texts: list[str] = []

	def train(self, texts: list[str]) -> None:
		"""Train FastText model on the provided texts"""
		logger.info("Training FastText model...")

		# No need to filter again - texts are already filtered in the pipeline
		if len(texts) < 10:
			logger.warning("Insufficient texts for training embeddings")
			return

		# Tokenize texts
		tokenized_texts = [text.split() for text in texts]

		# Train FastText model
		self.model = FastText(
			sentences=tokenized_texts,
			**self.config
		)

		# Store texts for later use
		self.texts = texts

		logger.info(f"FastText model trained on {len(texts)} texts, vocab: {len(self.model.wv.key_to_index)}")

	def generate_embeddings(self, texts: Optional[list[str]] = None) -> np.ndarray:
		"""Generate document embeddings using trained model"""
		if self.model is None:
			raise ValueError("Model not trained. Call train() first.")

		if texts is None:
			texts = self.texts

		logger.info(f"Generating embeddings for {len(texts)} documents...")

		doc_embeddings = []
		for text in texts:
			words = text.split()
			word_vectors = []

			for word in words:
				if word in self.model.wv:
					word_vectors.append(self.model.wv[word])

			if word_vectors:
				# Average word vectors to create document embedding
				doc_embedding = np.mean(word_vectors, axis=0)
			else:
				# Fallback to zero vector
				doc_embedding = np.zeros(self.config['vector_size'])

			doc_embeddings.append(doc_embedding)

		self.embeddings = np.array(doc_embeddings)
		logger.info(f"Generated embeddings: {self.embeddings.shape}")

		return self.embeddings

	def save_model(self, file_path: str) -> None:
		"""Save the trained model"""
		if self.model is None:
			raise ValueError("No model to save")

		self.model.save(file_path)
		logger.info(f"Model saved to {file_path}")

	def load_model(self, file_path: str) -> None:
		"""Load a pre-trained model"""
		self.model = FastText.load(file_path)
		logger.info(f"Model loaded from {file_path}")

	def save_embeddings(self, file_path: str) -> None:
		"""Save embeddings to file"""
		if self.embeddings is None:
			raise ValueError("No embeddings to save")

		with open(file_path, 'wb') as f:
			pickle.dump({
				'embeddings': self.embeddings,
				'texts': self.texts
			}, f)
		logger.info(f"Embeddings saved to {file_path}")

	def load_embeddings(self, file_path: str) -> None:
		"""Load embeddings from file"""
		with open(file_path, 'rb') as f:
			data = pickle.load(f)
			self.embeddings = data['embeddings']
			self.texts = data['texts']
		logger.info(f"Embeddings loaded from {file_path}")

	def find_similar_texts(self, query_text: str, top_k: int = 5) -> list[tuple[str, float]]:
		"""Find most similar texts to a query"""
		if self.model is None or self.embeddings is None:
			raise ValueError("Model and embeddings must be available")

		# Generate embedding for query
		query_words = query_text.split()
		query_vectors = [self.model.wv[word] for word in query_words if word in self.model.wv]

		if not query_vectors:
			logger.warning(f"No words in vocabulary for query: {query_text}")
			return []

		query_embedding = np.mean(query_vectors, axis=0).reshape(1, -1)

		# Calculate similarities
		similarities = cosine_similarity(query_embedding, self.embeddings)[0]

		# Get top-k most similar
		top_indices = np.argsort(similarities)[::-1][:top_k]

		results = []
		for idx in top_indices:
			text = self.texts[idx]
			similarity = similarities[idx]
			results.append((text, similarity))

		return results

	def cluster_texts(self, n_clusters: int = 5) -> tuple[np.ndarray, dict]:
		"""Cluster texts using K-means on embeddings"""
		if self.embeddings is None:
			raise ValueError("Embeddings not available. Generate embeddings first.")

		logger.info(f"🎯 Clustering texts into {n_clusters} clusters...")

		# Perform clustering
		kmeans = KMeans(n_clusters=n_clusters, random_state=42)
		labels = kmeans.fit_predict(self.embeddings)

		# Analyze clusters
		cluster_stats = {}
		for cluster_id in range(n_clusters):
			cluster_texts = [self.texts[i] for i, label in enumerate(labels) if label == cluster_id]
			cluster_stats[cluster_id] = {
				'size': len(cluster_texts),
				'texts': cluster_texts[:3]  # Sample texts
			}

		logger.info(f"✅ Clustering completed. Cluster sizes: {[stats['size'] for stats in cluster_stats.values()]}")

		return labels, cluster_stats


def create_embeddings_pipeline(processed_df: pd.DataFrame, output_dir: str = "output") -> FastTextEmbedder:
	"""Create complete embeddings pipeline"""
	logger.info("🚀 Starting embeddings pipeline...")

	# Prepare texts - filter for non-empty and sufficient length
	non_empty_df = processed_df[
		(processed_df['processed_text'].notna()) &
		(processed_df['processed_text'] != '')
	].copy()

	if len(non_empty_df) == 0:
		logger.warning("⚠️ No processed texts available for embeddings")
		return FastTextEmbedder()

	# Filter texts that meet minimum word requirements
	valid_texts = []
	valid_indices = []

	for idx, text in enumerate(non_empty_df['processed_text']):
		if text and len(text.split()) >= MIN_WORDS_FOR_EMBEDDINGS:
			valid_texts.append(text)
			valid_indices.append(idx)

	if len(valid_texts) < 10:
		logger.warning("⚠️ Insufficient texts for meaningful embeddings")
		return FastTextEmbedder()

	# Create DataFrame with only valid texts for clustering
	clustering_df = non_empty_df.iloc[valid_indices].copy()

	# Initialize and train embedder
	embedder = FastTextEmbedder()
	embedder.train(valid_texts)

	if embedder.model is None:
		logger.warning("⚠️ Failed to train embeddings model")
		return embedder

	# Generate embeddings
	embeddings = embedder.generate_embeddings()

	# Save models and embeddings
	output_path = Path(output_dir)
	output_path.mkdir(exist_ok=True)

	model_path = output_path / "fasttext_model.bin"
	embeddings_path = output_path / "embeddings.pkl"

	embedder.save_model(str(model_path))
	embedder.save_embeddings(str(embeddings_path))

	# Perform clustering analysis
	labels, cluster_stats = embedder.cluster_texts()

	# Save clustering results - now labels and clustering_df have same length
	clustering_df['cluster'] = labels

	clustering_file = output_path / "text_clusters.csv"
	clustering_df.to_csv(clustering_file, index=False)

	# Save cluster analysis
	cluster_analysis = []
	for cluster_id, stats in cluster_stats.items():
		cluster_analysis.append({
			'cluster_id': cluster_id,
			'size': stats['size'],
			'sample_texts': ' | '.join(stats['texts'])
		})

	cluster_df = pd.DataFrame(cluster_analysis)
	cluster_file = output_path / "cluster_analysis.csv"
	cluster_df.to_csv(cluster_file, index=False)

	logger.info(f"✅ Embeddings pipeline completed. Files saved to {output_dir}")
	logger.info(f"   Processed {len(valid_texts)} texts out of {len(non_empty_df)} non-empty texts")

	return embedder
