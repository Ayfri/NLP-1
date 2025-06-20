#!/usr/bin/env python3
"""
Embeddings Testing Pipeline
Tests and compares different text representation methods:
- BoW/TF-IDF
- TF-IDF + Word2Vec
- FastText
- BERT+ (CamemBERT for French)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import time
from typing import Any
import warnings
warnings.filterwarnings('ignore')

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# Traditional ML embeddings
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Word2Vec and FastText
from gensim.models import Word2Vec, FastText

# BERT and transformers
from transformers import CamembertTokenizer, CamembertModel, AutoTokenizer, AutoModel
import torch

# ================================
# CONSTANTS
# ================================

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Embedding dimensions
TF_IDF_MAX_FEATURES = 5000
WORD2VEC_DIM = 100
FASTTEXT_DIM = 100
BERT_MAX_LENGTH = 512

# Clustering parameters
N_CLUSTERS = 5
RANDOM_STATE = 42

# Visualization parameters
TSNE_PERPLEXITY = 30
TSNE_N_ITER = 1000

# ================================
# EMBEDDING CLASSES
# ================================

class EmbeddingsComparator:
	"""Class to test and compare different embedding methods"""

	def __init__(self, data_path: str = "output/messages_processed.csv"):
		"""Initialize with processed messages data"""
		self.data_path = Path(data_path)
		self.results = {}
		self.embeddings = {}
		self.processing_times = {}

		# Load data
		self.df = self._load_data()
		self.texts = self._prepare_texts()

		logger.info(f"✅ Loaded {len(self.texts)} messages for embedding comparison")

	def _load_data(self) -> pd.DataFrame:
		"""Load processed messages data"""
		if not self.data_path.exists():
			raise FileNotFoundError(f"Processed data not found at {self.data_path}")

		df = pd.read_csv(self.data_path)
		# Filter out empty processed texts
		df = df[df['processed_text'].notna() & (df['processed_text'] != '')]

		logger.info(f"📂 Loaded {len(df)} processed messages")
		return df

	def _prepare_texts(self) -> list[str]:
		"""Prepare texts for embedding"""
		texts = self.df['processed_text'].tolist()
		# Filter out very short texts (less than 3 words)
		texts = [text for text in texts if len(text.split()) >= 3]

		# Limit to first 1000 messages for faster testing
		if len(texts) > 1000:
			texts = texts[:1000]
			logger.info("⚠️ Limited to first 1000 messages for performance")

		return texts

	def test_bow_tfidf(self) -> dict[str, Any]:
		"""Test Bag of Words and TF-IDF embeddings"""
		logger.info("🧮 Testing BoW/TF-IDF embeddings...")
		start_time = time.time()

		# BoW (Count Vectorizer)
		bow_vectorizer = CountVectorizer(
			max_features=TF_IDF_MAX_FEATURES,
			ngram_range=(1, 2),
			min_df=2,
			max_df=0.8
		)
		bow_matrix = bow_vectorizer.fit_transform(self.texts)

		# TF-IDF
		tfidf_vectorizer = TfidfVectorizer(
			max_features=TF_IDF_MAX_FEATURES,
			ngram_range=(1, 2),
			min_df=2,
			max_df=0.8
		)
		tfidf_matrix = tfidf_vectorizer.fit_transform(self.texts)

		processing_time = time.time() - start_time
		self.processing_times['bow_tfidf'] = processing_time

		# Store embeddings
		self.embeddings['bow'] = bow_matrix
		self.embeddings['tfidf'] = tfidf_matrix

		# Clustering evaluation
		bow_clusters = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE).fit_predict(bow_matrix)
		tfidf_clusters = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE).fit_predict(tfidf_matrix)

		bow_silhouette = silhouette_score(bow_matrix, bow_clusters)
		tfidf_silhouette = silhouette_score(tfidf_matrix, tfidf_clusters)

		# Calculate sparsity safely
		bow_sparsity = 0
		try:
			if hasattr(bow_matrix, 'nnz'):
				bow_sparsity = 1 - bow_matrix.nnz / (bow_matrix.shape[0] * bow_matrix.shape[1])
		except AttributeError:
			# Dense matrix or no nnz attribute, skip sparsity calculation
			bow_sparsity = 0

		tfidf_sparsity = 0
		try:
			if hasattr(tfidf_matrix, 'nnz'):
				tfidf_sparsity = 1 - tfidf_matrix.nnz / (tfidf_matrix.shape[0] * tfidf_matrix.shape[1])
		except AttributeError:
			# Dense matrix or no nnz attribute, skip sparsity calculation
			tfidf_sparsity = 0

		results = {
			'bow_features': bow_matrix.shape[1],
			'tfidf_features': tfidf_matrix.shape[1],
			'bow_sparsity': bow_sparsity,
			'tfidf_sparsity': tfidf_sparsity,
			'bow_silhouette_score': bow_silhouette,
			'tfidf_silhouette_score': tfidf_silhouette,
			'processing_time': processing_time
		}

		logger.info(f"✅ BoW/TF-IDF completed in {processing_time:.2f}s")
		return results

	def test_word2vec(self) -> dict[str, Any]:
		"""Test Word2Vec embeddings combined with TF-IDF weighting"""
		logger.info("🔤 Testing Word2Vec embeddings...")
		start_time = time.time()

		# Prepare tokenized texts
		tokenized_texts = [text.split() for text in self.texts]

		# Train Word2Vec model
		w2v_model = Word2Vec(
			sentences=tokenized_texts,
			vector_size=WORD2VEC_DIM,
			window=5,
			min_count=2,
			workers=4,
			epochs=10,
			sg=0  # CBOW
		)

		# Get TF-IDF weights
		tfidf_vectorizer = TfidfVectorizer(min_df=2, max_df=0.8)
		tfidf_vectorizer.fit(self.texts)

		# Create document embeddings using TF-IDF weighted Word2Vec
		doc_embeddings = []
		for text in self.texts:
			words = text.split()
			word_vectors = []
			word_weights = []

			for word in words:
				if word in w2v_model.wv and word in tfidf_vectorizer.vocabulary_:
					word_vectors.append(w2v_model.wv[word])
					# Get TF-IDF weight
					tfidf_weight = tfidf_vectorizer.idf_[tfidf_vectorizer.vocabulary_[word]]
					word_weights.append(tfidf_weight)

			if word_vectors:
				# Weighted average of word vectors
				word_vectors = np.array(word_vectors)
				word_weights = np.array(word_weights)
				doc_embedding = np.average(word_vectors, axis=0, weights=word_weights)
			else:
				# Fallback to zero vector
				doc_embedding = np.zeros(WORD2VEC_DIM)

			doc_embeddings.append(doc_embedding)

		doc_embeddings = np.array(doc_embeddings)

		processing_time = time.time() - start_time
		self.processing_times['word2vec'] = processing_time

		# Store embeddings
		self.embeddings['word2vec'] = doc_embeddings

		# Clustering evaluation
		clusters = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE).fit_predict(doc_embeddings)
		silhouette = silhouette_score(doc_embeddings, clusters)

		results = {
			'vocabulary_size': len(w2v_model.wv.key_to_index),
			'embedding_dim': WORD2VEC_DIM,
			'silhouette_score': silhouette,
			'processing_time': processing_time
		}

		logger.info(f"✅ Word2Vec completed in {processing_time:.2f}s")
		return results

	def test_fasttext(self) -> dict[str, Any]:
		"""Test FastText embeddings"""
		logger.info("⚡ Testing FastText embeddings...")
		start_time = time.time()

		# Prepare tokenized texts
		tokenized_texts = [text.split() for text in self.texts]

		# Train FastText model
		ft_model = FastText(
			sentences=tokenized_texts,
			vector_size=FASTTEXT_DIM,
			window=5,
			min_count=2,
			workers=4,
			epochs=10,
			sg=0,  # CBOW
			min_n=3,  # Character n-grams
			max_n=6
		)

		# Create document embeddings (average of word vectors)
		doc_embeddings = []
		for text in self.texts:
			words = text.split()
			word_vectors = [ft_model.wv[word] for word in words if word in ft_model.wv]

			if word_vectors:
				doc_embedding = np.mean(word_vectors, axis=0)
			else:
				doc_embedding = np.zeros(FASTTEXT_DIM)

			doc_embeddings.append(doc_embedding)

		doc_embeddings = np.array(doc_embeddings)

		processing_time = time.time() - start_time
		self.processing_times['fasttext'] = processing_time

		# Store embeddings
		self.embeddings['fasttext'] = doc_embeddings

		# Clustering evaluation
		clusters = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE).fit_predict(doc_embeddings)
		silhouette = silhouette_score(doc_embeddings, clusters)

		results = {
			'vocabulary_size': len(ft_model.wv.key_to_index),
			'embedding_dim': FASTTEXT_DIM,
			'silhouette_score': silhouette,
			'processing_time': processing_time
		}

		logger.info(f"✅ FastText completed in {processing_time:.2f}s")
		return results

	def test_bert(self) -> dict[str, Any]:
		"""Test BERT embeddings (CamemBERT for French)"""
		logger.info("🤖 Testing BERT+ embeddings (CamemBERT)...")
		start_time = time.time()

		# Initialize model and tokenizer
		model_name = "camembert-base"
		try:
			tokenizer = CamembertTokenizer.from_pretrained(model_name)
			model = CamembertModel.from_pretrained(model_name)
		except:
			# Fallback to multilingual BERT
			logger.warning("CamemBERT not available, using multilingual BERT")
			model_name = "bert-base-multilingual-cased"
			tokenizer = AutoTokenizer.from_pretrained(model_name)
			model = AutoModel.from_pretrained(model_name)

		# Set device
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		model = model.to(device)
		model.eval()

		# Process texts in batches
		batch_size = 16
		doc_embeddings = []

		with torch.no_grad():
			for i in range(0, len(self.texts), batch_size):
				batch_texts = self.texts[i:i + batch_size]

				# Tokenize
				inputs = tokenizer(
					batch_texts,
					padding=True,
					truncation=True,
					max_length=BERT_MAX_LENGTH,
					return_tensors='pt'
				).to(device)

				# Get embeddings
				outputs = model(**inputs)

				# Use [CLS] token embeddings
				cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
				doc_embeddings.extend(cls_embeddings)

		doc_embeddings = np.array(doc_embeddings)

		processing_time = time.time() - start_time
		self.processing_times['bert'] = processing_time

		# Store embeddings
		self.embeddings['bert'] = doc_embeddings

		# Clustering evaluation
		clusters = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE).fit_predict(doc_embeddings)
		silhouette = silhouette_score(doc_embeddings, clusters)

		results = {
			'model_name': model_name,
			'embedding_dim': doc_embeddings.shape[1],
			'silhouette_score': silhouette,
			'processing_time': processing_time
		}

		logger.info(f"✅ BERT+ completed in {processing_time:.2f}s")
		return results

	def run_all_tests(self) -> dict[str, dict[str, Any]]:
		"""Run all embedding tests"""
		logger.info("🚀 Starting comprehensive embedding comparison...")

		all_results = {}

		# Test all methods
		all_results['bow_tfidf'] = self.test_bow_tfidf()
		all_results['word2vec'] = self.test_word2vec()
		all_results['fasttext'] = self.test_fasttext()
		all_results['bert'] = self.test_bert()

		self.results = all_results

		logger.info("✅ All embedding tests completed!")
		return all_results

	def create_visualizations(self, output_dir: str = "output"):
		"""Create comparison visualizations"""
		logger.info("📊 Creating visualizations...")

		output_path = Path(output_dir)
		output_path.mkdir(exist_ok=True)

		# 1. Performance comparison
		self._plot_performance_comparison(output_path)

		# 2. t-SNE visualizations
		self._plot_tsne_comparisons(output_path)

		# 3. Clustering quality comparison
		self._plot_clustering_comparison(output_path)

		logger.info("✅ Visualizations created")

	def _plot_performance_comparison(self, output_path: Path):
		"""Plot processing time comparison"""
		fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

		# Processing times
		methods = list(self.processing_times.keys())
		times = list(self.processing_times.values())

		ax1.bar(methods, times, color=['skyblue', 'lightgreen', 'orange', 'lightcoral'])
		ax1.set_title('Processing Time Comparison')
		ax1.set_ylabel('Time (seconds)')
		ax1.tick_params(axis='x', rotation=45)

		# Silhouette scores
		silhouette_scores = {}
		for method in methods:
			if method == 'bow_tfidf':
				silhouette_scores['BoW'] = self.results[method]['bow_silhouette_score']
				silhouette_scores['TF-IDF'] = self.results[method]['tfidf_silhouette_score']
			else:
				silhouette_scores[method.title()] = self.results[method]['silhouette_score']

		methods_sil = list(silhouette_scores.keys())
		scores_sil = list(silhouette_scores.values())

		ax2.bar(methods_sil, scores_sil, color=['skyblue', 'lightblue', 'lightgreen', 'orange', 'lightcoral'])
		ax2.set_title('Clustering Quality (Silhouette Score)')
		ax2.set_ylabel('Silhouette Score')
		ax2.tick_params(axis='x', rotation=45)
		ax2.set_ylim(0, max(scores_sil) * 1.1)

		plt.tight_layout()
		plt.savefig(output_path / 'embeddings_performance_comparison.png', dpi=300, bbox_inches='tight')
		plt.close()

	def _plot_tsne_comparisons(self, output_path: Path):
		"""Create t-SNE visualizations for all embedding methods"""
		methods_to_plot = ['tfidf', 'word2vec', 'fasttext', 'bert']

		fig, axes = plt.subplots(2, 2, figsize=(16, 12))
		axes = axes.flatten()

		for idx, method in enumerate(methods_to_plot):
			embeddings = self.embeddings[method]

			# Reduce dimensionality for large sparse matrices
			if hasattr(embeddings, 'toarray'):  # Sparse matrix
				embeddings = embeddings.toarray()

			# Apply PCA first if embeddings are high-dimensional
			if embeddings.shape[1] > 50:
				pca = PCA(n_components=50, random_state=RANDOM_STATE)
				embeddings = pca.fit_transform(embeddings)

			# t-SNE
			tsne = TSNE(
				n_components=2,
				perplexity=min(TSNE_PERPLEXITY, len(self.texts) - 1),
				max_iter=TSNE_N_ITER,
				random_state=RANDOM_STATE
			)
			embeddings_2d = tsne.fit_transform(embeddings)

			# Plot
			ax = axes[idx]
			scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
							   c=range(len(embeddings_2d)), cmap='viridis', alpha=0.6)
			ax.set_title(f't-SNE: {method.upper()}')
			ax.set_xlabel('t-SNE 1')
			ax.set_ylabel('t-SNE 2')

		plt.tight_layout()
		plt.savefig(output_path / 'embeddings_tsne_comparison.png', dpi=300, bbox_inches='tight')
		plt.close()

	def _plot_clustering_comparison(self, output_path: Path):
		"""Plot clustering quality metrics"""
		fig, ax = plt.subplots(figsize=(10, 6))

		# Prepare data
		methods = []
		silhouette_scores = []

		for method, results in self.results.items():
			if method == 'bow_tfidf':
				methods.extend(['BoW', 'TF-IDF'])
				silhouette_scores.extend([results['bow_silhouette_score'], results['tfidf_silhouette_score']])
			else:
				methods.append(method.title())
				silhouette_scores.append(results['silhouette_score'])

		# Create bar plot
		bars = ax.bar(methods, silhouette_scores,
					  color=['skyblue', 'lightblue', 'lightgreen', 'orange', 'lightcoral'])

		# Add value labels on bars
		for bar, score in zip(bars, silhouette_scores):
			height = bar.get_height()
			ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
					f'{score:.3f}', ha='center', va='bottom')

		ax.set_title('Clustering Quality Comparison (Silhouette Score)')
		ax.set_ylabel('Silhouette Score')
		ax.set_ylim(0, max(silhouette_scores) * 1.15)
		ax.tick_params(axis='x', rotation=45)

		plt.tight_layout()
		plt.savefig(output_path / 'clustering_quality_comparison.png', dpi=300, bbox_inches='tight')
		plt.close()

	def save_results(self, output_dir: str = "output"):
		"""Save detailed comparison results"""
		output_path = Path(output_dir)
		output_path.mkdir(exist_ok=True)

		# Create summary DataFrame
		summary_data = []

		for method, results in self.results.items():
			if method == 'bow_tfidf':
				# BoW
				summary_data.append({
					'method': 'BoW',
					'embedding_dim': results['bow_features'],
					'silhouette_score': results['bow_silhouette_score'],
					'sparsity': results['bow_sparsity'],
					'processing_time': results['processing_time'] / 2,  # Split time
					'notes': f"Features: {results['bow_features']}"
				})
				# TF-IDF
				summary_data.append({
					'method': 'TF-IDF',
					'embedding_dim': results['tfidf_features'],
					'silhouette_score': results['tfidf_silhouette_score'],
					'sparsity': results['tfidf_sparsity'],
					'processing_time': results['processing_time'] / 2,  # Split time
					'notes': f"Features: {results['tfidf_features']}"
				})
			else:
				summary_data.append({
					'method': method.title(),
					'embedding_dim': results['embedding_dim'],
					'silhouette_score': results['silhouette_score'],
					'sparsity': None,
					'processing_time': results['processing_time'],
					'notes': results.get('model_name', '') or f"Vocab: {results.get('vocabulary_size', 'N/A')}"
				})

		summary_df = pd.DataFrame(summary_data)
		summary_df = summary_df.sort_values('silhouette_score', ascending=False)

		# Save results
		summary_df.to_csv(output_path / 'embeddings_comparison_results.csv', index=False)

		# Create detailed report
		report_lines = [
			"# Rapport de Comparaison des Embeddings\n\n",
			f"**Généré le :** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
			f"**Nombre de documents :** {len(self.texts)}\n",
			f"**Paramètres de clustering :** {N_CLUSTERS} clusters\n\n",
			"## 📊 Résultats Synthétiques\n\n",
			"| Méthode | Dimension | Score Silhouette | Temps (s) | Notes |\n",
			"|---------|-----------|------------------|-----------|-------|\n"
		]

		# Add table rows
		for _, row in summary_df.iterrows():
			sparsity_info = f" (Sparsité: {row['sparsity']:.2f})" if row['sparsity'] is not None else ""
			report_lines.append(
				f"| {row['method']} | {row['embedding_dim']} | {row['silhouette_score']:.3f} | "
				f"{row['processing_time']:.2f} | {row['notes']}{sparsity_info} |\n"
			)

		report_lines.extend([
			"\n## 🔍 Résultats Détaillés\n\n"
		])

		# Add detailed results for each method
		for method, results in self.results.items():
			method_title = method.replace('_', ' ').title()
			report_lines.append(f"### {method_title}\n\n")

			if method == 'bow_tfidf':
				report_lines.extend([
					f"- **BoW Features :** {results['bow_features']}\n",
					f"- **TF-IDF Features :** {results['tfidf_features']}\n",
					f"- **BoW Silhouette Score :** {results['bow_silhouette_score']:.3f}\n",
					f"- **TF-IDF Silhouette Score :** {results['tfidf_silhouette_score']:.3f}\n",
					f"- **BoW Sparsity :** {results['bow_sparsity']:.3f}\n",
					f"- **TF-IDF Sparsity :** {results['tfidf_sparsity']:.3f}\n",
				])
			else:
				for key, value in results.items():
					key_formatted = key.replace('_', ' ').title()
					if isinstance(value, float):
						report_lines.append(f"- **{key_formatted} :** {value:.3f}\n")
					else:
						report_lines.append(f"- **{key_formatted} :** {value}\n")

			report_lines.append("\n")

		# Performance ranking
		report_lines.extend([
			"## 🏆 Classements de Performance\n\n",
			"### Par Qualité de Clustering (Score Silhouette)\n\n"
		])

		for i, (_, row) in enumerate(summary_df.iterrows()):
			emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "📍"
			report_lines.append(f"{emoji} **{i + 1}. {row['method']}** : {row['silhouette_score']:.3f}\n")

		report_lines.extend([
			"\n### Par Vitesse de Traitement\n\n"
		])

		speed_ranking = summary_df.sort_values('processing_time')
		for i, (_, row) in enumerate(speed_ranking.iterrows()):
			emoji = "⚡" if i == 0 else "🚀" if i == 1 else "🏃" if i == 2 else "🐌"
			report_lines.append(f"{emoji} **{i + 1}. {row['method']}** : {row['processing_time']:.2f}s\n")

		# Add recommendations
		best_quality = summary_df.iloc[0]
		fastest = speed_ranking.iloc[0]

		report_lines.extend([
			"\n## 💡 Recommandations\n\n",
			f"- **Meilleure qualité :** {best_quality['method']} (Score: {best_quality['silhouette_score']:.3f})\n",
			f"- **Plus rapide :** {fastest['method']} ({fastest['processing_time']:.2f}s)\n",
			f"- **Équilibré :** Considérer le compromis entre qualité et vitesse selon vos besoins\n\n",
			"## 📈 Visualisations Générées\n\n",
			"- `embeddings_performance_comparison.png` : Comparaison des temps de traitement et scores\n",
			"- `embeddings_tsne_comparison.png` : Visualisations t-SNE pour chaque méthode\n",
			"- `clustering_quality_comparison.png` : Comparaison des scores de qualité\n",
			"- `embeddings_comparison_results.csv` : Résultats détaillés au format CSV\n"
		])

		# Save report
		with open(output_path / 'embeddings_comparison_report.md', 'w', encoding='utf-8') as f:
			f.writelines(report_lines)

		logger.info(f"✅ Results saved to {output_path}")


def main():
	"""Main function to run embeddings comparison"""
	logger.info("🚀 Starting embeddings comparison pipeline...")

	try:
		# Initialize comparator
		comparator = EmbeddingsComparator()

		# Run all tests
		results = comparator.run_all_tests()

		# Create visualizations
		comparator.create_visualizations()

		# Save results
		comparator.save_results()

		# Print summary
		logger.info("📊 COMPARISON SUMMARY:")
		for method, result in results.items():
			if method == 'bow_tfidf':
				logger.info(f"  BoW: Silhouette={result['bow_silhouette_score']:.3f}")
				logger.info(f"  TF-IDF: Silhouette={result['tfidf_silhouette_score']:.3f}")
			else:
				logger.info(f"  {method.title()}: Silhouette={result['silhouette_score']:.3f}")

		logger.info("🎉 Embeddings comparison completed successfully!")

	except Exception as e:
		logger.error(f"❌ Error: {e}")
		raise


if __name__ == "__main__":
	main()
