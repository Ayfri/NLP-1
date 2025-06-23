#!/usr/bin/env python3
"""
Simplified Autoencoder Testing Pipeline for Discord Message Embeddings
Uses FastText embeddings as input for compression and reconstruction
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import os
import json
from sklearn.metrics import mean_squared_error, silhouette_score
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from gensim.models import FastText
import warnings
warnings.filterwarnings('ignore')

# Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"🖥️ Device: {device}")

# Constants
FASTTEXT_DIM = 100
COMPRESSION_DIM = 32
RECONSTRUCTION_DIM = 1000
BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
RANDOM_STATE = 42

torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

class SimpleAutoencoder(nn.Module):
	"""Autoencoder: 100D → 32D → 1000D → 100D"""

	def __init__(self):
		super().__init__()
		self.encoder = nn.Sequential(
			nn.Linear(FASTTEXT_DIM, COMPRESSION_DIM),
			nn.ReLU(),
			nn.Dropout(0.2)
		)
		self.decoder = nn.Sequential(
			nn.Linear(COMPRESSION_DIM, RECONSTRUCTION_DIM),
			nn.ReLU(),
			nn.Dropout(0.2),
			nn.Linear(RECONSTRUCTION_DIM, FASTTEXT_DIM),
			nn.Tanh()
		)

	def forward(self, x):
		compressed = self.encoder(x)
		reconstructed = self.decoder(compressed)
		return reconstructed, compressed

def load_data():
	"""Load processed messages"""
	df = pd.read_csv("output/messages_processed.csv")
	df = df[df['processed_text'].notna() & (df['processed_text'] != '')]
	df = df[df['processed_text'].str.split().str.len() >= 3]

	if len(df) > 2000:
		df = df.sample(n=2000, random_state=RANDOM_STATE)
		logger.info("⚠️ Limited to 2000 messages")

	logger.info(f"📂 Loaded {len(df)} messages")
	return df['processed_text'].tolist()

def create_embeddings(texts):
	"""Create FastText embeddings"""
	logger.info("🔤 Training FastText...")

	tokenized = [text.split() for text in texts]
	model = FastText(tokenized, vector_size=FASTTEXT_DIM, window=5, min_count=2,
					workers=4, epochs=10, sg=0, min_n=3, max_n=6)

	embeddings = []
	for text in texts:
		vectors = [model.wv[word] for word in text.split() if word in model.wv]
		embeddings.append(np.mean(vectors, axis=0) if vectors else np.zeros(FASTTEXT_DIM))

	embeddings = StandardScaler().fit_transform(embeddings)
	logger.info(f"✅ Created {len(embeddings)} embeddings")
	return embeddings

def train_model(embeddings):
	"""Train autoencoder"""
	# Split data
	n_train = int(len(embeddings) * 0.8)
	indices = np.random.permutation(len(embeddings))
	train_data = torch.FloatTensor(embeddings[indices[:n_train]])
	test_data = torch.FloatTensor(embeddings[indices[n_train:]])

	train_loader = DataLoader(TensorDataset(train_data, train_data), batch_size=BATCH_SIZE, shuffle=True)
	test_loader = DataLoader(TensorDataset(test_data, test_data), batch_size=BATCH_SIZE)

	# Train
	model = SimpleAutoencoder().to(device)
	criterion = nn.MSELoss()
	optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

	logger.info(f"🚀 Training {NUM_EPOCHS} epochs...")
	train_losses, test_losses = [], []

	for epoch in range(NUM_EPOCHS):
		# Training
		model.train()
		train_loss = 0
		for data, target in train_loader:
			data, target = data.to(device), target.to(device)
			optimizer.zero_grad()
			recon, _ = model(data)
			loss = criterion(recon, target)
			loss.backward()
			optimizer.step()
			train_loss += loss.item()

		# Testing
		model.eval()
		test_loss = 0
		with torch.no_grad():
			for data, target in test_loader:
				data, target = data.to(device), target.to(device)
				recon, _ = model(data)
				test_loss += criterion(recon, target).item()

		train_loss /= len(train_loader)
		test_loss /= len(test_loader)
		train_losses.append(train_loss)
		test_losses.append(test_loss)

		if (epoch + 1) % 10 == 0:
			logger.info(f"Epoch {epoch+1}: Train={train_loss:.4f}, Test={test_loss:.4f}")

	logger.info("✅ Training completed")
	return model, train_losses, test_losses, test_loader

def evaluate_model(model, test_loader):
	"""Evaluate performance"""
	logger.info("📊 Evaluating...")

	model.eval()
	all_original, all_recon, all_compressed = [], [], []

	with torch.no_grad():
		for data, _ in test_loader:
			data = data.to(device)
			recon, compressed = model(data)
			all_original.append(data.cpu().numpy())
			all_recon.append(recon.cpu().numpy())
			all_compressed.append(compressed.cpu().numpy())

	original = np.vstack(all_original)
	reconstructed = np.vstack(all_recon)
	compressed = np.vstack(all_compressed)

	# Metrics
	mse = mean_squared_error(original, reconstructed)

	try:
		kmeans = KMeans(n_clusters=5, random_state=RANDOM_STATE)
		labels = kmeans.fit_predict(compressed)
		silhouette = silhouette_score(compressed, labels)
	except:
		silhouette = 0.0

	logger.info(f"📈 Reconstruction MSE: {mse:.4f}")
	logger.info(f"📊 Silhouette score: {silhouette:.4f}")

	return mse, silhouette, original, reconstructed, compressed

def create_plots(train_losses, test_losses, original, reconstructed, compressed):
	"""Create visualizations"""
	os.makedirs("output", exist_ok=True)

	# Training curves
	plt.figure(figsize=(10, 6))
	epochs = range(1, len(train_losses) + 1)
	plt.plot(epochs, train_losses, 'b-', label='Training', linewidth=2)
	plt.plot(epochs, test_losses, 'r-', label='Testing', linewidth=2)
	plt.title('Autoencoder Training Progress')
	plt.xlabel('Epochs')
	plt.ylabel('MSE Loss')
	plt.legend()
	plt.grid(True, alpha=0.3)
	plt.tight_layout()
	plt.savefig('output/autoencoder_training.png', dpi=300, bbox_inches='tight')
	plt.close()

	# t-SNE comparison
	fig, axes = plt.subplots(1, 3, figsize=(15, 5))
	max_samples = min(300, len(original))
	indices = np.random.choice(len(original), max_samples, replace=False)

	data = {'Original': original[indices], 'Reconstructed': reconstructed[indices], 'Compressed': compressed[indices]}

	for idx, (name, embeddings) in enumerate(data.items()):
		tsne = TSNE(n_components=2, perplexity=min(20, len(embeddings)-1), random_state=RANDOM_STATE)
		embeddings_2d = tsne.fit_transform(embeddings)

		axes[idx].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
						 c=range(len(embeddings_2d)), cmap='viridis', alpha=0.7, s=20)
		axes[idx].set_title(f'{name} Embeddings')
		axes[idx].set_xlabel('t-SNE 1')
		axes[idx].set_ylabel('t-SNE 2')

	plt.tight_layout()
	plt.savefig('output/autoencoder_comparison.png', dpi=300, bbox_inches='tight')
	plt.close()

	logger.info("📊 Plots saved")

def main():
	"""Main pipeline"""
	logger.info("🚀 Starting autoencoder pipeline...")

	try:
		# Load and process data
		texts = load_data()
		embeddings = create_embeddings(texts)

		# Train model
		model, train_losses, test_losses, test_loader = train_model(embeddings)

		# Evaluate
		mse, silhouette, original, reconstructed, compressed = evaluate_model(model, test_loader)

		# Visualize
		create_plots(train_losses, test_losses, original, reconstructed, compressed)

		# Save model and results
		os.makedirs("models", exist_ok=True)
		torch.save(model.state_dict(), "models/autoencoder_fasttext.pth")

		results = {
			'reconstruction_mse': mse,
			'silhouette_score': silhouette,
			'compression_ratio': f"{FASTTEXT_DIM}/{COMPRESSION_DIM} = {FASTTEXT_DIM/COMPRESSION_DIM:.1f}x",
			'final_train_loss': train_losses[-1],
			'final_test_loss': test_losses[-1]
		}

		with open("output/autoencoder_results.json", "w") as f:
			json.dump(results, f, indent=2)

		# Final summary with interpretation
		logger.info("🎉 AUTOENCODER COMPLETED!")
		logger.info("📊 RESULTS INTERPRETATION:")

		# MSE interpretation
		mse_quality = "✅ Excellent" if mse < 0.1 else "👍 Good" if mse < 0.5 else "⚠️ Needs work"
		logger.info(f"  🎯 Reconstruction MSE: {mse:.4f} ({mse_quality})")

		# Silhouette interpretation
		sil_quality = "🏆 Excellent" if silhouette > 0.5 else "👍 Good" if silhouette > 0.3 else "⚠️ Poor structure"
		logger.info(f"  📊 Silhouette score: {silhouette:.4f} ({sil_quality})")

		logger.info(f"  🗜️ Compression: {FASTTEXT_DIM/COMPRESSION_DIM:.1f}x size reduction")

		# Overall interpretation
		if mse < 0.3 and silhouette > 0.4:
			logger.info("🌟 EXCELLENT: Autoencoder preserves structure with good compression!")
		elif mse < 0.5 and silhouette > 0.3:
			logger.info("✅ GOOD: Decent compression with acceptable quality")
		else:
			logger.info("⚠️ POOR: Consider adjusting architecture or training longer")

		logger.info("📁 Files generated:")
		logger.info("  - models/autoencoder_fasttext.pth")
		logger.info("  - output/autoencoder_results.json")
		logger.info("  - output/autoencoder_training.png")
		logger.info("  - output/autoencoder_comparison.png")

	except Exception as e:
		logger.error(f"❌ Error: {e}")
		raise

if __name__ == "__main__":
	try:
		main()
	except KeyboardInterrupt:
		logger.info("\n⏹️ Interrupted by user")
	except Exception as e:
		logger.error(f"❌ Error: {e}")
