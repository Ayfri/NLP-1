import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from typing import Callable
import pickle
import os
import argparse


class SentimentDataset(Dataset):
	"""PyTorch Dataset for sentiment analysis with tokenization"""

	def __init__(self, df: pd.DataFrame, tokenizer: Callable[[str], list[int]], max_len: int = 50) -> None:
		self.texts = df['Comment'].tolist()
		self.labels = df['label'].tolist()
		self.tokenizer = tokenizer
		self.max_len = max_len

	def __len__(self) -> int:
		return len(self.labels)

	def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
		tokens = self.tokenizer(self.texts[idx])

		# Ensure we have at least one token
		if not tokens:
			tokens = [0]  # Use padding token if no tokens

		tokens = torch.tensor(tokens[:self.max_len], dtype=torch.long)
		label = torch.tensor(self.labels[idx], dtype=torch.long)
		return tokens, label


class SelfAttention(nn.Module):
	"""Self-attention module to capture long-term dependencies"""

	def __init__(self, hidden_dim: int) -> None:
		super().__init__()
		self.query = nn.Linear(hidden_dim, hidden_dim)
		self.key = nn.Linear(hidden_dim, hidden_dim)
		self.value = nn.Linear(hidden_dim, hidden_dim)
		self.scale = hidden_dim ** -0.5

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		Q = self.query(x)
		K = self.key(x)
		V = self.value(x)
		scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
		w = F.softmax(scores, dim=-1)
		context = torch.matmul(w, V)
		return context


class SimpleRNN(nn.Module):
	"""Simple RNN with bidirectional support"""

	def __init__(self, embed_dim: int, hidden_dim: int, bidirectional: bool = True) -> None:
		super().__init__()
		self.rnn = nn.RNN(embed_dim, hidden_dim, bidirectional=bidirectional, batch_first=True)
		self.hidden_dim = hidden_dim * (2 if bidirectional else 1)

	def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
		return self.rnn(x)


class SentimentClassifier(nn.Module):
	"""Sentiment classification model with RNN + Self-Attention"""

	def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, num_classes: int, dropout: float = 0.2) -> None:
		super().__init__()
		self.vocab_size = vocab_size
		self.embed_dim = embed_dim
		self.hidden_dim = hidden_dim
		self.num_classes = num_classes

		self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
		self.encoder = SimpleRNN(embed_dim, hidden_dim, bidirectional=True)
		self.attn = SelfAttention(hidden_dim * 2)
		self.dropout = nn.Dropout(dropout)
		self.fc = nn.Linear(hidden_dim * 2, num_classes)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		# Token embedding
		emb = self.embedding(x)

		# Bidirectional RNN encoding
		enc, _ = self.encoder(emb)

		# Apply self-attention
		context = self.attn(enc)

		# Average pooling and dropout
		pooled = context.mean(dim=1)
		pooled = self.dropout(pooled)

		# Final classification
		return self.fc(pooled)


def collate_batch(batch: list) -> tuple[torch.Tensor, torch.Tensor]:
	"""Collation function for DataLoader with padding"""
	texts, labels = zip(*batch)
	texts = pad_sequence(texts, batch_first=True, padding_value=0)
	labels = torch.stack(labels)
	return texts, labels


def load_data(sample_size: int | None = None) -> pd.DataFrame:
	"""Load sentiment dataset"""
	print("Loading data...")

	# Load dataset - CSV has no header, format is: index,text,sentiment_label
	df = pd.read_csv('data/sentiment_data.csv', header=None, names=['index', 'Comment', 'Sentiment'])
	print(f"Loaded {len(df)} rows")

	# Remove header row if it exists (when first row contains 'Sentiment' as text)
	if df.iloc[0]['Sentiment'] == 'Sentiment':
		df = df.drop(0).reset_index(drop=True)

	# Ensure sentiment values are numeric
	df['Sentiment'] = pd.to_numeric(df['Sentiment'], errors='coerce')

	# Remove rows with invalid sentiment values
	df = df.dropna(subset=['Sentiment'])
	df['Sentiment'] = df['Sentiment'].astype(int)

	print(f"After cleaning: {len(df)} rows")

	# Option to sample dataset for rapid development
	if sample_size and sample_size < len(df):
		df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
		print(f"Sampled to {len(df)} rows for faster training")

	return df


def load_or_create_vocab_and_encoder(train_df: pd.DataFrame, load_existing: bool = True) -> tuple[LabelEncoder, dict[str, int], Callable[[str], list[int]]]:
	"""Load existing vocabulary and label encoder or create new ones"""
	# Ensure output directory exists
	os.makedirs('output', exist_ok=True)

	if load_existing:
		try:
			# Try to load existing vocabulary and label encoder
			with open('output/vocab.pkl', 'rb') as f:
				vocab = pickle.load(f)
			with open('output/label_encoder.pkl', 'rb') as f:
				le = pickle.load(f)
			print("Loaded existing vocabulary and label encoder")

			def simple_tokenizer(text: str) -> list[int]:
				"""Simple tokenizer based on space separation"""
				idxs = []
				for w in text.lower().split():
					token_id = vocab.get(w, 0)
					# Extra safety: ensure token is in valid range
					if token_id >= len(vocab):
						token_id = 0
					idxs.append(token_id)
				return idxs

			return le, vocab, simple_tokenizer

		except FileNotFoundError:
			print("No existing vocabulary found, creating new one...")
	else:
		print("Creating new vocabulary and label encoder (--no-load-vocab specified)...")

	# Build simple vocabulary
	vocab = {'<pad>': 0}

	def simple_tokenizer(text: str) -> list[int]:
		"""Simple tokenizer based on space separation"""
		idxs = []
		for w in text.lower().split():
			if w not in vocab:
				vocab[w] = len(vocab)
			token_id = vocab[w]
			# Extra safety: ensure token is in valid range (shouldn't happen but just in case)
			if token_id < 0:
				token_id = 0
			idxs.append(token_id)
		return idxs

	# Build vocabulary on training set
	for text in train_df['Comment']:
		simple_tokenizer(text)

	print(f"Built vocabulary with {len(vocab)} tokens")

	# Create label encoder
	le = LabelEncoder()

	# Save vocabulary (label encoder will be saved later in preprocess_data)
	with open('output/vocab.pkl', 'wb') as f:
		pickle.dump(vocab, f)

	return le, vocab, simple_tokenizer


def preprocess_data(df: pd.DataFrame, load_existing_vocab: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, LabelEncoder, dict[str, int], Callable[[str], list[int]]]:
	"""Data preparation with tokenization and label encoding"""
	print("Preprocessing data...")

	initial_len = len(df)
	# Drop missing values and duplicates
	df = df[['Comment', 'Sentiment']].dropna()
	df = df.drop_duplicates()
	df = df.reset_index(drop=True)
	print(f"Dropped {initial_len - len(df)} rows")

	# Train/validation split first
	label_counts = df['Sentiment'].value_counts()
	min_class_count = label_counts.min()

	if min_class_count >= 2:
		train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['Sentiment'])
		print("Used stratified split")
	else:
		print(f"Warning: Some classes have only {min_class_count} samples. Using non-stratified split.")
		train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

	# Load or create vocabulary and label encoder
	le, vocab, simple_tokenizer = load_or_create_vocab_and_encoder(train_df, load_existing_vocab)

	# Label encoding
	df['label'] = le.fit_transform(df['Sentiment'])
	train_df['label'] = le.transform(train_df['Sentiment'])
	val_df['label'] = le.transform(val_df['Sentiment'])

	# Ensure labels are in correct range
	num_classes = len(le.classes_)
	valid_mask = (df['label'] >= 0) & (df['label'] < num_classes)
	df = df[valid_mask].copy()

	print(f"Vocabulary size: {len(vocab)}")
	print(f"Number of classes: {num_classes}")
	print(f"Label distribution: {df['label'].value_counts().sort_index().to_dict()}")

	# Save label encoder
	os.makedirs('output', exist_ok=True)
	with open('output/label_encoder.pkl', 'wb') as f:
		pickle.dump(le, f)

	return train_df, val_df, le, vocab, simple_tokenizer


def create_data_loaders(
	train_df: pd.DataFrame,
	val_df: pd.DataFrame,
	tokenizer: Callable[[str], list[int]],
	batch_size: int = 64,
	max_len: int = 50,
	device: torch.device = torch.device('cpu')
) -> tuple[DataLoader, DataLoader]:
	"""Create DataLoaders for training and validation"""

	train_ds = SentimentDataset(train_df, tokenizer, max_len=max_len)
	val_ds = SentimentDataset(val_df, tokenizer, max_len=max_len)

	# Configure DataLoader based on device
	pin_memory = device.type == 'cuda'
	num_workers = 0  # Use 0 workers to avoid pickling issues with local tokenizer function

	train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_batch, num_workers=num_workers, pin_memory=pin_memory)
	val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_batch, num_workers=num_workers, pin_memory=pin_memory)

	print(f"Created data loaders: {len(train_loader)} training batches, {len(val_loader)} validation batches")

	return train_loader, val_loader


def train_model(
	model: SentimentClassifier,
	optimizer: torch.optim.Optimizer,
	criterion: nn.Module,
	num_epochs: int,
	train_loader: DataLoader,
	val_loader: DataLoader,
	device: torch.device
) -> None:
	"""Train model with validation"""
	print("Training model...")

	best_val_accuracy = 0
	patience = 5
	patience_counter = 0

	for epoch in range(1, num_epochs + 1):
		# Training phase
		model.train()
		total_loss = 0
		total_samples = 0

		for texts, labels in tqdm(train_loader, desc=f"Epoch {epoch}"):
			texts, labels = texts.to(device), labels.to(device)

			# Safety check for invalid indices
			if labels.max() >= model.num_classes or labels.min() < 0 or texts.max() >= model.vocab_size or texts.min() < 0:
				continue

			optimizer.zero_grad()
			logits = model(texts)
			loss = criterion(logits, labels)
			loss.backward()

			# Gradient clipping for stability
			torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

			optimizer.step()

			total_loss += loss.item() * texts.size(0)
			total_samples += texts.size(0)

		avg_loss = total_loss / total_samples
		print(f"Epoch {epoch} — Loss: {avg_loss:.4f}")

		# Evaluation phase
		model.eval()
		correct = total = 0
		val_loss = 0

		with torch.no_grad():
			for texts, labels in val_loader:
				texts, labels = texts.to(device), labels.to(device)

				# Safety check for invalid indices
				if labels.max() >= model.num_classes or labels.min() < 0 or texts.max() >= model.vocab_size or texts.min() < 0:
					continue

				logits = model(texts)
				loss = criterion(logits, labels)

				val_loss += loss.item() * texts.size(0)
				preds = logits.argmax(dim=1)
				correct += (preds == labels).sum().item()
				total += labels.size(0)

		val_accuracy = correct / total
		avg_val_loss = val_loss / total

		print(f"Validation — Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.4f}")

		# Early stopping
		if val_accuracy > best_val_accuracy:
			best_val_accuracy = val_accuracy
			patience_counter = 0
			# Save best model
			os.makedirs('output', exist_ok=True)
			torch.save(model.state_dict(), 'output/best_attention_model.pth')
			print(f"New best accuracy: {val_accuracy:.4f}")
		else:
			patience_counter += 1

		if patience_counter >= patience:
			print(f"Early stopping at epoch {epoch}")
			break

		print()


def evaluate_model(model: SentimentClassifier, criterion: nn.Module, val_loader: DataLoader, device: torch.device, le: LabelEncoder) -> None:
	"""Detailed model evaluation"""
	print("Evaluating model...")

	model.eval()
	total_loss = 0
	total_samples = 0
	correct_predictions = 0
	class_correct = [0] * len(le.classes_)
	class_total = [0] * len(le.classes_)

	with torch.no_grad():
		for texts, labels in val_loader:
			texts, labels = texts.to(device), labels.to(device)

			# Safety check for invalid indices
			if labels.max() >= len(le.classes_) or labels.min() < 0 or texts.max() >= model.vocab_size or texts.min() < 0:
				continue

			output = model(texts)
			loss = criterion(output, labels)

			total_loss += loss.item() * texts.size(0)
			total_samples += texts.size(0)

			# Calculate accuracy
			_, predicted = torch.max(output.data, 1)
			correct_predictions += (predicted == labels).sum().item()

			# Per-class accuracy
			for i in range(labels.size(0)):
				label = labels[i].item()
				class_total[label] += 1
				if predicted[i] == labels[i]:
					class_correct[label] += 1

	avg_loss = total_loss / total_samples
	overall_accuracy = correct_predictions / total_samples

	# Create sentiment mapping for better display
	sentiment_names = {0: "Negative", 1: "Neutral", 2: "Positive"}

	# Display per-class accuracy
	for i, sentiment_class in enumerate(le.classes_):
		if class_total[i] > 0:
			class_accuracy = class_correct[i] / class_total[i]
			sentiment_name = sentiment_names.get(sentiment_class, f"Class {sentiment_class}")
			print(f"{sentiment_name} accuracy: {class_accuracy:.4f}")

	print(f"Overall accuracy: {overall_accuracy:.4f}")


def test_model(review: str, model: SentimentClassifier, tokenizer: Callable[[str], list[int]], le: LabelEncoder, device: torch.device) -> None:
	"""Test model on user review"""
	tokens = tokenizer(review)

	# Ensure tokens are valid
	if not tokens:
		tokens = [0]  # Use padding token if no tokens

	# Clamp tokens to valid range
	tokens = [min(max(token, 0), model.vocab_size - 1) for token in tokens]

	tokens_tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)  # (1, seq_len)

	model.eval()
	with torch.no_grad():
		output = model(tokens_tensor)
		probabilities = torch.softmax(output, dim=1)
		_, predicted = torch.max(output.data, 1)

		predicted_class = int(predicted.item())

		# Ensure predicted class is valid
		if predicted_class >= len(le.classes_):
			predicted_class = 0  # Default to first class

		# Create sentiment mapping for better display
		sentiment_names = {0: "Negative", 1: "Neutral", 2: "Positive"}
		predicted_sentiment = sentiment_names.get(le.classes_[predicted_class], f"Class {le.classes_[predicted_class]}")
		confidence = probabilities[0][predicted_class].item()

		print(f"Predicted: {predicted_sentiment} (confidence: {confidence:.3f})")

		# Display probabilities with readable names
		prob_dict = {}
		for i, class_label in enumerate(le.classes_):
			sentiment_name = sentiment_names.get(class_label, f"Class {class_label}")
			prob_dict[sentiment_name] = f"{probabilities[0][i].item():.3f}"

		print(f"Probabilities: {prob_dict}")


def parse_args() -> argparse.Namespace:
	"""Parse command line arguments"""
	parser = argparse.ArgumentParser(description='Train sentiment classification model with RNN + Self-Attention')

	parser.add_argument(
		'--no-load-vocab',
		action='store_true',
		help='Create new vocabulary instead of loading existing one (default: load existing)'
    )
	parser.add_argument(
		'--load-model',
		action='store_true',
		help='Load existing best model instead of training new one (default: train new)'
    )
	parser.add_argument(
		'--sample-size',
		type=int,
		default=None,
		help='Sample size for faster development (default: use full dataset)'
    )
	parser.add_argument(
		'--epochs',
		type=int,
		default=6,
		help='Number of training epochs (default: 6)'
    )
	parser.add_argument(
		'--batch-size',
		type=int,
		default=64,
		help='Batch size for training (default: 64)'
    )

	return parser.parse_args()


def main() -> None:
	"""Main function orchestrating training and evaluation"""
	# Parse command line arguments
	args = parse_args()

	# Enable better CUDA error reporting
	os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

	# Hyperparameters
	embed_dim = 100
	hidden_dim = 128
	num_epochs = args.epochs
	batch_size = args.batch_size
	max_len = 50
	sample_size = args.sample_size

	# Device configuration
	if torch.cuda.is_available():
		device = torch.device('cuda')
		print(f"Using device: {device} ({torch.cuda.get_device_name()})")
	else:
		device = torch.device('cpu')
		print(f"CUDA not available, using device: {device}")
		print("To use GPU, ensure CUDA is installed and compatible with PyTorch")

	# Data loading and preprocessing
	df = load_data(sample_size=sample_size)
	train_df, val_df, le, vocab, tokenizer = preprocess_data(df, load_existing_vocab=not args.no_load_vocab)

	# Create DataLoaders
	train_loader, val_loader = create_data_loaders(train_df, val_df, tokenizer, batch_size, max_len, device)

	# Model instantiation
	vocab_size = len(vocab)
	num_classes = len(le.classes_)
	model = SentimentClassifier(vocab_size, embed_dim, hidden_dim, num_classes, dropout=0.2)
	model.to(device)

	# Optimizer and loss function
	optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
	criterion = nn.CrossEntropyLoss()

	print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
	print(f"Training batches: {len(train_loader)}, Validation batches: {len(val_loader)}")

	# Load existing model or train new one
	if args.load_model:
		try:
			model.load_state_dict(torch.load('output/best_attention_model.pth', weights_only=True))
			print("Loaded existing best model (--load-model specified)")
		except FileNotFoundError:
			print("No existing model found, training new one...")
			train_model(model, optimizer, criterion, num_epochs, train_loader, val_loader, device)
	else:
		# Train model
		train_model(model, optimizer, criterion, num_epochs, train_loader, val_loader, device)

		# Load best model after training
		try:
			model.load_state_dict(torch.load('output/best_attention_model.pth', weights_only=True))
			print("Loaded best model")
		except:
			print("Could not load best model, using current model")

	# Final evaluation
	evaluate_model(model, criterion, val_loader, device, le)

	# Interactive testing
	user_input = input("Enter a review (q to quit): ")
	while user_input.lower() != "q":
		test_model(user_input, model, tokenizer, le, device)
		print()
		user_input = input("Enter a review (q to quit): ")


if __name__ == "__main__":
	main()
