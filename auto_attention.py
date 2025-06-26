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
	print(f"Columns: {df.columns.tolist()}")
	print(f"Sentiment values: {df['Sentiment'].unique()}")

	# Remove header row if it exists (when first row contains 'Sentiment' as text)
	if df.iloc[0]['Sentiment'] == 'Sentiment':
		df = df.drop(0).reset_index(drop=True)
		print("Removed header row")

	# Ensure sentiment values are numeric
	df['Sentiment'] = pd.to_numeric(df['Sentiment'], errors='coerce')

	# Remove rows with invalid sentiment values
	df = df.dropna(subset=['Sentiment'])
	df['Sentiment'] = df['Sentiment'].astype(int)

	print(f"After cleaning: {len(df)} rows")
	print(f"Sentiment values: {sorted(df['Sentiment'].unique())}")
	print(f"Sentiment distribution: {df['Sentiment'].value_counts().sort_index().to_dict()}")

	# Option to sample dataset for rapid development
	if sample_size and sample_size < len(df):
		df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
		print(f"Sampled to {len(df)} rows for faster training")

	return df


def preprocess_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, LabelEncoder, dict[str, int], Callable[[str], list[int]]]:
	"""Data preparation with tokenization and label encoding"""
	print("Preprocessing data...")

	initial_len = len(df)
	# Drop missing values and duplicates
	df = df[['Comment', 'Sentiment']].dropna()
	df = df.drop_duplicates()
	df = df.reset_index(drop=True)
	print(f"Dropped {initial_len - len(df)} rows")

	# Label encoding
	le = LabelEncoder()
	df['label'] = le.fit_transform(df['Sentiment'])  # negative: 0, neutral: 1, positive: 2

	# Ensure labels are in correct range
	num_classes = len(le.classes_)
	print(f"Before filtering - Label range: {df['label'].min()} to {df['label'].max()}")
	print(f"LabelEncoder classes: {le.classes_}")
	print(f"Number of classes: {num_classes}")

	# Ensure all labels are in valid range [0, num_classes-1]
	valid_mask = (df['label'] >= 0) & (df['label'] < num_classes)
	df = df[valid_mask].copy()
	print(f"After filtering - Label range: {df['label'].min()} to {df['label'].max()}")
	print(f"Filtered out {(~valid_mask).sum()} invalid labels")

	# Train/validation split
	label_counts = df['label'].value_counts()
	min_class_count = label_counts.min()

	if min_class_count >= 2:
		train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
		print("Used stratified split")
	else:
		print(f"Warning: Some classes have only {min_class_count} samples. Using non-stratified split.")
		train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

	# Additional validation for train/val splits
	print(f"Train labels range: {train_df['label'].min()} to {train_df['label'].max()}")
	print(f"Val labels range: {val_df['label'].min()} to {val_df['label'].max()}")
	print(f"Train unique labels: {sorted(train_df['label'].unique())}")
	print(f"Val unique labels: {sorted(val_df['label'].unique())}")

	# Build simple vocabulary
	vocab = {'<pad>': 0}

	def simple_tokenizer(text: str) -> list[int]:
		"""Simple tokenizer based on space separation"""
		idxs = []
		for w in text.lower().split():
			if w not in vocab:
				vocab[w] = len(vocab)
			idxs.append(vocab[w])
		return idxs

	# Build vocabulary on training set
	for text in train_df['Comment']:
		simple_tokenizer(text)

		print(f"Vocabulary size: {len(vocab)}")
	print(f"Number of classes: {len(le.classes_)}")
	print(f"Label classes: {le.classes_}")
	print(f"Label distribution: {df['label'].value_counts().sort_index().to_dict()}")
	print(f"Label range: {df['label'].min()} to {df['label'].max()}")

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

	print(f"Created train loader with {len(train_loader)} batches")
	print(f"Created val loader with {len(val_loader)} batches")

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

			# Debug: Check for invalid labels
			if labels.max() >= model.num_classes or labels.min() < 0:
				print(f"Invalid labels detected! Min: {labels.min()}, Max: {labels.max()}, Expected range: [0, {model.num_classes-1}]")
				print(f"Invalid labels: {labels[labels >= model.num_classes]}")
				continue

			# Debug: Check for invalid token indices
			if texts.max() >= model.vocab_size or texts.min() < 0:
				print(f"Invalid token indices detected! Min: {texts.min()}, Max: {texts.max()}, Expected range: [0, {model.vocab_size-1}]")
				print(f"Vocab size: {model.vocab_size}")
				continue

			optimizer.zero_grad()
			logits = model(texts)

			# Debug: Check logits shape
			if logits.shape[1] != model.num_classes:
				print(f"Logits shape mismatch! Got {logits.shape[1]}, expected {model.num_classes}")
				continue

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
			torch.save(model.state_dict(), 'best_attention_model.pth')
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

	# Display per-class accuracy
	for i, sentiment in enumerate(le.classes_):
		if class_total[i] > 0:
			class_accuracy = class_correct[i] / class_total[i]
			print(f"{sentiment} accuracy: {class_accuracy:.4f}")

	print(f"Test Loss: {avg_loss:.4f}, Overall Accuracy: {overall_accuracy:.4f}")


def test_model(review: str, model: SentimentClassifier, tokenizer: Callable[[str], list[int]], le: LabelEncoder, device: torch.device) -> None:
	"""Test model on user review"""
	tokens = tokenizer(review)
	tokens_tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)  # (1, seq_len)

	model.eval()
	with torch.no_grad():
		output = model(tokens_tensor)
		probabilities = torch.softmax(output, dim=1)
		_, predicted = torch.max(output.data, 1)

		predicted_class = int(predicted.item())
		sentiment = le.classes_[predicted_class]
		confidence = probabilities[0][predicted_class].item()

		print(f"Predicted: {sentiment} (confidence: {confidence:.3f})")
		print(f"Probabilities: {dict(zip(le.classes_, probabilities[0].cpu().numpy()))}")


def main() -> None:
	"""Main function orchestrating training and evaluation"""
	# Hyperparameters
	embed_dim = 100
	hidden_dim = 128
	num_epochs = 6
	batch_size = 64
	max_len = 50
	sample_size = None  # Use entire dataset or sample for rapid testing

	# Device configuration
	if torch.cuda.is_available():
		device = torch.device('cpu')  # Force CPU for debugging
		print(f"CUDA available but forcing CPU for debugging")
	else:
		device = torch.device('cpu')
		print(f"CUDA not available, using device: {device}")
		print("To use GPU, ensure CUDA is installed and compatible with PyTorch")

	# Data loading and preprocessing
	df = load_data(sample_size=sample_size)
	train_df, val_df, le, vocab, tokenizer = preprocess_data(df)

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

	# Train model
	train_model(model, optimizer, criterion, num_epochs, train_loader, val_loader, device)

	# Load best model
	try:
		model.load_state_dict(torch.load('best_attention_model.pth'))
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
