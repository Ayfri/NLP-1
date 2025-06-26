import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import scipy.sparse


class FastLSTM(nn.Module):
	def __init__(self, input_size: int, hidden_size: int, output_size: int, dropout: float = 0.2, *args, **kwargs) -> None:
		super().__init__(*args, **kwargs)

		self.input_size = input_size
		self.hidden_size = hidden_size
		self.output_size = output_size

		# Single LSTM layer (plus rapide qu'un bidirectionnel)
		self.lstm = nn.LSTM(
			input_size=input_size,
			hidden_size=hidden_size,
			num_layers=1,  # Une seule couche pour la vitesse
			dropout=0,
			bidirectional=False,  # Unidirectionnel pour la vitesse
			batch_first=True
		)

		# Architecture plus simple et rapide
		self.fc = nn.Linear(hidden_size, output_size)
		self.dropout = nn.Dropout(dropout)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		# LSTM forward pass
		lstm_out, (h_n, c_n) = self.lstm(x)

		# Utiliser le dernier état caché
		last_hidden = h_n[-1]  # (batch_size, hidden_size)

		# Dropout et sortie
		output = self.fc(self.dropout(last_hidden))

		return output


def load_data(sample_size: int | None = None) -> pd.DataFrame:
	print("Loading data...")
	# Amazon reviews dataset, [Comment, Sentiment]
	df = pd.read_csv('./data/sentiment_data.csv')
	print(f"Loaded {len(df)} rows")

	# Option pour échantillonner le dataset pour développement rapide
	if sample_size and sample_size < len(df):
		df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
		print(f"Sampled to {len(df)} rows for faster training")

	return df


def preprocess_data(df: pd.DataFrame, input_size: int) -> tuple[torch.Tensor, torch.Tensor, TfidfVectorizer]:
	print("Preprocessing data...")

	initial_len = len(df)
	# Drop the rows with missing values
	df = df.dropna()
	df = df.drop_duplicates()
	df = df.reset_index(drop=True)
	print(f"Dropped {initial_len - len(df)} rows")

	# Preprocessing plus rapide
	df['Comment'] = df['Comment'].astype(str).str.lower()

	# TF-IDF optimisé pour la vitesse
	vectorizer = TfidfVectorizer(
		max_features=input_size,
		stop_words='english',
		ngram_range=(1, 1),  # Seulement unigrams pour la vitesse
		min_df=5,  # Plus restrictif pour moins de features
		max_df=0.8,  # Plus restrictif
		sublinear_tf=True,
		dtype=np.float32  # Utiliser float32 directement
	)
	x_text = vectorizer.fit_transform(df['Comment'])

	# Les sentiments sont déjà encodés comme 0, 1, 2
	y_encoded = df['Sentiment'].values

	# Conversion optimisée - Fix linter error
	x_dense = x_text.toarray().astype(np.float32)

	x = torch.tensor(x_dense, dtype=torch.float32).unsqueeze(1)  # (batch_size, seq_len=1, features)
	y = torch.tensor(y_encoded, dtype=torch.long)  # CrossEntropyLoss attend des entiers

	print(f"Preprocessed {len(x)} rows")
	print(f"Sentiment distribution: {np.bincount(y_encoded.astype(int))}")

	return x, y, vectorizer # (batch_size, seq_len=1, features), (batch_size,), TfidfVectorizer


def create_data_loaders(x_train: torch.Tensor, y_train: torch.Tensor, x_val: torch.Tensor, y_val: torch.Tensor, batch_size: int):
	# Create datasets
	train_dataset = TensorDataset(x_train, y_train)
	val_dataset = TensorDataset(x_val, y_val)

	# Create data loaders avec plus de workers pour la vitesse
	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
	val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

	return train_loader, val_loader


def train_model(
	model: FastLSTM,
	optimizer: optim.AdamW,
	scheduler: optim.lr_scheduler.ReduceLROnPlateau,
	criterion: nn.CrossEntropyLoss,
	num_epochs: int,
	train_loader: DataLoader,
	val_loader: DataLoader
) -> None:
	print("Training model...")
	best_val_accuracy = 0
	patience = 8  # Patience réduite
	patience_counter = 0

	for epoch in range(num_epochs):
		model.train()
		total_train_loss = 0
		total_train_samples = 0

		# Training loop avec optimisations
		for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
			# Reset the gradients
			optimizer.zero_grad()
			output = model(batch_x)
			loss = criterion(output, batch_y)

			# Backward pass
			loss.backward()

			# Gradient clipping plus agressif
			torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)

			optimizer.step()

			total_train_loss += loss.item() * batch_x.size(0)
			total_train_samples += batch_x.size(0)

			# Progress indicator
			if batch_idx % 100 == 0:
				print(f"Epoch {epoch+1}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")

		avg_train_loss = total_train_loss / total_train_samples

		# Validation plus fréquente mais plus rapide
		model.eval()
		total_val_loss = 0
		total_val_samples = 0
		correct_predictions = 0

		with torch.no_grad():
			for batch_x, batch_y in val_loader:
				val_output = model(batch_x)
				val_loss = criterion(val_output, batch_y)

				total_val_loss += val_loss.item() * batch_x.size(0)
				total_val_samples += batch_x.size(0)

				# Calculate accuracy
				_, predicted = torch.max(val_output.data, 1)
				correct_predictions += (predicted == batch_y).sum().item()

		avg_val_loss = total_val_loss / total_val_samples
		accuracy = correct_predictions / total_val_samples

		print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.4f}")

		# Learning rate scheduling
		scheduler.step(avg_val_loss)

		# Early stopping
		if accuracy > best_val_accuracy:
			best_val_accuracy = accuracy
			patience_counter = 0
			# Save best model
			torch.save(model.state_dict(), 'best_model.pth')
			print(f"New best accuracy: {accuracy:.4f}")
		else:
			patience_counter += 1

		if patience_counter >= patience:
			print(f"Early stopping at epoch {epoch+1}")
			break


def evaluate_model(model: FastLSTM, criterion: nn.CrossEntropyLoss, val_loader: DataLoader) -> None:
	print("Evaluating model...")

	# Set the model to evaluation mode
	model.eval()
	total_loss = 0
	total_samples = 0
	correct_predictions = 0
	class_correct = [0, 0, 0]
	class_total = [0, 0, 0]

	# Disable gradient computation
	with torch.no_grad():
		for batch_x, batch_y in val_loader:
			# Make a prediction
			output = model(batch_x)
			# Calculate the loss
			loss = criterion(output, batch_y)

			total_loss += loss.item() * batch_x.size(0)
			total_samples += batch_x.size(0)

			# Calculate accuracy
			_, predicted = torch.max(output.data, 1)
			correct_predictions += (predicted == batch_y).sum().item()

			# Per-class accuracy
			for i in range(batch_y.size(0)):
				label = batch_y[i].item()
				class_total[label] += 1
				if predicted[i] == batch_y[i]:
					class_correct[label] += 1

	avg_loss = total_loss / total_samples
	overall_accuracy = correct_predictions / total_samples

	# Calculate per-class accuracy
	sentiment_names = ['Negative', 'Neutral', 'Positive']
	for i in range(3):
		if class_total[i] > 0:
			class_accuracy = class_correct[i] / class_total[i]
			print(f"{sentiment_names[i]} accuracy: {class_accuracy:.4f}")

	# Print the loss and accuracy
	print(f"Test Loss: {avg_loss:.4f}, Overall Accuracy: {overall_accuracy:.4f}")


def test_model(review: str, model: FastLSTM, vectorizer: TfidfVectorizer) -> None:
	# Convert the review to a vector
	review_vector = vectorizer.transform([review])
	# Convert the vector to a tensor - Fix linter error
	review_dense = review_vector.toarray().astype(np.float32)

	review_tensor = torch.tensor(review_dense, dtype=torch.float32).unsqueeze(1)

	model.eval()
	with torch.no_grad():
		# Make a prediction
		output = model(review_tensor)
		probabilities = torch.softmax(output, dim=1)
		_, predicted = torch.max(output.data, 1)

		sentiment_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
		predicted_class = int(predicted.item())
		sentiment = sentiment_map[predicted_class]
		confidence = probabilities[0][predicted_class].item()

		print(f"Predicted: {sentiment} (confidence: {confidence:.3f})")
		print(f"Probabilities: Negative={probabilities[0][0]:.3f}, Neutral={probabilities[0][1]:.3f}, Positive={probabilities[0][2]:.3f}")


def main() -> None:
	# Hyperparamètres optimisés pour la VITESSE
	input_size = 1000  # Réduit encore pour la vitesse
	output_size = 3  # 3 classes: 0, 1, 2
	hidden_size = 64   # Beaucoup plus petit pour la vitesse
	num_epochs = 30    # Moins d'epochs
	batch_size = 128   # Batch plus gros pour l'efficacité

	# Option pour tester rapidement avec un échantillon
	sample_size = 50000  # Utiliser seulement 50k échantillons au lieu de 240k
	print(f"Fast training mode: using {sample_size} samples")

	df = load_data(sample_size=sample_size)
	x, y, vectorizer = preprocess_data(df, input_size)

	# Split data into train and validation sets
	x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

	# Create data loaders for batch processing
	train_loader, val_loader = create_data_loaders(x_train, y_train, x_val, y_val, batch_size)

	model = FastLSTM(input_size, hidden_size, output_size, dropout=0.2)

	# Optimiseur plus agressif
	optimizer = optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.01)
	scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=3)

	# Class weights for imbalanced dataset
	class_counts = np.bincount(y_train.numpy())
	class_weights = torch.FloatTensor(len(class_counts) / class_counts)
	criterion = nn.CrossEntropyLoss(weight=class_weights)

	print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
	print(f"Training batches: {len(train_loader)}, Validation batches: {len(val_loader)}")

	# Train the model
	train_model(model, optimizer, scheduler, criterion, num_epochs, train_loader, val_loader)

	# Load best model
	try:
		model.load_state_dict(torch.load('best_model.pth'))
		print("Loaded best model")
	except:
		print("Could not load best model, using current model")

	evaluate_model(model, criterion, val_loader)

	user_input = input("Enter a review (q to quit): ")
	while user_input.lower() != "q":
		test_model(user_input, model, vectorizer)
		print()
		user_input = input("Enter a review (q to quit): ")


if __name__ == "__main__":
	main()
