#!/usr/bin/env python3
"""
NLP Models Testing Script for Discord Message Analysis
Optimized version for fast training with GPU support
"""

import argparse
import logging
import os
import pandas as pd
import torch
from datetime import datetime
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import (
	AutoModelForSeq2SeqLM,
	AutoModelForSequenceClassification,
	AutoTokenizer
)
from transformers.trainer_callback import EarlyStoppingCallback
from transformers.data.data_collator import DataCollatorForSeq2Seq
from transformers.trainer import Trainer
from transformers.trainer_seq2seq import Seq2SeqTrainer
from transformers.training_args import TrainingArguments
from transformers.training_args_seq2seq import Seq2SeqTrainingArguments

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"🖥️ Using device: {device}")
if torch.cuda.is_available():
	logger.info(f"🚀 GPU detected: {torch.cuda.get_device_name(0)}")
	logger.info(f"💾 GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
	logger.info("⚠️ No GPU detected, using CPU")

class DiscordMessagesDataset(Dataset):
	"""Custom dataset for Discord message analysis"""

	def __init__(self, texts, labels, tokenizer, max_length=128, task_type="classification"):
		self.labels = labels
		self.max_length = max_length
		self.task_type = task_type
		self.texts = texts
		self.tokenizer = tokenizer

	def __getitem__(self, idx):
		text = str(self.texts[idx])

		# Input text tokenization
		inputs = self.tokenizer(
			text,
			max_length=self.max_length,
			padding='max_length',
			return_tensors='pt',
			truncation=True
		)

		if self.task_type == "classification":
			return {
				'attention_mask': inputs['attention_mask'].flatten(),
				'input_ids': inputs['input_ids'].flatten(),
				'labels': torch.tensor(self.labels[idx], dtype=torch.long)
			}
		else:  # summarization
			targets = self.tokenizer(
				str(self.labels[idx]),
				max_length=64,
				padding='max_length',
				return_tensors='pt',
				truncation=True
			)
			return {
				'attention_mask': inputs['attention_mask'].flatten(),
				'input_ids': inputs['input_ids'].flatten(),
				'labels': targets['input_ids'].flatten()
			}

	def __len__(self):
		return len(self.texts)

class DiscordConversationDataset(Dataset):
	"""Dataset for Discord conversation summarization"""

	def __init__(self, conversations, summaries, tokenizer, max_input_length=256, max_target_length=64):
		self.conversations = conversations
		self.max_input_length = max_input_length
		self.max_target_length = max_target_length
		self.summaries = summaries
		self.tokenizer = tokenizer

	def __getitem__(self, idx):
		conversation = str(self.conversations[idx])
		summary = str(self.summaries[idx])

		# Conversation tokenization
		inputs = self.tokenizer(
			conversation,
			max_length=self.max_input_length,
			padding='max_length',
			return_tensors='pt',
			truncation=True
		)

		# Summary tokenization
		targets = self.tokenizer(
			summary,
			max_length=self.max_target_length,
			padding='max_length',
			return_tensors='pt',
			truncation=True
		)

		return {
			'attention_mask': inputs['attention_mask'].flatten(),
			'input_ids': inputs['input_ids'].flatten(),
			'labels': targets['input_ids'].flatten()
		}

	def __len__(self):
		return len(self.conversations)

def load_processed_messages(file_path="output/messages_processed.csv"):
	"""Load processed messages from CSV"""
	logger.info(f"📊 Loading messages: {file_path}")

	if not os.path.exists(file_path):
		logger.error(f"❌ File not found: {file_path}")
		return None

	df = pd.read_csv(file_path)
	logger.info(f"✅ Messages loaded: {len(df)} samples")

	# Clean missing values
	df = df.dropna(subset=['processed_text', 'sentiment_score'])
	logger.info(f"🧹 After cleaning: {len(df)} samples")

	return df

def prepare_sentiment_data(df, test_size=0.2, max_samples=2000):  # Increased for GPU
	"""Prepare data for sentiment classification"""
	logger.info("📈 Preparing sentiment data...")

	# Convert sentiment scores to classes
	df['sentiment_class'] = df['sentiment_score'].apply(
		lambda x: 0 if x < -0.1 else (2 if x > 0.1 else 1)  # Negative, Neutral, Positive
	)

	# Limit dataset size for faster training
	if len(df) > max_samples:
		df = df.sample(n=max_samples, random_state=42)
		logger.info(f"📉 Dataset reduced to {max_samples} samples for faster training")

	texts = df['processed_text'].tolist()
	labels = df['sentiment_class'].tolist()

	train_texts, test_texts, train_labels, test_labels = train_test_split(
		texts, labels, test_size=test_size, random_state=42, stratify=labels
	)

	logger.info(f"📊 Sentiment distribution:")
	logger.info(f" - Negative: {labels.count(0)}")
	logger.info(f" - Neutral: {labels.count(1)}")
	logger.info(f" - Positive: {labels.count(2)}")

	return train_texts, test_texts, train_labels, test_labels

def prepare_conversation_data(conversations_file="output/conversations_analysis.csv", test_size=0.2, max_samples=200):  # Increased for GPU
	"""Prepare data for conversation summarization"""
	logger.info("📈 Preparing conversation data...")

	if not os.path.exists(conversations_file):
		logger.warning(f"⚠️ Conversations file not found: {conversations_file}")
		return None, None, None, None

	df = pd.read_csv(conversations_file)
	df = df.dropna(subset=['key_topics', 'top_participants'])

	# Limit dataset size for faster training
	if len(df) > max_samples:
		df = df.sample(n=max_samples, random_state=42)
		logger.info(f"📉 Dataset reduced to {max_samples} samples for faster training")

	# Use top participants + duration info as input and key topics as summary
	# Create a synthetic "conversation description" from available data
	conversations = []
	for _, row in df.iterrows():
		conv_desc = f"Conversation between {row['top_participants']} lasted {row['duration_minutes']:.1f} minutes with {row['message_count']} messages. Dominant emotion: {row['dominant_emotion']}. Sentiment: {row['avg_sentiment_score']:.3f}"
		conversations.append(conv_desc)

	summaries = df['key_topics'].tolist()

	train_conv, test_conv, train_summ, test_summ = train_test_split(
		conversations, summaries, test_size=test_size, random_state=42
	)

	logger.info(f"📊 Conversations prepared:")
	logger.info(f" - Training: {len(train_conv)}")
	logger.info(f" - Test: {len(test_conv)}")

	return train_conv, test_conv, train_summ, test_summ

def setup_sentiment_model(model_name="distilbert-base-multilingual-cased"):
	"""Configure sentiment classification model (optimized for speed)"""
	logger.info(f"🤖 Loading fast sentiment model: {model_name}")

	tokenizer = AutoTokenizer.from_pretrained(model_name)
	model = AutoModelForSequenceClassification.from_pretrained(
		model_name,
		num_labels=3,  # Negative, Neutral, Positive
		ignore_mismatched_sizes=True
	)

	# Move model to GPU if available
	model.to(device)

	if tokenizer.pad_token is None:
		tokenizer.pad_token = tokenizer.eos_token

	logger.info("✅ Fast sentiment model configured")
	return model, tokenizer

def setup_summarization_model(model_name="sshleifer/distilbart-cnn-6-6"):
	"""Configure summarization model (lightweight version)"""
	logger.info(f"🤖 Loading lightweight summarization model: {model_name}")

	tokenizer = AutoTokenizer.from_pretrained(model_name)
	model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

	# Move model to GPU if available
	model.to(device)

	if tokenizer.pad_token is None:
		tokenizer.pad_token = tokenizer.eos_token

	logger.info("✅ Lightweight summarization model configured")
	return model, tokenizer

def train_sentiment_model(train_dataset, test_dataset, model, tokenizer, output_dir="./models/sentiment"):
	"""Train sentiment classification model (fast version)"""
	logger.info("🚀 Starting fast sentiment model training...")

	# Adjust batch sizes based on device
	train_batch_size = 32 if torch.cuda.is_available() else 16
	eval_batch_size = 64 if torch.cuda.is_available() else 16

	training_args = TrainingArguments(
		eval_strategy="steps",
		eval_steps=50,
		learning_rate=3e-5,  # Optimal for GPU
		load_best_model_at_end=True,
		logging_dir=f"{output_dir}/logs",
		logging_steps=10,
		metric_for_best_model="eval_loss",
		num_train_epochs=3,  # Increased for GPU
		output_dir=output_dir,
		per_device_eval_batch_size=eval_batch_size,
		per_device_train_batch_size=train_batch_size,
		report_to=None,
		save_strategy="steps",
		save_steps=50,
		save_total_limit=1,
		weight_decay=0.01,
		warmup_steps=100,
		dataloader_num_workers=0,  # Avoid multiprocessing issues on Windows
		fp16=torch.cuda.is_available(),  # Use mixed precision on GPU
		dataloader_pin_memory=torch.cuda.is_available()
	)

	# Early stopping to prevent overfitting and speed up training
	early_stopping = EarlyStoppingCallback(early_stopping_patience=3)

	trainer = Trainer(
		args=training_args,
		callbacks=[early_stopping],
		eval_dataset=test_dataset,
		model=model,
		train_dataset=train_dataset
	)

	trainer.train()
	trainer.save_model()
	tokenizer.save_pretrained(output_dir)

	logger.info(f"✅ Sentiment model saved to: {output_dir}")
	return trainer

def train_summarization_model(train_dataset, test_dataset, model, tokenizer, output_dir="./models/summarization"):
	"""Train conversation summarization model (fast version)"""
	logger.info("🚀 Starting fast summarization model training...")

	# Adjust batch sizes based on device
	train_batch_size = 16 if torch.cuda.is_available() else 8
	eval_batch_size = 32 if torch.cuda.is_available() else 8

	training_args = Seq2SeqTrainingArguments(
		eval_strategy="steps",
		eval_steps=25,
		learning_rate=3e-5,
		load_best_model_at_end=True,
		logging_dir=f"{output_dir}/logs",
		logging_steps=10,
		metric_for_best_model="eval_loss",
		num_train_epochs=3,  # Increased for GPU
		output_dir=output_dir,
		per_device_eval_batch_size=eval_batch_size,
		per_device_train_batch_size=train_batch_size,
		predict_with_generate=True,
		push_to_hub=False,
		report_to=None,
		save_strategy="steps",
		save_steps=25,
		save_total_limit=1,
		weight_decay=0.01,
		warmup_steps=50,
		dataloader_num_workers=0,
		fp16=torch.cuda.is_available(),  # Use mixed precision on GPU
		dataloader_pin_memory=torch.cuda.is_available()
	)

	data_collator = DataCollatorForSeq2Seq(
		model=model,
		padding=True,
		tokenizer=tokenizer
	)

	early_stopping = EarlyStoppingCallback(early_stopping_patience=2)

	trainer = Seq2SeqTrainer(
		args=training_args,
		callbacks=[early_stopping],
		data_collator=data_collator,
		eval_dataset=test_dataset,
		model=model,
		train_dataset=train_dataset
	)

	trainer.train()
	trainer.save_model()
	tokenizer.save_pretrained(output_dir)

	logger.info(f"✅ Summarization model saved to: {output_dir}")
	return trainer

def evaluate_sentiment_model(trainer, test_dataset, test_labels):
	"""Evaluate sentiment classification model"""
	logger.info("📊 Evaluating sentiment model...")

	predictions = trainer.predict(test_dataset)
	predicted_labels = predictions.predictions.argmax(-1)

	accuracy = accuracy_score(test_labels, predicted_labels)
	report = classification_report(test_labels, predicted_labels,
									target_names=['Negative', 'Neutral', 'Positive'])

	logger.info(f"📈 Accuracy: {accuracy:.4f}")
	logger.info("📋 Classification report:")
	logger.info(f"\n{report}")

	return accuracy, report

def test_model_inference(model, tokenizer, test_text, task_type="sentiment"):
	"""Test model inference on an example"""
	logger.info(f"🧪 Testing {task_type} inference...")

	inputs = tokenizer(
		test_text,
		max_length=128,
		padding='max_length',
		return_tensors='pt',
		truncation=True
	)

	# Move inputs to the same device as model
	inputs = {k: v.to(device) for k, v in inputs.items()}

	with torch.no_grad():
		if task_type == "sentiment":
			outputs = model(**inputs)
			predicted_class = outputs.logits.argmax(-1).item()
			sentiment_labels = ['Negative', 'Neutral', 'Positive']
			result = sentiment_labels[predicted_class]
		else:  # summarization
			summary_ids = model.generate(
				inputs['input_ids'],
				attention_mask=inputs['attention_mask'],
				early_stopping=True,
				max_length=64,
				num_beams=4 if torch.cuda.is_available() else 2  # More beams on GPU
			)
			result = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

	logger.info(f"📝 {task_type.capitalize()} example:")
	logger.info(f"Text: {test_text[:100]}...")
	logger.info(f"Result: {result}")

	return result

def load_existing_model(model_path, model_class, tokenizer_class, num_labels=None, force_retrain=False):
	"""Load existing model and tokenizer if they exist"""
	if force_retrain:
		logger.info("🔄 Force retrain enabled, skipping model loading")
		return None, None, False

	if os.path.exists(model_path) and any(os.scandir(model_path)):
		logger.info(f"📂 Loading existing model from: {model_path}")
		try:
			tokenizer = tokenizer_class.from_pretrained(model_path)
			if num_labels:
				model = model_class.from_pretrained(model_path, num_labels=num_labels)
			else:
				model = model_class.from_pretrained(model_path)
			model.to(device)
			logger.info("✅ Model loaded successfully")
			return model, tokenizer, True
		except Exception as e:
			logger.warning(f"⚠️ Failed to load model: {e}")
			logger.info("🔄 Will train new model instead")
			return None, None, False
	return None, None, False

def main():
	"""Main function for model testing (optimized version)"""
	# Parse command line arguments
	parser = argparse.ArgumentParser(description="Discord NLP Models Testing")
	parser.add_argument('--retrain', action='store_true',
						help='Force retrain models even if they exist')
	args = parser.parse_args()

	print("=== Fast NLP Models Testing for Discord ===\n")
	print(f"🖥️ Using device: {device}")
	if torch.cuda.is_available():
		print(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
		print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

	if args.retrain:
		print("🔄 Force retrain mode enabled")
	else:
		print("📂 Will use existing models if available")
	print("⚡ Optimized version for fast training\n")

	# Create directories
	os.makedirs("models", exist_ok=True)
	os.makedirs("logs", exist_ok=True)

	try:
		# Load data
		df = load_processed_messages()
		if df is None:
			return

		# Test sentiment model
		logger.info("🎯 Testing fast sentiment classification model")
		max_samples = 2000 if torch.cuda.is_available() else 1000
		train_texts, test_texts, train_labels, test_labels = prepare_sentiment_data(df, max_samples=max_samples)

		# Try to load existing model first
		sentiment_model_path = "./models/sentiment"
		model, tokenizer, model_loaded = load_existing_model(
			sentiment_model_path,
			AutoModelForSequenceClassification,
			AutoTokenizer,
			num_labels=3,
			force_retrain=args.retrain
		)

		# If no existing model, create and train new one
		if not model_loaded:
			model, tokenizer = setup_sentiment_model()
			train_dataset = DiscordMessagesDataset(train_texts, train_labels, tokenizer, max_length=128, task_type="classification")
			test_dataset = DiscordMessagesDataset(test_texts, test_labels, tokenizer, max_length=128, task_type="classification")
			sentiment_trainer = train_sentiment_model(train_dataset, test_dataset, model, tokenizer)
		else:
			# Create datasets for evaluation only
			test_dataset = DiscordMessagesDataset(test_texts, test_labels, tokenizer, max_length=128, task_type="classification")
			# Create a minimal trainer for evaluation
			eval_args = TrainingArguments(
				output_dir="./temp_eval",
				per_device_eval_batch_size=64 if torch.cuda.is_available() else 16,
				dataloader_num_workers=0
			)
			sentiment_trainer = Trainer(model=model, args=eval_args, eval_dataset=test_dataset)

		accuracy, report = evaluate_sentiment_model(sentiment_trainer, test_dataset, test_labels)

		# Test sentiment inference
		if test_texts:
			test_model_inference(model, tokenizer, test_texts[0], "sentiment")

		# Test summarization model (if data available)
		logger.info("📝 Testing fast conversation summarization model")
		max_conv_samples = 200 if torch.cuda.is_available() else 50
		train_conv, test_conv, train_summ, test_summ = prepare_conversation_data(max_samples=max_conv_samples)

		if train_conv is not None:
			# Try to load existing summarization model first
			summarization_model_path = "./models/summarization"
			summ_model, summ_tokenizer, summ_model_loaded = load_existing_model(
				summarization_model_path,
				AutoModelForSeq2SeqLM,
				AutoTokenizer,
				force_retrain=args.retrain
			)

			# If no existing model, create and train new one
			if not summ_model_loaded:
				summ_model, summ_tokenizer = setup_summarization_model()
				conv_train_dataset = DiscordConversationDataset(train_conv, train_summ, summ_tokenizer)
				conv_test_dataset = DiscordConversationDataset(test_conv, test_summ, summ_tokenizer)
				summ_trainer = train_summarization_model(conv_train_dataset, conv_test_dataset,
														summ_model, summ_tokenizer)
			else:
				logger.info("📊 Using existing summarization model for inference")

			# Test summarization inference
			if test_conv:
				test_model_inference(summ_model, summ_tokenizer, test_conv[0], "summarization")

		# Save results
		results_file = "models/training_results.txt"
		with open(results_file, "w", encoding="utf-8") as f:
			f.write(f"Fast Training Results - {datetime.now()}\n")
			f.write("="*50 + "\n\n")
			f.write(f"DEVICE USED: {device}\n")
			if torch.cuda.is_available():
				f.write(f"GPU: {torch.cuda.get_device_name(0)}\n")
			f.write("\nMODELS USED:\n")
			f.write("- Sentiment: DistilBERT (fast version)\n")
			f.write("- Summarization: DistilBART (lightweight)\n\n")
			f.write("SENTIMENT CLASSIFICATION:\n")
			f.write(f"Accuracy: {accuracy:.4f}\n")
			f.write(f"Report:\n{report}\n\n")
			f.write(f"Samples processed: {len(df)}\n")

		logger.info(f"✅ Fast testing completed! Results in: {results_file}")
		logger.info(f"🖥️ Training performed on: {device}")
		if torch.cuda.is_available():
			logger.info("🚀 GPU acceleration was used for faster training")

	except Exception as e:
		logger.error(f"❌ Error during testing: {e}")
		raise

if __name__ == "__main__":
	try:
		main()
	except KeyboardInterrupt:
		logger.info("\n\n⏹️ Testing interrupted by user. Goodbye! 👋")
	except Exception as e:
		logger.error(f"❌ Error: {e}")
