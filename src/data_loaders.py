#!/usr/bin/env python3
"""
PyTorch DataLoaders for efficient NLP pipeline processing
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import Any
import logging

from .config import BATCH_SIZE
from .nlp_processor import DiscordNLPProcessor

logger = logging.getLogger(__name__)


class DiscordMessageDataset(Dataset):
	"""PyTorch Dataset for Discord messages"""

	def __init__(self, df: pd.DataFrame):
		"""
		Initialize dataset with Discord messages
		"""
		self.df = df.reset_index(drop=True)
		self.messages = df['Content'].tolist()
		self.authors = df['Author'].tolist() if 'Author' in df.columns else None
		self.dates = df['Date'].tolist() if 'Date' in df.columns else None
		self.author_ids = df['AuthorID'].tolist() if 'AuthorID' in df.columns else None

	def __len__(self) -> int:
		return len(self.messages)

	def __getitem__(self, idx: int) -> dict[str, Any]:
		"""Get a single message and its metadata"""
		item = {'message': self.messages[idx], 'index': idx}

		if self.authors:
			item['author'] = self.authors[idx]
		if self.dates:
			item['date'] = self.dates[idx]
		if self.author_ids:
			item['author_id'] = self.author_ids[idx]

		return item


class EmbeddingDataset(Dataset):
	"""PyTorch Dataset for embeddings and text pairs"""

	def __init__(self, texts: list[str], embeddings: np.ndarray | None = None):
		"""
		Initialize dataset with texts and optional embeddings

		:param texts: List of processed texts
		:param embeddings: Optional precomputed embeddings
		"""
		self.texts = texts
		self.embeddings = embeddings

		if embeddings is not None:
			assert len(texts) == len(embeddings), "Texts and embeddings must have same length"

	def __len__(self) -> int:
		return len(self.texts)

	def __getitem__(self, idx: int) -> dict[str, Any]:
		"""Get text and optional embedding"""
		item = {'text': self.texts[idx], 'index': idx}

		if self.embeddings is not None:
			item['embedding'] = torch.FloatTensor(self.embeddings[idx])

		return item


def collate_messages(batch: list[dict]) -> dict[str, Any]:
	"""Custom collate function for message batches"""
	keys = batch[0].keys()
	collated = {}

	for key in keys:
		values = [item[key] for item in batch]

		if key == 'embedding' and isinstance(values[0], torch.Tensor):
			collated[key] = torch.stack(values)
		elif key in ['index', 'token_count', 'lemma_count']:
			collated[key] = torch.LongTensor(values)
		elif key in ['sentiment_score']:
			collated[key] = torch.FloatTensor(values)
		else:
			collated[key] = values

	return collated


def collate_embeddings(batch: list[dict]) -> dict[str, Any]:
	"""Custom collate function for embedding batches"""
	texts = [item['text'] for item in batch]
	indices = torch.LongTensor([item['index'] for item in batch])

	collated = {'texts': texts, 'indices': indices}

	if 'embedding' in batch[0]:
		embeddings = torch.stack([item['embedding'] for item in batch])
		collated['embeddings'] = embeddings

	return collated


class NLPDataLoader:
	"""DataLoader manager for NLP pipeline"""

	def __init__(self, batch_size: int = BATCH_SIZE):
		"""
		Initialize DataLoader manager

		:param batch_size: Batch size for DataLoaders
		"""
		self.batch_size = batch_size
		self.pin_memory = torch.cuda.is_available()

		logger.info(f"DataLoader config: batch_size={batch_size}, pin_memory={self.pin_memory}")

	def create_message_dataloader(self, df: pd.DataFrame) -> DataLoader:
		"""
		Create DataLoader for Discord messages

		:param df: DataFrame with Discord messages

		:return: DataLoader for messages
		"""
		dataset = DiscordMessageDataset(df)

		return DataLoader(
			dataset,
			batch_size=self.batch_size,
			shuffle=False,
			num_workers=0,
			pin_memory=self.pin_memory,
			collate_fn=collate_messages
		)

	def create_embedding_dataloader(self, texts: list[str], embeddings: np.ndarray | None = None) -> DataLoader:
		"""
		Create DataLoader for embeddings

		:param texts: List of processed texts
		:param embeddings: Optional precomputed embeddings

		:return: DataLoader for embeddings
		"""
		dataset = EmbeddingDataset(texts, embeddings)

		return DataLoader(
			dataset,
			batch_size=self.batch_size,
			shuffle=False,
			num_workers=0,
			pin_memory=self.pin_memory,
			collate_fn=collate_embeddings
		)


def process_messages_with_dataloader(df: pd.DataFrame, processor: DiscordNLPProcessor) -> pd.DataFrame:
	"""
	Process messages using PyTorch DataLoader for efficient batching

	:param df: DataFrame with Discord messages
	:param processor: NLP processor instance

	:return: DataFrame with processed messages
	"""
	logger.info(f"Processing {len(df)} messages with DataLoader...")

	dataloader_manager = NLPDataLoader()
	dataloader = dataloader_manager.create_message_dataloader(df)

	processed_data = []

	for batch_idx, batch in enumerate(dataloader):
		if batch_idx % 10 == 0:
			logger.info(f"Processing batch {batch_idx + 1}/{len(dataloader)}...")

		# Process batch using existing processor
		batch_results = processor.process_batch(batch['message'])

		# Compile results with metadata
		for i, result in enumerate(batch_results):
			processed_data.append({
				'AuthorID': batch['author_id'][i] if 'author_id' in batch else None,
				'Author': batch['author'][i] if 'author' in batch else None,
				'Date': batch['date'][i] if 'date' in batch else None,
				'original_content': result['original'],
				'cleaned_content': result['cleaned'],
				'processed_text': result['processed_text'],
				'token_count': len(result['tokens']),
				'lemma_count': len(result['lemmas']),
				'sentiment_score': result['sentiment']['sentiment_score'],
				'sentiment_label': result['sentiment']['sentiment_label']
			})

	processed_df = pd.DataFrame(processed_data)
	logger.info("✅ DataLoader processing complete")

	return processed_df
