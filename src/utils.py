#!/usr/bin/env python3
"""
Utility functions for the NLP pipeline
"""

import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def find_first_csv(data_dir: str = "data") -> Path:
	"""Find the first CSV file in data directory"""
	data_path = Path(data_dir)

	if not data_path.exists():
		raise FileNotFoundError(f"Data directory '{data_dir}' not found")

	csv_files = sorted(data_path.glob("*.csv"))

	if not csv_files:
		raise FileNotFoundError(f"No CSV files found in '{data_dir}' directory")

	return csv_files[0]


def load_discord_csv(file_path: Path) -> pd.DataFrame:
	"""Load Discord CSV file"""
	logger.info(f"Loading {file_path.name}")

	df = pd.read_csv(file_path)
	df['Date'] = pd.to_datetime(df['Date'], utc=True)

	logger.info(f"Loaded {len(df)} messages")
	return df


def setup_logging(log_level: int, log_format: str):
	"""Setup logging configuration"""
	logging.basicConfig(level=log_level, format=log_format)
	logger = logging.getLogger(__name__)
	return logger
