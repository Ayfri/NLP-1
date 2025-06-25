#!/usr/bin/env python3
"""
NLP Conversation Analysis Tool
Main entry point for processing and analyzing conversation data with FastText embeddings.
"""

import argparse
import os
import sys
from pathlib import Path

from src.config import DEFAULT_DATA_PATH, DEFAULT_OUTPUT_PATH, DEFAULT_GAP_MINUTES
from src.utils import load_discord_csv, find_first_csv
from src.nlp_processor import DiscordNLPProcessor
from src.embeddings import FastTextEmbedder
from src.analysis import analyze_conversations
from src.data_loaders import process_messages_with_dataloader


def parse_arguments():
	"""Parse command line arguments."""
	parser = argparse.ArgumentParser(
		description="NLP Conversation Analysis Tool with FastText Embeddings",
		formatter_class=argparse.ArgumentDefaultsHelpFormatter
	)

	parser.add_argument(
		"--data", "-d",
		type=str,
		default=DEFAULT_DATA_PATH,
		help="Path to conversation data directory"
	)

	parser.add_argument(
		"--output", "-o",
		type=str,
		default=DEFAULT_OUTPUT_PATH,
		help="Output directory for results"
	)

	parser.add_argument(
		"--no-embeddings",
		action="store_true",
		help="Skip FastText embeddings generation"
	)

	parser.add_argument(
		"--no-conversations",
		action="store_true",
		help="Skip conversation analysis"
	)

	parser.add_argument(
		"--verbose", "-v",
		action="store_true",
		help="Enable verbose output"
	)

	parser.add_argument(
		"--gap-minutes",
		type=int,
		default=DEFAULT_GAP_MINUTES,
		help="Minutes gap to define conversation boundaries"
	)

	return parser.parse_args()


def setup_output_directory(output_path: str, verbose: bool = False) -> None:
	"""Create output directory if it doesn't exist."""
	Path(output_path).mkdir(parents=True, exist_ok=True)
	if verbose:
		print(f"Output directory: {output_path}")


def main():
	"""Main execution function."""
	args = parse_arguments()

	if args.verbose:
		print("=== NLP Conversation Analysis Tool ===")
		print(f"Data path: {args.data}")
		print(f"Output path: {args.output}")

	# Setup output directory
	setup_output_directory(args.output, args.verbose)

	# Check if data directory exists
	if not os.path.exists(args.data):
		print(f"Error: Data directory '{args.data}' not found.")
		sys.exit(1)

	try:
		# Load conversation data
		if args.verbose:
			print("Loading conversation files...")

		if args.data.endswith('.csv'):
			csv_file = args.data
		else:
			csv_file = find_first_csv(args.data)

		df = load_discord_csv(csv_file)

		if df.empty:
			print("No conversation data found.")
			sys.exit(1)

		if args.verbose:
			print(f"Loaded {len(df)} messages")

		# Process messages with NLP using DataLoader
		if args.verbose:
			print("Processing messages using DataLoader...")

		processor = DiscordNLPProcessor()
		processed_df = process_messages_with_dataloader(df, processor)

		# Save processed data
		processed_df.to_csv(os.path.join(args.output, "processed_messages.csv"), index=False)

		# Generate FastText embeddings
		if not args.no_embeddings:
			if args.verbose:
				print("Generating FastText embeddings...")

			embedder = FastTextEmbedder()

			# Get valid texts for training
			valid_texts = processed_df[processed_df['processed_text'].notna() &
									 (processed_df['processed_text'].str.len() > 0)]['processed_text'].tolist()

			if valid_texts:
				# Train FastText model
				embedder.train(valid_texts)

				# Generate embeddings using DataLoader
				embeddings = embedder.generate_embeddings(valid_texts)

				# Perform clustering
				cluster_labels, cluster_stats = embedder.cluster_texts(5)

				# Save results
				embedder.save_model(os.path.join(args.output, "fasttext_model.bin"))
				embedder.save_embeddings(os.path.join(args.output, "embeddings.pkl"))

				if args.verbose:
					print(f"Generated {len(embeddings)} embeddings with {len(set(cluster_labels))} clusters")
			else:
				print("No valid texts found for embedding generation.")

		# Analyze conversations
		if not args.no_conversations:
			if args.verbose:
				print("Analyzing conversations...")

			stats = analyze_conversations(
				df=processed_df,
				gap_minutes=args.gap_minutes,
				output_path=args.output,
				verbose=args.verbose
			)

			if args.verbose:
				print("Conversation analysis complete.")

		if args.verbose:
			print("Processing complete. Check output directory for results.")

	except Exception as e:
		print(f"Error during processing: {e}")
		if args.verbose:
			import traceback
			traceback.print_exc()
		sys.exit(1)


if __name__ == "__main__":
	main()
