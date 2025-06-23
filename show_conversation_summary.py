#!/usr/bin/env python3
"""
Discord Conversations Display with AI Summaries
"""

import pandas as pd
import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import warnings
import re
warnings.filterwarnings('ignore')

# Configuration
CONVERSATIONS_FILE = "output/conversations_analysis.csv"
MESSAGES_FILE = "output/messages_processed.csv"
SUMMARIZATION_MODEL_PATH = "./models/summarization"

MIN_MESSAGES_PER_CONVERSATION = 12
MIN_DURATION_MINUTES = 8
MAX_CONVERSATIONS_TO_SHOW = 5
FALLBACK_MESSAGES_COUNT = 20
MAX_MESSAGES_PER_CONVERSATION = 50

MIN_TEXT_LENGTH_FOR_SUMMARY = 50
MAX_INPUT_LENGTH = 1000
MAX_SUMMARY_LENGTH = 150
MIN_SUMMARY_LENGTH = 20
NUM_BEAMS = 4
NO_REPEAT_NGRAM_SIZE = 2

def load_conversations():
	"""Load conversations with filtering"""
	if not os.path.exists(CONVERSATIONS_FILE):
		print(f"Error: {CONVERSATIONS_FILE} not found")
		print("Run first: python main.py")
		return None

	df = pd.read_csv(CONVERSATIONS_FILE)
	filtered_df = df[
		(df['key_topics'].notna()) &
		(df['message_count'] >= MIN_MESSAGES_PER_CONVERSATION) &
		(df['duration_minutes'] >= MIN_DURATION_MINUTES)
	].sort_values('message_count', ascending=False)

	if len(filtered_df) == 0:
		print("Warning: No long conversations found, using all available")
		return df[df['key_topics'].notna()].head(MAX_CONVERSATIONS_TO_SHOW)

	return filtered_df.head(MAX_CONVERSATIONS_TO_SHOW)

def load_messages():
	"""Load all messages"""
	if not os.path.exists(MESSAGES_FILE):
		print(f"Error: {MESSAGES_FILE} not found")
		return None

	df = pd.read_csv(MESSAGES_FILE)
	print(f"Columns: {', '.join(df.columns.tolist())}")
	return df

def get_conversation_messages(messages_df, start_time, end_time):
	"""Get messages from specific conversation"""
	if 'Date' in messages_df.columns:
		try:
			if not pd.api.types.is_datetime64_any_dtype(messages_df['Date']):
				messages_df['Date'] = pd.to_datetime(messages_df['Date'], format='mixed', utc=True)

			start_time = pd.to_datetime(start_time, utc=True)
			end_time = pd.to_datetime(end_time, utc=True)

			if messages_df['Date'].dt.tz is None:
				messages_df['Date'] = messages_df['Date'].dt.tz_localize('UTC')
			elif str(messages_df['Date'].dt.tz) != 'UTC':
				messages_df['Date'] = messages_df['Date'].dt.tz_convert('UTC')

			conv_messages = messages_df[
				(messages_df['Date'] >= start_time) &
				(messages_df['Date'] <= end_time)
			].copy()

			print(f"Found {len(conv_messages)} messages for this period")

		except Exception as e:
			print(f"Date parsing error: {e}")
			conv_messages = messages_df.head(FALLBACK_MESSAGES_COUNT).copy()
	else:
		conv_messages = messages_df.head(FALLBACK_MESSAGES_COUNT).copy()

	if len(conv_messages) == 0:
		print("No messages found for this period, using sample")
		conv_messages = messages_df.head(FALLBACK_MESSAGES_COUNT).copy()

	return conv_messages[['Author', 'original_content']].dropna()

def generate_ai_summary(conversation_text):
	"""Generate AI summary"""
	if not os.path.exists(SUMMARIZATION_MODEL_PATH):
		print("Warning: Summarization model not found. Use: python test_models.py")
		return "Summary not available (model not trained)"

	try:
		tokenizer = AutoTokenizer.from_pretrained(SUMMARIZATION_MODEL_PATH)
		model = AutoModelForSeq2SeqLM.from_pretrained(SUMMARIZATION_MODEL_PATH)
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		model.to(device)

		if len(conversation_text) < MIN_TEXT_LENGTH_FOR_SUMMARY:
			return "Conversation too short to generate summary"

		# Better text preprocessing for French conversations
		conversation_text = conversation_text.strip()

		# Clean up the text - remove excessive spaces and truncation markers
		conversation_text = ' '.join(conversation_text.split())
		conversation_text = conversation_text.replace('...', ' ')

		# Truncate more intelligently - try to end at sentence boundaries
		if len(conversation_text) > 2000:
			# Find last sentence ending before 2000 chars
			truncate_pos = 2000
			for i in range(min(2000, len(conversation_text)) - 1, max(1500, 0), -1):
				if conversation_text[i] in '.!?':
					truncate_pos = i + 1
					break
			conversation_text = conversation_text[:truncate_pos]

		# Add French context prefix to help model understand language
		prefixed_text = f"Résumé de conversation en français: {conversation_text}"

		inputs = tokenizer(
			prefixed_text,
			max_length=MAX_INPUT_LENGTH,
			padding='max_length',
			truncation=True,
			return_tensors='pt'
		).to(device)

		with torch.no_grad():
			summary_ids = model.generate(
				inputs['input_ids'],
				attention_mask=inputs['attention_mask'],
				max_length=80,  # Reduced for better quality
				min_length=15,  # Reduced minimum
				num_beams=NUM_BEAMS,
				early_stopping=True,
				no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
				temperature=0.7,  # Add some creativity
				do_sample=False,  # Keep deterministic with beam search
				length_penalty=1.2  # Encourage proper length
			)

		summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

		# Clean up the generated summary
		summary = summary.replace("Résumé de conversation en français:", "").strip()

		# If summary is still mixed language or poor quality, provide fallback
		if len(summary) < 10 or "pilot-chat" in summary.lower():
			return "Résumé automatique non disponible - conversation complexe"

		return summary

	except Exception as e:
		return f"Generation error: {str(e)[:50]}..."

def display_conversation(conv_data, messages_df, conv_num):
	"""Display conversation with summary"""
	print(f"\n--- CONVERSATION {conv_num} ---")
	print(f"Duration: {conv_data['duration_minutes']:.1f} minutes")
	print(f"Messages: {conv_data['message_count']}")
	print(f"Participants: {conv_data['top_participants']}")
	print(f"Emotion: {conv_data['dominant_emotion']}")

	if 'start_time' in conv_data and 'end_time' in conv_data:
		conv_messages = get_conversation_messages(
			messages_df,
			conv_data['start_time'],
			conv_data['end_time']
		)
	else:
		conv_messages = messages_df.sample(min(8, len(messages_df)))

	# Limit number of messages to process
	conv_messages_limited = conv_messages.head(MAX_MESSAGES_PER_CONVERSATION)

	print(f"\nMessages (showing {len(conv_messages_limited)}/{len(conv_messages)}):")
	conversation_text = ""
	for i, (_, msg) in enumerate(conv_messages_limited.iterrows(), 1):
		try:
			author = str(msg['Author'])[:15]
			content = str(msg['original_content'])

			# Better content handling - don't truncate individual messages
			if len(content) > 200:
				# Find a good truncation point for very long messages
				truncate_pos = 200
				for j in range(min(200, len(content)) - 1, max(150, 0), -1):
					if content[j] in '.!? ':
						truncate_pos = j + 1
						break
				content_display = content[:truncate_pos] + "..."
			else:
				content_display = content

			print(f"{i:2d}. {author}: {content_display}")

			# For AI input, use full content but clean it
			clean_content = content.strip()
			# Remove URLs and mentions for cleaner summaries
			clean_content = re.sub(r'http[s]?://\S+', '[lien]', clean_content)
			clean_content = re.sub(r'<@\d+>', '[mention]', clean_content)
			clean_content = re.sub(r'@\w+', '[mention]', clean_content)

			# Add to conversation text with better structure
			conversation_text += f"{clean_content}. "

		except Exception as e:
			print(f"Message display error {i}: {e}")
			continue

	print(f"\nKey topics: {conv_data['key_topics']}")

	ai_summary = generate_ai_summary(conversation_text)
	print(f"AI summary: {ai_summary}")

def main():
	"""Main function"""
	print("Displaying conversations with summaries\n")

	conversations_df = load_conversations()
	if conversations_df is None:
		return

	messages_df = load_messages()
	if messages_df is None:
		return

	print(f"{len(conversations_df)} conversations found")
	print(f"{len(messages_df)} messages loaded")

	if len(conversations_df) > 0:
		avg_messages = conversations_df['message_count'].mean()
		avg_duration = conversations_df['duration_minutes'].mean()
		print(f"Average: {avg_messages:.1f} messages, {avg_duration:.1f} min duration")

	for i, (_, conv) in enumerate(conversations_df.iterrows(), 1):
		display_conversation(conv, messages_df, i)
		if i < len(conversations_df):
			input("\nPress Enter for next conversation...")

	print("\nAll conversations displayed!")
	print("To train/improve the summarization model: python test_models.py")

if __name__ == "__main__":
	main()
