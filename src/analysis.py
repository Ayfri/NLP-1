#!/usr/bin/env python3
"""
Analysis module for conversation and statistical analysis
"""

import pandas as pd
from pathlib import Path
import logging
from collections import Counter
import matplotlib.pyplot as plt
import os

from .config import OUTPUT_DIR, FREQ_ANALYSIS_TOP_N, STATS_TOP_LEMMAS, DEFAULT_GAP_MINUTES

logger = logging.getLogger(__name__)


def analyze_conversations(df: pd.DataFrame, gap_minutes: int = DEFAULT_GAP_MINUTES,
                         output_path: str = "output", verbose: bool = False) -> dict:
    """
    Analyze conversation patterns and generate statistics.

    Args:
        df: DataFrame with conversation data
        gap_minutes: Minutes gap to define conversation boundaries
        output_path: Directory to save analysis results
        verbose: Whether to print detailed information

    Returns:
        Dictionary with analysis results
    """
    if verbose:
        print(f"Starting conversation analysis with {gap_minutes}-minute gaps...")

    # Ensure datetime column
    df['datetime'] = pd.to_datetime(df['Date'])
    df = df.sort_values('datetime')

    # Identify conversation sessions
    conversations = identify_conversations(df, gap_minutes)

    # Calculate statistics
    stats = calculate_conversation_stats(conversations, df)

    # Generate visualizations
    create_visualizations(conversations, df, output_path, verbose)

    # Save detailed analysis
    save_conversation_analysis(conversations, stats, output_path)

    if verbose:
        print_analysis_summary(stats)

    return stats


def identify_conversations(df: pd.DataFrame, gap_minutes: int) -> list[dict]:
    """Identify conversation sessions based on time gaps."""
    conversations = []
    current_conv = None

    for _, message in df.iterrows():
        if current_conv is None:
            # Start new conversation
            current_conv = {
                'start_time': message['datetime'],
                'end_time': message['datetime'],
                'messages': [message],
                'participants': {message['Author']}
            }
        else:
            # Check if message continues current conversation
            time_gap = (message['datetime'] - current_conv['end_time']).total_seconds() / 60

            if time_gap <= gap_minutes:
                # Continue current conversation
                current_conv['end_time'] = message['datetime']
                current_conv['messages'].append(message)
                current_conv['participants'].add(message['Author'])
            else:
                # End current conversation and start new one
                conversations.append(current_conv)
                current_conv = {
                    'start_time': message['datetime'],
                    'end_time': message['datetime'],
                    'messages': [message],
                    'participants': {message['Author']}
                }

    # Add last conversation
    if current_conv:
        conversations.append(current_conv)

    return conversations


def calculate_conversation_stats(conversations: list[dict], df: pd.DataFrame) -> dict:
    """Calculate detailed conversation statistics."""
    if not conversations:
        return {}

    # Basic conversation metrics
    total_conversations = len(conversations)
    total_messages = len(df)

    # Duration statistics
    durations = []
    message_counts = []
    participant_counts = []

    for conv in conversations:
        duration = (conv['end_time'] - conv['start_time']).total_seconds() / 60
        durations.append(duration)
        message_counts.append(len(conv['messages']))
        participant_counts.append(len(conv['participants']))

    # Participant activity
    participant_stats = calculate_participant_activity(df)

    # Temporal patterns
    temporal_stats = calculate_temporal_patterns(df)

    return {
        'total_conversations': total_conversations,
        'total_messages': total_messages,
        'avg_messages_per_conversation': sum(message_counts) / total_conversations,
        'avg_conversation_duration_minutes': sum(durations) / total_conversations,
        'avg_participants_per_conversation': sum(participant_counts) / total_conversations,
        'conversation_durations': durations,
        'conversation_message_counts': message_counts,
        'participant_stats': participant_stats,
        'temporal_patterns': temporal_stats
    }


def calculate_participant_activity(df: pd.DataFrame) -> dict:
    """Calculate participant activity statistics."""
    participant_counts = df['Author'].value_counts()

    return {
        'total_participants': len(participant_counts),
        'most_active_participant': participant_counts.index[0],
        'message_distribution': participant_counts.to_dict(),
        'participation_balance': participant_counts.std() / participant_counts.mean()
    }


def calculate_temporal_patterns(df: pd.DataFrame) -> dict:
    """Analyze temporal conversation patterns."""
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.day_name()
    df['month'] = df['datetime'].dt.month

    hourly_activity = df['hour'].value_counts().sort_index()
    daily_activity = df['day_of_week'].value_counts()
    monthly_activity = df['month'].value_counts().sort_index()

    return {
        'hourly_distribution': hourly_activity.to_dict(),
        'daily_distribution': daily_activity.to_dict(),
        'monthly_distribution': monthly_activity.to_dict(),
        'peak_hour': hourly_activity.idxmax(),
        'peak_day': daily_activity.idxmax()
    }


def create_visualizations(conversations: list[dict], df: pd.DataFrame,
                         output_path: str, verbose: bool = False) -> None:
    """Create and save visualization charts."""
    if verbose:
        print("Generating visualizations...")

    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Conversation Analysis Dashboard', fontsize=16)

    # 1. Conversation duration distribution
    durations = [(conv['end_time'] - conv['start_time']).total_seconds() / 60
                for conv in conversations]
    axes[0, 0].hist(durations, bins=30, alpha=0.7, color='skyblue')
    axes[0, 0].set_title('Conversation Duration Distribution')
    axes[0, 0].set_xlabel('Duration (minutes)')
    axes[0, 0].set_ylabel('Frequency')

    # 2. Messages per conversation
    message_counts = [len(conv['messages']) for conv in conversations]
    axes[0, 1].hist(message_counts, bins=30, alpha=0.7, color='lightgreen')
    axes[0, 1].set_title('Messages per Conversation')
    axes[0, 1].set_xlabel('Number of Messages')
    axes[0, 1].set_ylabel('Frequency')

    # 3. Hourly activity pattern
    df['hour'] = df['datetime'].dt.hour
    hourly_counts = df['hour'].value_counts().sort_index()
    axes[1, 0].plot(hourly_counts.index, hourly_counts.values, marker='o', color='orange')
    axes[1, 0].set_title('Activity by Hour of Day')
    axes[1, 0].set_xlabel('Hour')
    axes[1, 0].set_ylabel('Number of Messages')
    axes[1, 0].set_xticks(range(0, 24, 2))

    # 4. Participant activity
    participant_counts = df['Author'].value_counts().head(10)
    axes[1, 1].bar(range(len(participant_counts)), participant_counts.values, color='coral')
    axes[1, 1].set_title('Top 10 Most Active Participants')
    axes[1, 1].set_xlabel('Participants')
    axes[1, 1].set_ylabel('Number of Messages')
    axes[1, 1].set_xticks(range(len(participant_counts)))
    axes[1, 1].set_xticklabels([name[:10] + '...' if len(name) > 10 else name
                               for name in participant_counts.index], rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'conversation_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()


def save_conversation_analysis(conversations: list[dict], stats: dict, output_path: str) -> None:
    """Save detailed conversation analysis to files."""
    # Save conversation details
    conv_data = []
    for i, conv in enumerate(conversations):
        conv_data.append({
            'conversation_id': i + 1,
            'start_time': conv['start_time'],
            'end_time': conv['end_time'],
            'duration_minutes': (conv['end_time'] - conv['start_time']).total_seconds() / 60,
            'message_count': len(conv['messages']),
            'participant_count': len(conv['participants']),
            'participants': ', '.join(conv['participants'])
        })

    conv_df = pd.DataFrame(conv_data)
    conv_df.to_csv(os.path.join(output_path, 'conversation_details.csv'), index=False)

    # Save summary statistics
    with open(os.path.join(output_path, 'conversation_summary.txt'), 'w', encoding='utf-8') as f:
        f.write("=== CONVERSATION ANALYSIS SUMMARY ===\n\n")
        f.write(f"Total conversations: {stats['total_conversations']}\n")
        f.write(f"Total messages: {stats['total_messages']}\n")
        f.write(f"Average messages per conversation: {stats['avg_messages_per_conversation']:.2f}\n")
        f.write(f"Average conversation duration: {stats['avg_conversation_duration_minutes']:.2f} minutes\n")
        f.write(f"Average participants per conversation: {stats['avg_participants_per_conversation']:.2f}\n\n")

        f.write("=== PARTICIPANT STATISTICS ===\n")
        f.write(f"Total participants: {stats['participant_stats']['total_participants']}\n")
        f.write(f"Most active participant: {stats['participant_stats']['most_active_participant']}\n\n")

        f.write("=== TEMPORAL PATTERNS ===\n")
        f.write(f"Peak activity hour: {stats['temporal_patterns']['peak_hour']}:00\n")
        f.write(f"Most active day: {stats['temporal_patterns']['peak_day']}\n")


def print_analysis_summary(stats: dict) -> None:
    """Print a summary of the analysis results."""
    print("\n=== CONVERSATION ANALYSIS RESULTS ===")
    print(f"Total conversations: {stats['total_conversations']}")
    print(f"Total messages: {stats['total_messages']}")
    print(f"Average messages per conversation: {stats['avg_messages_per_conversation']:.2f}")
    print(f"Average conversation duration: {stats['avg_conversation_duration_minutes']:.2f} minutes")
    print(f"Peak activity hour: {stats['temporal_patterns']['peak_hour']}:00")
    print(f"Most active day: {stats['temporal_patterns']['peak_day']}")
    print(f"Most active participant: {stats['participant_stats']['most_active_participant']}")
    print("Analysis complete. Check output directory for detailed results.\n")


def create_conversation_summary(processed_df: pd.DataFrame, output_dir: str = OUTPUT_DIR):
	"""Create a simplified conversation analysis file"""
	output_path = Path(output_dir)

	# Basic conversation metrics
	total_messages = len(processed_df)
	non_empty_df = processed_df[processed_df['processed_text'] != '']

	# Get time range
	date_range = processed_df['Date'].agg(['min', 'max'])
	duration_hours = (date_range['max'] - date_range['min']).total_seconds() / 3600

	# Participant analysis
	participants = processed_df['Author'].nunique()
	top_participants = processed_df['Author'].value_counts().head(5)

	# Sentiment summary
	sentiment_counts = processed_df['sentiment_label'].value_counts()

	# Top lemmas
	all_lemmas = []
	for text in non_empty_df['processed_text']:
		if text:
			all_lemmas.extend(text.split())

	top_lemmas = Counter(all_lemmas).most_common(10) if all_lemmas else []

	# Create summary data
	summary_data = {
		'total_messages': total_messages,
		'messages_with_content': len(non_empty_df),
		'empty_messages': total_messages - len(non_empty_df),
		'participants': participants,
		'duration_hours': round(duration_hours, 2),
		'avg_tokens_per_message': round(non_empty_df['token_count'].mean(), 1) if len(non_empty_df) > 0 else 0,
		'avg_lemmas_per_message': round(non_empty_df['lemma_count'].mean(), 1) if len(non_empty_df) > 0 else 0,
		'sentiment_positive': sentiment_counts.get('positive', 0),
		'sentiment_neutral': sentiment_counts.get('neutral', 0),
		'sentiment_negative': sentiment_counts.get('negative', 0),
		'top_5_participants': '; '.join([f"{author}: {count}" for author, count in top_participants.items()]),
		'top_10_words': '; '.join([f"{word}: {count}" for word, count in top_lemmas])
	}

	# Save as CSV
	summary_df = pd.DataFrame([summary_data])
	summary_file = output_path / 'conversation_summary.csv'
	summary_df.to_csv(summary_file, index=False)
	logger.info(f"✅ Conversation summary saved to {summary_file}")


def save_results(processed_df: pd.DataFrame, output_dir: str = OUTPUT_DIR):
	"""Save processing results"""
	output_path = Path(output_dir)
	output_path.mkdir(exist_ok=True)

	# Save processed messages
	output_file = output_path / 'messages_processed.csv'
	processed_df.to_csv(output_file, index=False)
	logger.info(f"✅ Results saved to {output_file}")

	# Statistics
	total_messages = len(processed_df)
	empty_messages = len(processed_df[processed_df['processed_text'] == ''])
	non_empty_df = processed_df[processed_df['processed_text'] != '']

	if len(non_empty_df) > 0:
		avg_tokens = non_empty_df['token_count'].mean()
		avg_lemmas = non_empty_df['lemma_count'].mean()
	else:
		avg_tokens = avg_lemmas = 0

	logger.info("📊 Processing Statistics:")
	logger.info(f"  Total messages: {total_messages:,}")
	logger.info(f"  Empty after processing: {empty_messages:,} ({empty_messages/total_messages*100:.1f}%)")
	logger.info(f"  Messages with content: {len(non_empty_df):,}")
	logger.info(f"  Average tokens per message: {avg_tokens:.1f}")
	logger.info(f"  Average lemmas per message: {avg_lemmas:.1f}")

	# Sentiment distribution
	sentiment_counts = processed_df['sentiment_label'].value_counts()

	# Word frequency analysis
	if len(non_empty_df) > 0:
		all_lemmas = []
		for text in non_empty_df['processed_text']:
			if text:
				all_lemmas.extend(text.split())

		if all_lemmas:
			lemma_counts = Counter(all_lemmas)
			common_lemmas = lemma_counts.most_common(STATS_TOP_LEMMAS)

			logger.info(f"🔤 Top {STATS_TOP_LEMMAS} most common lemmas:")
			for lemma, count in common_lemmas:
				percentage = (count / len(all_lemmas)) * 100
				logger.info(f"  {lemma}: {count} ({percentage:.2f}%)")

			# Save frequency analysis
			freq_df = pd.DataFrame(
				lemma_counts.most_common(FREQ_ANALYSIS_TOP_N),
				columns=['lemma', 'count']
			)
			freq_df['percentage'] = (freq_df['count'] / len(all_lemmas)) * 100

			freq_file = output_path / 'lemma_frequency.csv'
			freq_df.to_csv(freq_file, index=False)
			logger.info(f"✅ Lemma frequency analysis saved to {freq_file}")

	# Create conversation summary
	create_conversation_summary(processed_df, output_dir)
