# Discord Messages NLP Pipeline

A comprehensive NLP pipeline for analyzing Discord messages with advanced text processing and conversational analysis features.

## Features

### Text Processing
- **Advanced text cleaning**: Removes Discord mentions, emojis, URLs, special characters
- **French contractions support**: Automatic expansion (c'est → ce est, qu'il → que il, etc.)
- **Smart tokenization**: Token splitting with length filtering
- **Stopwords removal**: French and English support with contextual preservation
- **Lemmatization**: Reduction to canonical forms using spaCy
- **Batch processing**: Efficient processing of large message volumes

### Sentiment Analysis
- **Sentiment scoring**: VADER analysis (range -1 to +1)
- **Emotion classification**: Positive, negative, neutral
- **Multilingual support**: French and English

### Conversation Analysis
- **Automatic conversation detection**: Based on time intervals
- **Per-conversation metrics**:
  - Duration and message count
  - Active participants
  - Dominant emotion
  - Key topics (most frequent words)
- **Participant analysis**: Activity ranking

### Statistical Analysis
- **Lemma frequency**: Top 100 words with percentages
- **Global statistics**: Average tokens, sentiment distribution
- **Conversation summary**: Activity overview

## Requirements

- Python 3.8+
- spaCy French model: `python -m spacy download fr_core_news_sm`
- Dependencies listed in `requirements.txt`

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download spaCy model:
   ```bash
   python -m spacy download fr_core_news_sm
   ```

## Discord Export

To export Discord messages in CSV format, use [DiscordChatExporter](https://github.com/Tyrrrz/DiscordChatExporter):

1. Download DiscordChatExporter
2. Export conversations in **CSV format**
3. Place CSV files in the `data/` folder

## Usage

1. Place your Discord CSV files in the `data/` directory
2. Run the processor:
   ```bash
   python main.py
   ```
3. Results will be saved in the `output/` directory

## Input Format

CSV files should contain the following columns:
- `AuthorID`: Discord user ID
- `Author`: Username
- `Date`: Message timestamp
- `Content`: Message text
- `Attachments`: File attachments (optional)
- `Reactions`: Message reactions (optional)

## Output Files

- `messages_processed.csv`: Processed messages with NLP results
- `conversations_analysis.csv`: Detailed conversation analysis
- `conversation_summary.csv`: Global conversation summary
- `lemma_frequency.csv`: Lemma frequency analysis (top 100)
- Detailed logs in console

## Project Structure

```
nlp-2/
├── data/              # Input CSV files
├── output/            # Processing results
├── main.py            # Main processing script
├── requirements.txt   # Python dependencies
└── README.md          # This file
```

## Configuration

Parameters can be adjusted in `main.py`:
- `BATCH_SIZE`: Processing batch size (default: 2000)
- `MIN_TOKEN_LENGTH` / `MAX_TOKEN_LENGTH`: Token filtering
- `FREQ_ANALYSIS_TOP_N`: Number of lemmas in frequency analysis
- `gap_minutes`: Conversation detection threshold (default: 30 min)

## Processing Pipeline

```
Raw Message → Cleaning → Tokenization → Stopwords Removal → Lemmatization → Sentiment Analysis
```

Each step is optimized for processing large datasets with French/English multilingual support.
