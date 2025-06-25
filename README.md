# NLP Pipeline for Discord Messages

A complete NLP pipeline for analyzing Discord conversations, designed for educational purposes and NLP coursework.

## Features

- **Text processing**: Cleaning, tokenization, lemmatization with spaCy
- **Sentiment analysis**: Positive/negative/neutral classification using VADER
- **Conversation analysis**: Automatic detection and metrics per conversation
- **FastText embeddings**: Optimized vector representations for social media text
- **Multilingual support**: French and English

## Installation

```bash
pip install -r requirements.txt
python -m spacy download fr_core_news_sm
```

## Usage

```bash
# Basic processing
python main.py

# With options
python main.py --data data/messages.csv --output results --verbose

# Skip embeddings
python main.py --no-embeddings --no-conversations

# Compare embeddings (optional)
python test_embeddings.py
```

### Available options

- `--data, -d`: Directory or CSV file (default: data/)
- `--output, -o`: Output directory (default: output/)
- `--no-embeddings`: Skip FastText generation
- `--no-conversations`: Skip conversation analysis
- `--verbose, -v`: Verbose mode
- `--gap-minutes`: Conversation detection threshold in minutes (default: 30)

## Output files

- `messages_processed.csv`: Processed messages with NLP results
- `conversations_analysis.csv`: Detailed conversation analysis
- `conversation_summary.csv`: Global summary
- `lemma_frequency.csv`: Lemma frequency analysis (top 100)
- `fasttext_model.bin`: Trained FastText model
- `embeddings.pkl`: Document vectors
- `text_clusters.csv`: Message clustering

## Notes

- Export Discord data using [DiscordChatExporter](https://github.com/Tyrrrz/DiscordChatExporter) in CSV format
- Place CSV files in `data/` directory
- FastText is used by default for embeddings (recommended from [`EMBEDDINGS_RAPPORT.md`](./EMBEDDINGS_RAPPORT.md))
- Detailed configuration available in `src/config.py`
