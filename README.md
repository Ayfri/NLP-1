# Discord Messages NLP Pipeline

A simple NLP pipeline for processing Discord messages with basic text analysis.

## Features

- **Text Cleaning**: Removes Discord mentions, emojis, URLs, and special characters
- **Tokenization**: Splits text into individual tokens using NLTK
- **Stopwords Removal**: Filters out common French and English stopwords
- **Lemmatization**: Reduces words to their base forms using spaCy

## Pipeline Steps

```
Raw Discord Message → Cleaning → Tokenization → Stopwords Removal → Lemmatization → Processed Text
```

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

## Usage

1. Place your Discord CSV files in the `data/` directory
2. Run the processor:
   ```bash
   python main.py
   ```
3. Results will be saved in the `output/` directory

## Input Format

The CSV files should have the following columns:
- `AuthorID`: Discord user ID
- `Author`: Username
- `Date`: Message timestamp
- `Content`: Message text
- `Attachments`: File attachments (optional)
- `Reactions`: Message reactions (optional)

## Output

- `output/messages_processed.csv`: Processed messages with NLP results
- Console logs with processing statistics

## Project Structure

```
nlp-2/
├── data/           # Input CSV files
├── output/         # Processing results
├── main.py         # Main processing script
├── requirements.txt # Python dependencies
└── README.md       # This file
```
