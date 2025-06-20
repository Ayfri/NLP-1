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

### Embeddings Comparison
- **BoW/TF-IDF**: Traditional sparse representations with n-grams
- **Word2Vec + TF-IDF**: Dense word embeddings weighted by TF-IDF scores
- **FastText**: Character n-gram enhanced word embeddings
- **BERT+**: Transformer-based contextualized embeddings (CamemBERT/multilingual BERT)
- **Performance evaluation**: Clustering quality (silhouette score), processing time
- **Visualizations**: t-SNE plots, performance comparisons

## Requirements

- Python 3.8+
- spaCy French model: `python -m spacy download fr_core_news_sm`
- spaCy English model: `python -m spacy download en_core_web_sm`
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
   python -m spacy download en_core_web_sm
   ```
4. Download NLTK data:
   ```bash
   python -m nltk.downloader punkt_tab
   ```

## Discord Export

To export Discord messages in CSV format, use [DiscordChatExporter](https://github.com/Tyrrrz/DiscordChatExporter):

1. Download DiscordChatExporter
2. Export conversations in **CSV format**
3. Place CSV files in the `data/` folder

## Usage

### Basic NLP Processing
1. Place your Discord CSV files in the `data/` directory
2. Run the main processor:
   ```bash
   python main.py
   ```
3. Results will be saved in the `output/` directory

### Embeddings Comparison
After running the main processor, test different embedding methods:
```bash
python test_embeddings.py
```

This will:
- Load processed messages from `output/messages_processed.csv`
- Test 4 different embedding approaches
- Generate comparison visualizations
- Save detailed results and reports

## Input Format

CSV files should contain the following columns:
- `AuthorID`: Discord user ID
- `Author`: Username
- `Date`: Message timestamp
- `Content`: Message text
- `Attachments`: File attachments (optional)
- `Reactions`: Message reactions (optional)

## Output Files

### Basic NLP Processing
- `messages_processed.csv`: Processed messages with NLP results
- `conversations_analysis.csv`: Detailed conversation analysis
- `conversation_summary.csv`: Global conversation summary
- `lemma_frequency.csv`: Lemma frequency analysis (top 100)
- Detailed logs in console

### Embeddings Testing
- `embeddings_comparison_results.csv`: Summary comparison table
- `embeddings_comparison_report.txt`: Detailed analysis report
- `embeddings_performance_comparison.png`: Processing time & clustering quality
- `embeddings_tsne_comparison.png`: t-SNE visualizations for all methods
- `clustering_quality_comparison.png`: Silhouette scores comparison

## Project Structure

```
NLP-1/
├── data/                    # Input CSV files
├── output/                  # Processing results
├── main.py                  # Main NLP processing script
├── test_embeddings.py       # Embeddings comparison script
├── requirements.txt         # Python dependencies
├── tet.ipynb               # Jupyter notebook (optional)
└── README.md               # This file
```

## Configuration

### Main Processing (`main.py`)
- `BATCH_SIZE`: Processing batch size (default: 2000)
- `MIN_TOKEN_LENGTH` / `MAX_TOKEN_LENGTH`: Token filtering
- `FREQ_ANALYSIS_TOP_N`: Number of lemmas in frequency analysis
- `gap_minutes`: Conversation detection threshold (default: 30 min)

### Embeddings Testing (`test_embeddings.py`)
- `TF_IDF_MAX_FEATURES`: Maximum features for TF-IDF (default: 5000)
- `WORD2VEC_DIM` / `FASTTEXT_DIM`: Embedding dimensions (default: 100)
- `N_CLUSTERS`: Number of clusters for evaluation (default: 5)
- `BERT_MAX_LENGTH`: Maximum sequence length for BERT (default: 512)

## Processing Pipeline

### Main NLP Pipeline
```
Raw Message → Cleaning → Tokenization → Stopwords Removal → Lemmatization → Sentiment Analysis
```

### Embeddings Testing Pipeline
```
Processed Messages → BoW/TF-IDF → Word2Vec+TF-IDF → FastText → BERT+ → Evaluation & Visualization
```

Each step is optimized for processing large datasets with French/English multilingual support.

## Embeddings Methods Details

### 1. BoW/TF-IDF (Traditional)
- **BoW**: Simple word counts with n-grams (1-2)
- **TF-IDF**: Term frequency-inverse document frequency weighting
- **Features**: Sparse high-dimensional vectors (up to 5000 features)
- **Pros**: Fast, interpretable, baseline performance
- **Cons**: No semantic understanding, high sparsity

### 2. Word2Vec + TF-IDF (Hybrid)
- **Word2Vec**: CBOW model trained on dataset (100 dimensions)
- **TF-IDF weighting**: Document vectors as TF-IDF weighted word averages
- **Features**: Dense low-dimensional vectors
- **Pros**: Semantic relationships, contextual weighting
- **Cons**: Averaged representations lose word order

### 3. FastText (Enhanced Word2Vec)
- **Character n-grams**: Subword information (3-6 character n-grams)
- **Out-of-vocabulary**: Can handle unseen words
- **Features**: Dense vectors with morphological awareness
- **Pros**: Robust to misspellings, works with rare words
- **Cons**: Still averaged document representations

### 4. BERT+ (Contextualized)
- **CamemBERT**: French-specific BERT model (or multilingual fallback)
- **Contextualized**: Word meanings depend on context
- **Features**: High-dimensional dense vectors (768 dimensions)
- **Pros**: State-of-the-art semantic understanding
- **Cons**: Computationally expensive, requires GPU for speed
