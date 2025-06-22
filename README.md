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

### Machine Learning Models
- **Sentiment Classification**: DistilBERT-based multilingual sentiment analysis
- **Conversation Summarization**: DistilBART-powered automatic summary generation
- **GPU Acceleration**: Automatic hardware detection with mixed precision training
- **Fast Training**: Optimized configurations for rapid model development

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
   python -c "import nltk; nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('vader_lexicon')"
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

### Machine Learning Models Training
For advanced sentiment classification and conversation summarization:
```bash
python test_models.py
```

This will:
- Train DistilBERT for sentiment classification
- Train DistilBART for conversation summarization
- Automatically detect and use GPU if available
- Generate comprehensive evaluation reports

### 📊 Embeddings Evaluation & Recommendation

I ran an automated benchmark (`test_embeddings.py`) to compare four embedding strategies on clustering quality (silhouette score) and processing speed. The full methodology and numbers are in [`EMBEDDINGS_RAPPORT.md`](./EMBEDDINGS_RAPPORT.md).

**TL;DR**

| Best Quality | Best Speed | Balanced Choice |
|--------------|-----------|-----------------|
| **FastText** *(silhouette 0.54)* | **BoW / TF-IDF** *(< 0.1 s per 1 k msgs)* | **FastText** provides the best trade-off |

I therefore default to **FastText** vectors for downstream conversation and sentiment analyses. Feel free to switch to Word2Vec when sub-second latency is critical, or to BERT variants if GPU resources are available and you need sentence-level semantics.

### 🤖 Machine Learning Models Documentation

For detailed information about the ML models used for sentiment classification and conversation summarization, including model selection rationale, training configurations, and performance expectations, see [`MODELS.md`](./MODELS.md).

**Key highlights:**
- **Fast training**: 2-10 minutes on GPU, optimized for Discord data
- **Multilingual support**: Handles French/English code-switching
- **Automatic optimization**: GPU detection with mixed precision training

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

## Technical Choices & Design Decisions

Key technical decisions and their rationale:

### Core Architecture
- **spaCy + NLTK hybrid**: spaCy provides superior French linguistic models for lemmatization, while NLTK offers mature tokenization and the proven VADER sentiment lexicon for informal text
- **French-first multilingual**: Discord conversations are primarily French, so prioritizing `fr_core_news_sm` ensures better accuracy while maintaining English fallback for mixed-language channels
- **Batch processing**: Processing 2000 messages at once maximizes spaCy's efficiency while preventing memory overflow on large datasets

### French Language Optimization
- **Contractions handling**: French contractions (c'est, qu'il) break standard tokenizers, so custom expansion ensures proper word boundary detection and improves downstream analysis
- **Contextual stopwords**: Standard French stopword lists remove semantically important words like "pas" (negation) and "ça", which are crucial for Discord sentiment analysis
- **Pre-compiled regex**: Discord messages contain repetitive patterns (mentions, emojis), so compiling patterns once prevents performance degradation on large batches

### Processing Strategy
- **Multi-stage pipeline**: Sequential cleaning (Discord → contractions → normalization) ensures each step works on predictably formatted text, reducing edge cases
- **VADER sentiment analysis**: Transformer models are overkill for Discord messages and too slow for real-time analysis; VADER handles informal language and emojis effectively
- **Time-gap conversation detection**: Complex conversation threading is unreliable in Discord exports; 30-minute gaps capture natural conversation breaks without over-engineering

### Design Philosophy
- **Pandas-centric workflow**: CSV processing and analysis is pandas' strength; avoiding unnecessary data format conversions keeps the pipeline simple and debuggable
- **Modular outputs**: Separate files allow users to analyze specific aspects (conversations vs. word frequency) without loading unnecessary data
- **Graceful degradation**: Discord data is messy and models may be missing; fallback mechanisms ensure the pipeline never crashes on real-world data

These choices prioritize speed, French accuracy, and maintainability for Discord conversation analysis.

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
