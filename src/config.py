#!/usr/bin/env python3
"""
Configuration and constants for the NLP pipeline
"""

import re
import logging

# ================================
# LOGGING CONFIGURATION
# ================================

LOG_LEVEL = logging.INFO
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'

# ================================
# PROCESSING PARAMETERS
# ================================

MIN_TOKEN_LENGTH = 2
MAX_TOKEN_LENGTH = 50
BATCH_SIZE = 2000

# ================================
# REGEX PATTERNS
# ================================

REGEX_PATTERNS = {
	'mentions': re.compile(r'<@!?\d+>'),
	'mentions_legacy': re.compile(r'@\w+(?:#\d{4})?'),
	'emojis_custom': re.compile(r'<a?:\w+:\d+>'),
	'emojis_standard': re.compile(r':\w+:'),
	'urls': re.compile(r'https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:\w*))?)?'),
	'apostrophes': re.compile(r"([a-zA-Z])'([a-zA-Z])"),
	'special_chars': re.compile(r'[^\w\s.,!?;:\-àáâäçèéêëìíîïñòóôöùúûüÿæœÀÁÂÄÇÈÉÊËÌÍÎÏÑÒÓÔÖÙÚÛÜŸÆŒ]'),
	'multiple_spaces': re.compile(r'\s+')
}

# ================================
# FRENCH CONTRACTIONS
# ================================

FRENCH_CONTRACTIONS = {
	r"c'est": "ce est",
	r"qu'": "que ",
	r"d'": "de ",
	r"n'": "ne ",
	r"s'": "se ",
	r"j'": "je ",
	r"t'": "te ",
	r"m'": "me ",
	r"l'": "le ",
	r"jusqu'": "jusque ",
	r"aujourd'hui": "aujourd hui",
	r"quelqu'": "quelque ",
}

# Compile contraction patterns
CONTRACTION_PATTERNS = {re.compile(pattern, re.IGNORECASE): replacement
						for pattern, replacement in FRENCH_CONTRACTIONS.items()}

# ================================
# VOCABULARY FILTERING
# ================================

# Important short words to preserve
IMPORTANT_SHORT_WORDS = {
	'ça', 'si', 'ou', 'et', 'en', 'un', 'le', 'la', 'de', 'du', 'ce', 'se',
	'me', 'te', 'ne', 'je', 'tu', 'il', 'on', 'au', 'ai', 'as', 'va', 'eu'
}

# Contextually important words to keep
CONTEXTUAL_WORDS_TO_KEEP = {
	'pas', 'non', 'oui', 'bien', 'tout', 'tous', 'faire', 'dit', 'voir',
	'très', 'plus', 'moins', 'beaucoup', 'peu', 'assez', 'trop'
}

# ================================
# OUTPUT CONFIGURATION
# ================================

OUTPUT_DIR = "output"
FREQ_ANALYSIS_TOP_N = 100
STATS_TOP_LEMMAS = 15

# ================================
# EMBEDDINGS CONFIGURATION
# ================================

# FastText parameters
FASTTEXT_CONFIG = {
	'vector_size': 100,
	'window': 5,
	'min_count': 2,
	'workers': 4,
	'epochs': 10,
	'sg': 0,  # CBOW
	'min_n': 3,  # Character n-grams
	'max_n': 6
}

# Minimum words for meaningful embeddings
MIN_WORDS_FOR_EMBEDDINGS = 5

# File paths and directories
DEFAULT_DATA_PATH = "data"
DEFAULT_OUTPUT_PATH = "output"
TEMP_EVAL_PATH = "temp_eval"

# Processing parameters
DEFAULT_GAP_MINUTES = 10
BATCH_SIZE = 1000
MIN_TEXT_LENGTH = 10
MAX_TEXT_LENGTH = 500

# Clustering configuration
N_CLUSTERS = 5

# Regex patterns for text cleaning
PATTERNS = {
	'url': re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'),
	'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
	'phone': re.compile(r'(\+33|0)[1-9](\d{8}|\s\d{2}\s\d{2}\s\d{2}\s\d{2})'),
	'mentions': re.compile(r'@\w+'),
	'hashtags': re.compile(r'#\w+'),
	'multiple_spaces': re.compile(r'\s+'),
	'special_chars': re.compile(r'[^\w\s\-.,!?;:()\[\]{}"\']'),
	'repeated_chars': re.compile(r'(.)\1{2,}')
}

# Message types to exclude
EXCLUDED_MESSAGE_TYPES = {
	'service',
	'system',
	'call',
	'file_share',
	'location_share',
	'contact_share',
	'voice_message',
	'video_message',
	'sticker',
	'gif',
	'photo',
	'video',
	'audio',
	'document'
}

# Stop words (basic French set)
FRENCH_STOP_WORDS = {
	'le', 'de', 'et', 'à', 'un', 'il', 'être', 'et', 'en', 'avoir', 'que', 'pour',
	'dans', 'ce', 'son', 'une', 'sur', 'avec', 'ne', 'se', 'pas', 'tout', 'plus',
	'par', 'grand', 'en', 'me', 'même', 'la', 'lui', 'nous', 'comme', 'mais',
	'pouvoir', 'dire', 'elle', 'prendre', 'vous', 'ou', 'si', 'leur', 'faire',
	'mon', 'du', 'te', 'au', 'aussi', 'que', 'très', 'bien', 'où', 'sans',
	'oui', 'non', 'donc', 'alors', 'après', 'avant', 'ça', 'va', 'je', 'tu',
	'des', 'les', 'est', 'sont', 'était', 'ont', 'ses', 'ces', 'cette', 'celui'
}
