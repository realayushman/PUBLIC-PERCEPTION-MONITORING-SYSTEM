# src/data/data_preprocessing.py

import numpy as np
import pandas as pd
import os
import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import logging

# logging configuration
logger = logging.getLogger('data_preprocessing')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('preprocessing_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Download required NLTK data
nltk.download('wordnet', quiet=True)  # Added quiet=True
nltk.download('stopwords', quiet=True)  # Added quiet=True

# Define the preprocessing function - SAME NAME, IMPROVED LOGIC
def preprocess_comment(comment):
    """Apply preprocessing transformations to a comment."""
    try:
        # Early return for empty or invalid comments
        if not isinstance(comment, str) or not comment.strip():
            return ""
        
        original_comment = comment
        
        # 1. Remove URLs (common in Reddit comments)
        comment = re.sub(r'https?://\S+|www\.\S+', '', comment)
        
        # 2. Remove HTML entities
        comment = re.sub(r'&[a-z]+;', '', comment)
        
        # 3. Convert to lowercase
        comment = comment.lower()
        
        # 4. Remove trailing and leading whitespaces
        comment = comment.strip()
        
        # 5. Remove newline characters and multiple spaces
        comment = re.sub(r'\n+', ' ', comment)
        comment = re.sub(r'\s+', ' ', comment)
        
        # 6. Remove special characters but keep basic punctuation for sentiment
        # Keep: letters, numbers, basic punctuation (! ? , .) and spaces
        comment = re.sub(r'[^a-z0-9\s!?.,]', '', comment)
        
        # 7. Tokenize before stopword removal for better handling
        words = comment.split()
        
        # 8. Remove stopwords but retain important ones for sentiment analysis
        # Expanded list of sentiment-bearing words to keep
        sentiment_words = {
            'not', 'no', 'nor', 'neither', 'never', 'none', 
            'but', 'however', 'although', 'though', 'yet', 'still',
            'very', 'extremely', 'absolutely', 'completely', 'totally',
            'quite', 'rather', 'somewhat', 'slightly', 'fairly',
            'good', 'bad', 'great', 'terrible', 'awesome', 'awful',
            'love', 'hate', 'like', 'dislike', 'amazing', 'horrible',
            'best', 'worst', 'better', 'worse', 'excellent', 'poor'
        }
        
        stop_words = set(stopwords.words('english')) - sentiment_words
        
        # Filter words: keep if not in stopwords or if it's a sentiment word
        filtered_words = []
        for word in words:
            if word in sentiment_words or word not in stop_words:
                filtered_words.append(word)
        
        # 9. Lemmatize the words (more accurate than stemming for sentiment)
        lemmatizer = WordNetLemmatizer()
        lemmatized_words = []
        
        for word in filtered_words:
            # Try to lemmatize, fallback to original word if error
            try:
                lemmatized_words.append(lemmatizer.lemmatize(word))
            except:
                lemmatized_words.append(word)
        
        # 10. Remove very short words (likely noise) but keep negation words
        final_words = []
        for word in lemmatized_words:
            if len(word) >= 2 or word in {'no', 'ok'}:
                final_words.append(word)
        
        result = ' '.join(final_words).strip()
        
        # Log if processing removed too much (debug info)
        if len(original_comment.split()) > 5 and len(result.split()) <= 1:
            logger.debug(f"Heavy reduction: '{original_comment[:50]}...' -> '{result}'")
        
        return result
        
    except Exception as e:
        logger.error(f"Error in preprocessing comment: {e}")
        # Return original comment as fallback (same as before)
        return comment if isinstance(comment, str) else ""

def normalize_text(df):
    """Apply preprocessing to the text data in the dataframe."""
    try:
        # Check if column exists
        if 'clean_comment' not in df.columns:
            logger.error("Column 'clean_comment' not found in dataframe")
            raise ValueError("Column 'clean_comment' not found")
        
        # Get initial stats for logging
        initial_count = len(df)
        initial_non_empty = df['clean_comment'].astype(str).str.strip().ne('').sum()
        
        logger.debug(f'Starting text normalization on {initial_count} rows')
        
        # Apply preprocessing with error handling per row
        df['clean_comment'] = df['clean_comment'].apply(preprocess_comment)
        
        # Get final stats
        final_non_empty = df['clean_comment'].str.strip().ne('').sum()
        removed_count = initial_non_empty - final_non_empty
        
        logger.debug(f'Text normalization completed')
        logger.debug(f'Initial non-empty: {initial_non_empty}')
        logger.debug(f'Final non-empty: {final_non_empty}')
        logger.debug(f'Rows became empty after processing: {removed_count}')
        
        return df
    except Exception as e:
        logger.error(f"Error during text normalization: {e}")
        raise

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """Save the processed train and test datasets."""
    try:
        interim_data_path = os.path.join(data_path, 'interim')
        logger.debug(f"Creating directory {interim_data_path}")
        
        os.makedirs(interim_data_path, exist_ok=True)
        logger.debug(f"Directory {interim_data_path} created or already exists")
        
        # Save with compression to reduce file size
        train_path = os.path.join(interim_data_path, "train_processed.csv")
        test_path = os.path.join(interim_data_path, "test_processed.csv")
        
        # Save with compression if files are large
        train_data.to_csv(train_path, index=False)
        test_data.to_csv(test_path, index=False)
        
        # Log file sizes
        train_size = os.path.getsize(train_path) / (1024 * 1024)  # MB
        test_size = os.path.getsize(test_path) / (1024 * 1024)    # MB
        
        logger.debug(f"Processed data saved to {interim_data_path}")
        logger.debug(f"Train file size: {train_size:.2f} MB")
        logger.debug(f"Test file size: {test_size:.2f} MB")
        logger.debug(f"Train shape: {train_data.shape}")
        logger.debug(f"Test shape: {test_data.shape}")
        
    except Exception as e:
        logger.error(f"Error occurred while saving data: {e}")
        raise

def main():
    try:
        logger.debug("Starting data preprocessing...")
        
        # Fetch the data from data/raw
        train_path = './data/raw/train.csv'
        test_path = './data/raw/test.csv'
        
        logger.debug(f"Loading train data from {train_path}")
        logger.debug(f"Loading test data from {test_path}")
        
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)
        
        logger.debug(f'Train data loaded: {train_data.shape}')
        logger.debug(f'Test data loaded: {test_data.shape}')
        
        # Check for required columns
        required_columns = ['clean_comment']
        for col in required_columns:
            if col not in train_data.columns:
                logger.error(f"Required column '{col}' missing in train data")
                raise ValueError(f"Column '{col}' not found")
            if col not in test_data.columns:
                logger.error(f"Required column '{col}' missing in test data")
                raise ValueError(f"Column '{col}' not found")
        
        # Preprocess the data
        logger.debug("Preprocessing training data...")
        train_processed_data = normalize_text(train_data)
        
        logger.debug("Preprocessing test data...")
        test_processed_data = normalize_text(test_data)
        
        # Save the processed data
        save_data(train_processed_data, test_processed_data, data_path='./data')
        
        logger.info('Data preprocessing completed successfully!')
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        print(f"Error: Make sure the data files exist in ./data/raw/")
    except Exception as e:
        logger.error(f'Failed to complete the data preprocessing process: {e}')
        print(f"Error: {e}")

if __name__ == '__main__':
    main()