"""
Download and preprocess English books from Project Gutenberg.

This script:
- Fetches a catalog of Project Gutenberg books.
- Filters for English-language books and constructs download URLs.
- Downloads each book's plain text file (with delay to avoid rate limit).
- Tokenizes books into sentences and words.
- Removes stopwords and lemmatizes tokens.
- Saves processed books to a local cache for faster re-use.
- Processes books in parallel with limited threads and a progress bar.

Main function: `get_books_tokenized()`
Example usage and sample output shown at end of file.
"""

import os
import re
import csv
import gzip
import pickle
import requests
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

import nltk
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

CATALOG_URL = "https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv.gz"
CACHE_DIR = "book_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def get_ebook_urls(limit=100):
    r = requests.get(CATALOG_URL)
    r.raise_for_status()
    content = gzip.decompress(r.content).decode("utf-8", errors="replace")
    urls = []
    reader = csv.DictReader(content.splitlines())
    for row in reader:
        if row.get("Language") == "en":
            book_id = row["Text#"]
            url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
            urls.append(url)
            if len(urls) >= limit:
                break
    return urls

def get_cache_path(url):
    book_id = url.split("/")[-1].replace(".txt", "")
    return os.path.join(CACHE_DIR, f"{book_id}.pkl")

def download_and_process(url, delay=1.0):
    cache_path = get_cache_path(url)
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    try:
        text = requests.get(url, timeout=10).text
        text = text.lower()
        sentences = sent_tokenize(text)
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
        processed = []
        for sent in sentences:
            tokens = word_tokenize(sent)
            tokens = [t for t in tokens if re.match(r"^[a-z']+$", t)]
            tokens = [t for t in tokens if t not in stop_words]
            tokens = [lemmatizer.lemmatize(t) for t in tokens]
            if tokens:
                processed.append(tokens)
        with open(cache_path, "wb") as f:
            pickle.dump(processed, f)
        time.sleep(delay)
        return processed
    except Exception as e:
        print(f"Error processing {url}: {e}")
        return []

def get_books_tokenized(limit=100, max_workers=2, delay=1.0):
    urls = get_ebook_urls(limit)
    tokenized_corpus = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(download_and_process, url, delay) for url in urls]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Books processed"):
            tokenized_corpus.extend(future.result())
    return tokenized_corpus

# Example usage:
if __name__ == "__main__":
    sentences = get_books_tokenized(limit=10, max_workers=2, delay=2.0)
    print(f"{len(sentences)} tokenized sentences collected.")
    if sentences:
        print("Sample:", sentences[0])