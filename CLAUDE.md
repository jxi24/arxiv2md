# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

`arxiv2md` is a single-file Python script that fetches today's arXiv RSS feed across one or more archives, ranks the articles by relevance to a user-defined keyword list using TF-IDF cosine similarity, and writes the results as a Markdown file to `news/arxiv_YYMMDD.md`.

## Running the Script

The script uses [uv](https://github.com/astral-sh/uv) inline script dependencies — no virtual environment setup needed:

```bash
# Fetch default archives (hep-ph, hep-th, hep-lat, hep-ex, nucl-th, nucl-ex, cs.LG, stat.ML)
./arxiv2md

# Fetch specific archives
./arxiv2md --archive hep-ph cs.LG

# Include replacement/updated submissions
./arxiv2md -r

# Link to abstract page instead of PDF
./arxiv2md -a
```

Output is written to `news/arxiv_YYMMDD.md` (the `news/` directory is gitignored and must exist before running).

## Architecture

The pipeline in `arxiv2md` flows linearly:

1. **`collect_articles`** — fetches the arXiv RSS feed via `feedparser`, deduplicates by entry ID, optionally drops `"Type: replace"` entries.
2. **`preprocess_article` / `preprocess_keywords`** — strips LaTeX math tokens, numbers, and lemmatizes all words using NLTK's WordNet lemmatizer to normalize vocabulary before vectorization.
3. **`vectorize`** — builds a TF-IDF matrix from `arxiv_keywords.txt` plus all article texts. The n-gram range is inferred dynamically from the keyword file (min/max words per phrase).
4. **`rank_sort_articles`** — computes cosine similarity between the keyword vector (row 0) and all article vectors, returns articles sorted by descending relevance.
5. **`create_markdown`** — re-orders articles by rank, cleans up author strings (handles collaboration entries and parenthesized author counts), converts assembled HTML to Markdown via `markdownify`.

## Keywords File

`arxiv_keywords.txt` — one keyword or phrase per line. Multi-word phrases are supported; the n-gram range of the TF-IDF vectorizer is set automatically to cover the shortest and longest phrase in the file. Edit this file to tune which articles float to the top.

## Dependencies

Managed inline by uv (no `pyproject.toml` or `requirements.txt`):
- `feedparser` — RSS parsing
- `markdownify` — HTML → Markdown conversion
- `scikit-learn` — TF-IDF vectorization and cosine similarity
- `nltk` — tokenization and lemmatization (requires `wordnet` and `averaged_perceptron_tagger` NLTK data downloaded at runtime)
