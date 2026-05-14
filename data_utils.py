import pandas as pd
import json
import ast
import re

def parse_json_names(text):
    """Extract 'name' values from a JSON-like list string, e.g. [{"id":1,"name":"Action"}]."""
    if pd.isna(text) or text.strip() == '':
        return []
    try:
        items = json.loads(text.replace("'", '"'))
        return [item['name'] for item in items if 'name' in item]
    except (json.JSONDecodeError, TypeError):
        return []

def build_description(row):
    """Combine multiple text fields into one rich description for embedding."""
    parts = []
    parts.append(f"Title: {row['title']}.")
    if row['tagline']:
        parts.append(f"Tagline: {row['tagline']}.")
    if row['overview']:
        parts.append(f"Plot: {row['overview']}")
    if row['genres_clean']:
        parts.append(f"Genres: {row['genres_clean']}.")
    if row['keywords_clean']:
        parts.append(f"Keywords: {row['keywords_clean']}.")
    if row['tags_text']:
        parts.append(f"Tags: {row['tags_text'][:200]}.")
    if pd.notna(row['release_year']):
        parts.append(f"Year: {int(row['release_year'])}.")
    if pd.notna(row['vote_average']):
        parts.append(f"Rating: {row['vote_average']}/10.")
    return ' '.join(parts)

def load_and_preprocess_data(csv_path):
    """
    Load movie data from a CSV file, clean and preprocess it for BM25 search and embedding generation.
    """
    # 1. load data
    df = pd.read_csv(csv_path)
    
    # 2. parse JSON columns
    df['genre_list'] = df['genres'].apply(parse_json_names)
    df['keyword_list'] = df['keywords'].apply(parse_json_names)
    df['production_company_list'] = df['production_companies'].apply(parse_json_names)
    df['production_country_list'] = df['production_countries'].apply(parse_json_names)
    df['spoken_language_list'] = df['spoken_languages'].apply(parse_json_names)

    df['genres_clean'] = df['genre_list'].apply(lambda x: ', '.join(x))
    df['keywords_clean'] = df['keyword_list'].apply(lambda x: ', '.join(x))

    # 3. handle missing values and dates
    df['overview'] = df['overview'].fillna('')
    df['tagline'] = df['tagline'].fillna('')
    df['tags_text'] = df['tags_text'].fillna('')
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
    df['release_year'] = df['release_date'].dt.year
    
    # Drop rows where runtime is missing (only 2)
    df = df.dropna(subset=['runtime']).copy()

    # 4. build description for embedding
    df['description'] = df.apply(build_description, axis=1)

    # 5. create BM25 text field
    df["bm25_text"] = (
        df["title"].fillna("") + " "
        + df["overview"].fillna("") + " "
        + df["genres_clean"].fillna("") + " "
        + df["keywords_clean"].fillna("")
    )

    print(f"✅ Data preprocessing completed, current shape: {df.shape}")
    return df