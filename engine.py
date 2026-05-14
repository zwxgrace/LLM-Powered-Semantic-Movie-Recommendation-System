import json
import re
import os
import time
import pandas as pd
import numpy as np
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi

# ===============================
# Global configuration
# ===============================

os.environ["OPENROUTER_API_KEY"] = "YOUR_OPENROUTER_API_KEY_HERE "  # <-- Set your OpenRouter API key here or in your environment variables

client = OpenAI(
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
    default_headers={
        "HTTP-Referer": "http://localhost",
        "X-Title": "Movie Recommender Project"
    }
)

# LLM_MODEL = "google/gemma-3-27b-it:free"
LLM_MODEL = "deepseek/deepseek-chat-v3.1"

# ===============================
# Recommendation Pipeline
# ===============================

def extract_llm_text(response):
    """Extract text from LLM response, handling reasoning-only models."""
    msg = response.choices[0].message
    # Prefer content (normal models put the answer here)
    if msg.content:
        return msg.content.strip()
    # Some free models put output in 'reasoning' instead of 'content'
    reasoning = getattr(msg, 'reasoning', None)
    if reasoning:
        # If reasoning contains JSON, extract it
        if '{' in reasoning:
            start = reasoning.find('{')
            end = reasoning.rfind('}')
            if start != -1 and end > start:
                return reasoning[start:end+1]
        return reasoning.strip()
    return ""

ALLOWED_GENRES = [
    "Action", "Adventure", "Animation", "Comedy", "Crime", "Documentary",
    "Drama", "Family", "Fantasy", "History", "Horror", "Music", "Mystery",
    "Romance", "Science Fiction", "Thriller", "War", "Western"
]
# align the allowed genres with the ones in our dataset (after cleaning)

SYSTEM_PROMPT = f"""You are a movie recommendation assistant.

Given a user's natural language query about what kind of movie they want to watch, extract structured information to help find the best matches.

Return a JSON object with exactly these fields:
- "search_query": a concise description of the ideal movie for semantic search
- "genres": list of relevant genres chosen only from: {", ".join(ALLOWED_GENRES)}
- "mood": the mood or tone the user wants, as a short string, or null if unclear
- "themes": list of key themes
- "min_year": minimum release year, or null if not specified
- "max_year": maximum release year, or null if not specified
- "min_rating": minimum vote average on a 0-10 scale, or null if not specified

Rules:
- Return only valid JSON
- Do not include markdown fences
- Do not add extra keys
- If a field is unknown, use null for scalars and [] for lists
- "genres" must contain only allowed genres
"""

def _extract_json_object(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start:end+1]
    return text

def _safe_json_loads(text: str):
    text = _extract_json_object(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    cleaned = text
    cleaned = cleaned.replace("\u201c", '"').replace("\u201d", '"').replace("\u2019", "'")
    cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        print("\u26a0\ufe0f Raw LLM output was not valid JSON:")
        print(text[:200])
        return {"search_query": None, "genres": [], "mood": None, "themes": [], "min_year": None, "max_year": None, "min_rating": None}

def _normalize_parsed_output(data: dict) -> dict:
    allowed = set(ALLOWED_GENRES)
    result = {
        "search_query": data.get("search_query"),
        "genres": data.get("genres", []),
        "mood": data.get("mood"),
        "themes": data.get("themes", []),
        "min_year": data.get("min_year"),
        "max_year": data.get("max_year"),
        "min_rating": data.get("min_rating"),
    }
    if not isinstance(result["genres"], list):
        result["genres"] = []
    result["genres"] = [g for g in result["genres"] if isinstance(g, str) and g in allowed]
    if not isinstance(result["themes"], list):
        result["themes"] = []
    if result["search_query"] is not None:
        result["search_query"] = str(result["search_query"]).strip()
    if result["mood"] is not None:
        result["mood"] = str(result["mood"]).strip()
    for key in ["min_year", "max_year"]:
        if result[key] is not None:
            try:
                result[key] = int(result[key])
            except (TypeError, ValueError):
                result[key] = None
    if result["min_rating"] is not None:
        try:
            result["min_rating"] = float(result["min_rating"])
        except (TypeError, ValueError):
            result["min_rating"] = None
    return result

def parse_user_query(user_query: str) -> dict:
    try:
        combined_prompt = SYSTEM_PROMPT + "\n\nUser query:\n" + user_query
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "user", "content": combined_prompt}
            ],
            temperature=0.1,
            max_tokens=400,
        )
        raw = extract_llm_text(response)
        if not raw:
            print("\u26a0\ufe0f LLM returned empty content. Using fallback.")
            return _normalize_parsed_output({"search_query": user_query})
        parsed = _safe_json_loads(raw)
        return _normalize_parsed_output(parsed)
    except Exception as e:
        print(f"\u26a0\ufe0f Query parsing failed: {e}")
        return _normalize_parsed_output({"search_query": user_query})

def generate_recommendation_explanations(user_query, results, client):
    """
    Generate one plain-text explanation per movie.
    No JSON parsing needed.
    """
    explanations = []

    for i, r in enumerate(results):
        if i > 0:
            time.sleep(2)  # Avoid free-tier rate limits

        prompt = f"""The user asked:
"{user_query}"

Recommended movie:
Title: {r['title']}
Year: {r['year']}
Genres: {r['genres']}
Rating: {r['rating']}/10
Plot: {r['overview']}

Write 1-2 concise sentences explaining why this movie matches the user's request.
Return only the explanation text, nothing else.
"""

        explanation = "This movie matches the requested tone, themes, and style."

        try:
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "user", "content": "You are a helpful movie recommendation assistant. Return only a short explanation.\n\n" + prompt}
                ],
                temperature=0.3,
                max_tokens=120,
            )

            text = extract_llm_text(response)
            if text:
                explanation = text
        except Exception as e:
            print(f"\u26a0\ufe0f Explanation failed for {r['title']}: {e}")

        explanations.append({
            "rank": r["rank"],
            "title": r["title"],
            "explanation": explanation
        })

    return explanations

def simple_tokenize(text):
    if pd.isna(text):
        return []
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text.split()

def minmax_normalize(arr):
    arr = np.asarray(arr, dtype=float)
    if len(arr) == 0:
        return arr
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-12:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)

def hybrid_retrieve(
    user_query,
    df,
    embeddings,
    model,
    bm25,
    top_n=10,
    bm25_weight=0.4,
    semantic_weight=0.6,
    candidate_pool=300
):
    """
    Hybrid retrieval:
    1. BM25 keyword retrieval
    2. Semantic embedding retrieval
    3. Combine normalized scores
    """

    # --- BM25 scores over whole corpus ---
    query_tokens = simple_tokenize(user_query)
    bm25_scores = bm25.get_scores(query_tokens)

    # --- Semantic scores over whole corpus ---
    query_emb = model.encode([user_query], convert_to_numpy=True)
    semantic_scores = cosine_similarity(query_emb, embeddings)[0]

    # --- Normalize both ---
    bm25_norm = minmax_normalize(bm25_scores)
    semantic_norm = minmax_normalize(semantic_scores)

    # --- Combine ---
    hybrid_scores = bm25_weight * bm25_norm + semantic_weight * semantic_norm

    # --- Optional: take top candidate_pool before final top_n ---
    candidate_idx = np.argsort(hybrid_scores)[::-1][:candidate_pool]
    final_idx = candidate_idx[:top_n]

    results = df.iloc[final_idx][["title", "release_year", "genres_clean", "vote_average", "overview"]].copy()
    results["bm25_score"] = bm25_scores[final_idx]
    results["semantic_score"] = semantic_scores[final_idx]
    results["hybrid_score"] = hybrid_scores[final_idx]

    return results.sort_values("hybrid_score", ascending=False).reset_index(drop=True)

def hybrid_recall_rerank(
    user_query,
    df,
    embeddings,
    model,
    bm25,
    recall_k_bm25=200,
    recall_k_semantic=200,
    top_n=10,
    bm25_weight=0.4,
    semantic_weight=0.6
):
    """
    Hybrid recall:
    - recall top K from BM25
    - recall top K from semantic search
    - union candidates
    - rerank with weighted hybrid score
    """

    # BM25 recall
    query_tokens = simple_tokenize(user_query)
    bm25_scores_all = bm25.get_scores(query_tokens)
    bm25_top_idx = np.argsort(bm25_scores_all)[::-1][:recall_k_bm25]

    # Semantic recall
    query_emb = model.encode([user_query], convert_to_numpy=True)
    semantic_scores_all = cosine_similarity(query_emb, embeddings)[0]
    semantic_top_idx = np.argsort(semantic_scores_all)[::-1][:recall_k_semantic]

    # Candidate union
    candidate_idx = np.array(sorted(set(bm25_top_idx).union(set(semantic_top_idx))))

    # Candidate scores
    bm25_scores = bm25_scores_all[candidate_idx]
    semantic_scores = semantic_scores_all[candidate_idx]

    bm25_norm = minmax_normalize(bm25_scores)
    semantic_norm = minmax_normalize(semantic_scores)

    hybrid_scores = bm25_weight * bm25_norm + semantic_weight * semantic_norm

    order = np.argsort(hybrid_scores)[::-1][:top_n]
    final_idx = candidate_idx[order]

    results = df.iloc[final_idx][["title", "release_year", "genres_clean", "vote_average", "overview"]].copy()
    results["bm25_score"] = bm25_scores_all[final_idx]
    results["semantic_score"] = semantic_scores_all[final_idx]
    results["hybrid_score"] = hybrid_scores[order]

    return results.sort_values("hybrid_score", ascending=False).reset_index(drop=True)

def hybrid_recall_rerank_filtered(
    search_text,
    candidates,
    candidate_embeddings,
    model,
    bm25_weight=0.4,
    semantic_weight=0.6,
    top_n=10
):
    tokenized = candidates["bm25_text"].apply(simple_tokenize).tolist()
    local_bm25 = BM25Okapi(tokenized)

    bm25_scores = local_bm25.get_scores(simple_tokenize(search_text))
    query_emb = model.encode([search_text], convert_to_numpy=True)
    semantic_scores = cosine_similarity(query_emb, candidate_embeddings)[0]

    bm25_norm = minmax_normalize(bm25_scores)
    semantic_norm = minmax_normalize(semantic_scores)

    hybrid_scores = bm25_weight * bm25_norm + semantic_weight * semantic_norm
    top_idx = np.argsort(hybrid_scores)[::-1][:top_n]

    out = candidates.iloc[top_idx].copy()
    out["bm25_score"] = bm25_scores[top_idx]
    out["semantic_score"] = semantic_scores[top_idx]
    out["hybrid_score"] = hybrid_scores[top_idx]
    return out.sort_values("hybrid_score", ascending=False)

# ======================================================
# The Final Function: LLM-Enhanced Hybrid Recommendation
# ======================================================

def llm_recommend_hybrid(user_query, df, embeddings, model, client, top_n=5, verbose=True):
    parsed = parse_user_query(user_query)

    if verbose:
        print("📋 Parsed query:")
        print(json.dumps(parsed, indent=2))
        print()

    mask = pd.Series(True, index=df.index)

    if parsed.get("genres"):
        target_genres = {g.lower() for g in parsed["genres"]}
        mask &= df["genre_list"].apply(
            lambda gl: isinstance(gl, list) and bool(target_genres & {g.lower() for g in gl})
        )

    if parsed.get("min_year") is not None:
        mask &= df["release_year"] >= parsed["min_year"]

    if parsed.get("max_year") is not None:
        mask &= df["release_year"] <= parsed["max_year"]

    if parsed.get("min_rating") is not None:
        mask &= df["vote_average"] >= parsed["min_rating"]

    candidates = df[mask].copy()

    if verbose:
        print(f"🎬 {len(candidates)} movies pass the filters (from {len(df)} total)")

    if len(candidates) == 0:
        if verbose:
            print("No movies match the filters. Falling back to full dataset.")
        candidates = df.copy()

    search_text = parsed.get("search_query") or user_query

    candidate_indices = candidates.index.to_list()
    candidate_embeddings = embeddings[candidate_indices]

    ranked = hybrid_recall_rerank_filtered(
        search_text=search_text,
        candidates=candidates,
        candidate_embeddings=candidate_embeddings,
        model=model,
        bm25_weight=0.4,
        semantic_weight=0.6,
        top_n=top_n
    )

    results = []
    for rank, (_, row) in enumerate(ranked.head(top_n).iterrows(), start=1):
        results.append({
            "rank": rank,
            "title": row["title"],
            "genres": row.get("genres_clean", ""),
            "year": int(row["release_year"]) if pd.notna(row["release_year"]) else None,
            "rating": float(row["vote_average"]) if pd.notna(row["vote_average"]) else None,
            "overview": str(row.get("overview", "")),
            "bm25_score": float(row["bm25_score"]),
            "semantic_score": float(row["semantic_score"]),
            "hybrid_score": float(row["hybrid_score"]),
            "id": row.get("id"), 
            "poster_url": row.get("poster_url", ""),
        })

    llm_explanations = generate_recommendation_explanations(user_query, results, client)

    for r, e in zip(results, llm_explanations):
        r["explanation"] = e["explanation"]

    results_df = pd.DataFrame(results)

    if verbose:
        print(f"\n{'='*80}")
        print(f'🎯 Top {top_n} Hybrid Recommendations for: "{user_query}"')
        print(f"{'='*80}\n")
        for _, row in results_df.iterrows():
            print(f"{row['rank']}. {row['title']} ({row['year']})")
            print(f"   Genres: {row['genres']}")
            print(
                f"   Rating: {row['rating']} | "
                f"BM25: {row['bm25_score']:.3f} | "
                f"Semantic: {row['semantic_score']:.3f} | "
                f"Hybrid: {row['hybrid_score']:.3f}"
            )
            print(f"   Why it matches: {row['explanation']}")
            print()

    return results_df
