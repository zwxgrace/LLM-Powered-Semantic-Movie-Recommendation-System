import streamlit as st
import pandas as pd
import pickle
import os
from sentence_transformers import SentenceTransformer

from data_utils import load_and_preprocess_data
from engine import llm_recommend_hybrid, client

# ===============================
# Global configuration
# ===============================

# File paths for train and test datasets, upload the datasets as part of the repo
DATA_PATH = "tmdb_full_with_posters.csv"

# ===============================
# Data loading and preprocessing
# ===============================

@st.cache_resource
def init_assets():
    """
    Data Cleaning, Load Semantic Model, Read Vector Files.
    """
    df = load_and_preprocess_data(DATA_PATH)
    
    # load sentence transformer model for embedding generation
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # load pre-computed movie embeddings
    embedding_path = 'movie_embeddings.pkl'

    try:
        with open(embedding_path, 'rb') as f:
            embeddings = pickle.load(f)
        import numpy as np
        print("Embeddings loaded successfully.")
    except (ReferenceError, AttributeError, ModuleNotFoundError, ImportError):
        descriptions = df['description'].tolist()
        embeddings = model.encode(descriptions, show_progress_bar=True, batch_size=64)
        with open(embedding_path, 'wb') as f:
            pickle.dump(embeddings, f)
        st.success("Vector files have been regenerated and saved!")
        
    return df, embeddings, model

# ===============================
# Streamlit main app
# ===============================

def main():
    st.set_page_config(page_title="AI Movie Expert", page_icon="🎬", layout="wide")
    st.markdown("""
    <style>
        /* 1. 全局最底层字体放大 - 覆盖所有非标题文字 */
        html, body, [class*="st-"], .stMarkdown, p, span, label {
            font-size: 1.4rem !important; 
            line-height: 1.4 !important;
        }

        /* 2. 页面顶级标题 (Movie Expert) 极大化 */
        .stHeading h1 {
            font-size: 4.5rem !important;
            padding-bottom: 1rem !important;
            font-weight: 800 !important;
        }
                
        /* 3. 强制主容器撑满屏幕，消除两侧巨大的留白 */
        .block-container {
            max-width: 95% !important;
            padding-top: 2rem !important;
            padding-bottom: 0rem !important;
        }

        /* 4. 搜索框：彻底解决文字截断问题 */
        .stTextInput input {
            font-size: 2.2rem !important; /* 字号适中 */
            height: 3rem !important;    /* 框体高度 */
            line-height: 1.2 !important; /* 强制行高 */
            padding: 0.5rem 1rem !important;
            border-radius: 10px !important;
        }
                
        /* 隐藏搜索框上方那个小标题，或者让它也变大 */
        [data-testid="stWidgetLabel"] p {
            font-size: 2rem !important;
            font-weight: bold !important;
        }

        /* 3. 字体全局大放血 */
        /* 针对所有 Markdown、标题、标签进行缩放 */
        .stMarkdown p, .stMarkdown li, span {
            font-size: 4rem !important; /* 基础正文非常大 */
            line-height: 5 !important;
        }
        h2 { font-size: 4rem !important; margin-bottom: 0.5rem !important; } /* 电影标题 */
        h3 { font-size: 4rem !important; }

        /* 4. 图片尺寸：让它更有存在感 */
        [data-testid="stImage"] img {
            width: 100% !important; /* 占满它所在的列 */
            max-width: 400px !important; /* 但不要超过400px，防止模糊 */
            border-radius: 15px !important;
        }

        /* 5. 推荐理由框 (Success Box) 的文字 */
        .stSuccess div {
            font-size: 1.4rem !important;
            padding: 1.5rem !important;
        }

        /* 6. 分数 (Metric) 放大 */
        [data-testid="stMetricValue"] {
            font-size: 3rem !important;
            font-weight: bold !important;
        }
    </style>
""", unsafe_allow_html=True)

    st.title("Movie Expert 🎬🍺🍿")

    # --- Initialize assets ---
    with st.spinner("System resources initializing..."):
        df, embeddings, model = init_assets()

    # --- Set sidebar ---
    with st.sidebar:
        st.header("⚙️ Search Settings")
        top_n = st.slider("Recommendation Count", min_value=1, max_value=10, value=5)
        st.divider()
        st.info("💡 Tip: You can input complex requests including years, genres, moods, or specific themes.")

    # --- Main search interface ---
    query = st.text_input(
        "🔍 Please enter your request:", 
        placeholder="e.g., 'Looking for a heartwarming comedy from the 90s with a high rating'"
    )

    if query:
        # use st.status to display workflow progress and enhance user experience
        with st.status("matching...", expanded=True) as status:
            st.write("🧠 analyzing your demand...")
            
            # --- Call the final model function ---
            try:
                results_df = llm_recommend_hybrid(
                    user_query=query, 
                    df=df, 
                    embeddings=embeddings, 
                    model=model, 
                    client=client, 
                    top_n=top_n,
                    verbose=False  
                )
                status.update(label="Search completed!", state="complete", expanded=False)
            except Exception as e:
                status.update(label="Search error", state="error")
                st.error(f"Recommendation pipeline failed: {e}")
                results_df = pd.DataFrame()

        # --- Display recommendation results ---
        if not results_df.empty:
            st.subheader(f"🎯  {len(results_df)} Movies Recommended")
            
            for _, row in results_df.iterrows():
                # use st.container to group each movie's display, with columns for score and details
                with st.container(border=True):
                    col1, col2 = st.columns([2, 5])
                    
                    with col1:
                        poster_url = row.get('poster_url')
                        if poster_url and str(poster_url).startswith('http'):
                            st.image(poster_url, use_container_width=True)
                        else:
                            st.image("https://via.placeholder.com/500x750?text=No+Poster", use_container_width=True)
                        
                        # 分数紧贴在海报下面
                        st.metric("Match Score", f"{row['hybrid_score']:.2f}")
                        # 可以顺便展示下用户原始评分
                        st.caption(f"User Rating: ⭐ {row['rating']}/10")
                    
                    with col2:
                        st.write(f"### {row['rank']}. {row['title']} ({row['year']})")
                        st.markdown(f"**Type:** `{row['genres']}`  |  **Rating:** ⭐ `{row['rating']}/10`")
                        
                        # Highlight the AI-generated explanation
                        st.success(f"**AI Recommendation Reason:** {row['explanation']}")
                        
                        # by default, only show the overview when user clicks to expand, to keep the interface clean
                        with st.expander("📖 View Plot Summary"):
                            st.write(row['overview'])
        else:
            st.warning("No movies found matching your criteria. Try relaxing the filters (e.g., year or rating) and search again.")

# Run the Streamlit app
if __name__ == "__main__":
    main()
