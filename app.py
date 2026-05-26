"""
Streamlit Web Application - SIMPLIFIED VERSION
Visual Product Similarity & Image-Based Recommendation System
Amazon-Style Interface with Filtering and Error Handling
"""

import streamlit as st
from PIL import Image
import os
import logging
from search_engine import SearchEngine

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Product Recommender",
    page_icon="🛍️",
    layout="wide"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================

st.markdown("""
    <style>
    .product-card {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .similarity-score {
        font-size: 24px;
        font-weight: bold;
        color: #FF9900;
    }
    .rank-badge {
        background-color: #FF9900;
        color: white;
        padding: 5px 10px;
        border-radius: 5px;
        font-weight: bold;
    }
    .category-badge {
        background-color: #146eb4;
        color: white;
        padding: 3px 8px;
        border-radius: 3px;
        font-size: 12px;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# INITIALIZE SEARCH ENGINE (NO CACHING)
# ============================================================================

@st.cache_resource
def load_engine():
    """Load search engine"""
    return SearchEngine()

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def filter_results_by_category(results, category):
    """Filter results by category"""
    if category == "All Categories":
        return results
    return [r for r in results if r['category'] == category]

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    st.markdown("# 🛍️ Product Similarity & Recommendation System")
    st.markdown("### Amazon-Style Visual Search Engine")
    st.markdown("---")
    
    # Load engine
    engine = load_engine()
    
    # Get categories
    categories = sorted(engine.metadata['category'].unique().tolist())
    
    # ====================================================================
    # SIDEBAR
    # ====================================================================
    
    st.sidebar.markdown("## ⚙️ Configuration")
    k = st.sidebar.slider("Number of results", 1, 20, 10)
    selected_category = st.sidebar.selectbox("Filter by category", ["All Categories"] + categories)
    similarity_threshold = st.sidebar.slider("Min similarity", 0.0, 1.0, 0.0, 0.05)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("## 📊 Dataset Info")
    st.sidebar.info(f"""
    **Total Products:** {len(engine.metadata)}
    **Categories:** {len(categories)}
    **Embedding Dim:** 2048
    **Model:** ResNet50
    """)
    
    # ====================================================================
    # MAIN CONTENT
    # ====================================================================
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("## 📸 Upload Query Image")
        option = st.radio("Select input method:", ["Upload Image", "Use Sample Image"])
        
        query_image_path = None
        query_image = None
        
        if option == "Upload Image":
            uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
            if uploaded_file is not None:
                query_image = Image.open(uploaded_file)
                temp_path = "temp_query.jpg"
                query_image.save(temp_path)
                query_image_path = temp_path
                st.success("✅ Image uploaded!")
        
        else:
            sample_images = sorted([f for f in os.listdir('data/images') if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            if sample_images:
                sample_name = st.selectbox("Select sample:", sample_images[:50])
                query_image_path = os.path.join('data/images', sample_name)
                query_image = Image.open(query_image_path)
        
        if query_image is not None:
            st.image(query_image, caption="Query Image", use_column_width=True)
    
    with col2:
        st.markdown("## 🔍 Similar Products")
        
        if query_image_path is not None:
            with st.spinner("🔄 Searching..."):
                results = engine.search_by_image(query_image_path, k=k)
                
                if not results:
                    st.error("No results found")
                else:
                    # Filter by category
                    filtered_results = filter_results_by_category(results, selected_category)
                    
                    # Filter by similarity
                    filtered_results = [r for r in filtered_results if r['similarity_score'] >= similarity_threshold]
                    
                    if not filtered_results:
                        st.warning(f"No results matching filters")
                    else:
                        st.success(f"Found {len(filtered_results)} products!")
                        st.markdown("---")
                        
                        # Pagination
                        results_per_page = 5
                        total_pages = (len(filtered_results) + results_per_page - 1) // results_per_page
                        
                        if 'page' not in st.session_state:
                            st.session_state.page = 1
                        
                        col_prev, col_page, col_next = st.columns([1, 2, 1])
                        
                        with col_prev:
                            if st.button("← Previous", disabled=(st.session_state.page == 1)):
                                st.session_state.page -= 1
                                st.rerun()
                        
                        with col_page:
                            st.markdown(f"<div style='text-align: center;'><strong>Page {st.session_state.page} of {total_pages}</strong></div>", unsafe_allow_html=True)
                        
                        with col_next:
                            if st.button("Next →", disabled=(st.session_state.page == total_pages)):
                                st.session_state.page += 1
                                st.rerun()
                        
                        st.markdown("---")
                        
                        # Display results
                        start_idx = (st.session_state.page - 1) * results_per_page
                        end_idx = min(start_idx + results_per_page, len(filtered_results))
                        
                        for result in filtered_results[start_idx:end_idx]:
                            st.markdown(f"""
                            <div class="product-card">
                                <div style="display: flex; justify-content: space-between;">
                                    <div>
                                        <span class="rank-badge">#{result['rank']}</span>
                                        <span class="category-badge">{result['category'].upper()}</span>
                                        <h4 style="display: inline; margin-left: 10px;">{result['product_id']}</h4>
                                    </div>
                                    <div class="similarity-score">{result['similarity_score']:.2%}</div>
                                </div>
                                <p style="color: #666;">Distance: {result['distance']:.4f}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            try:
                                product_image = Image.open(result['image_path'])
                                st.image(product_image, width=200)
                            except:
                                st.warning(f"Could not load image")
                            
                            st.markdown("---")
        else:
            st.info("👈 Please upload or select an image")
    
    # ====================================================================
    # FOOTER
    # ====================================================================
    
    st.markdown("---")
    st.markdown("""
    ### 📚 About This System
    
    **Visual Product Similarity & Image-Based Recommendation System**
    
    - **ResNet50**: Feature extraction (2048-dim embeddings)
    - **FAISS**: Fast similarity search
    - **Streamlit**: Web interface
    
    **Features:**
    - 🖼️ Image upload and sample selection
    - 🏷️ Category filtering
    - 📊 Similarity scoring
    - ⚡ Fast FAISS search
    - 📄 Pagination
    - 🛡️ Error handling
    """)

if __name__ == "__main__":
    main()
