"""
MelodyMatch AI - Music Recommendation System
Main Streamlit Application

College Project using SrvDB Vector Database
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from srvdb_manager import SrvDBManager
from recommender import BookRecommender
import time

# Page Configuration
st.set_page_config(
    page_title="PageTurner AI",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Professional/Library Theme
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background-color: #Fdfbf7; /* Cream/Paper white */
        color: #2c3e50;
        font-family: 'Merriweather', 'Georgia', serif;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #2c3e50;
        color: #ecf0f1;
    }
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] span {
        color: #ecf0f1;
    }
    
    /* Typography */
    h1, h2, h3, h4 {
        font-family: 'Playfair Display', 'Times New Roman', serif;
        color: #2c3e50;
    }
    
    /* Buttons */
    .stButton>button {
        width: 100%;
        border-radius: 4px;
        background-color: #bfa378; /* Gold/Antique Paper */
        color: white;
        border: none;
        padding: 8px 16px;
        font-family: 'Lato', sans-serif;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #a68b5e;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Input Fields */
    .stTextInput>div>div>input {
        background-color: #ffffff;
        border: 1px solid #bdc3c7;
        border-radius: 4px;
        padding: 10px;
        font-family: 'Lato', sans-serif;
    }
    
    /* Book Card */
    .book-card {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 2px;
        padding: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
        margin-bottom: 20px;
    }
    .book-card:hover {
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
        transform: translateY(-2px);
    }
    
    /* Metrics */
    .metric-container {
        pointer-events: none;
        background-color: #ffffff;
        border: 1px solid #ecf0f1;
        padding: 15px;
        border-radius: 2px;
        text-align: center;
    }
    .metric-value {
        font-family: 'Playfair Display', serif;
        font-size: 28px;
        color: #2c3e50;
        font-weight: bold;
    }
    .metric-label {
        font-family: 'Lato', sans-serif;
        font-size: 12px;
        color: #7f8c8d;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Rating Stars */
    .star-rating {
        color: #f1c40f;
        font-size: 16px;
    }
</style>
""", unsafe_allow_html=True)

# Application State
if 'recommender' not in st.session_state:
    st.session_state.recommender = None
    st.session_state.bookmarks = []
    st.session_state.search_history = []
    st.session_state.search_results = None

def init_system():
    """Initialize the recommendation system"""
    if st.session_state.recommender is None:
        try:
            with st.spinner("� Initializing PageTurner AI..."):
                # Using specific book vectors path
                db_manager = SrvDBManager(db_path="./db/book_vectors")
                st.session_state.recommender = BookRecommender(db_manager)
            st.success("✅ System initialized successfully!")
            return True
        except Exception as e:
            st.error(f"❌ Failed to initialize system: {e}")
            st.info("💡 Please ensure the book vector database is built.")
            return False
    return True

def display_book_card(book_data, score=None):
    """Display a professional book card"""
    # Create a nice layout
    with st.container():
        col1, col2 = st.columns([1, 4])
        
        with col1:
            st.image(book_data.get('cover_url', 'https://via.placeholder.com/150'), use_column_width=True)
            
        with col2:
            st.markdown(f"### {book_data['title']}")
            st.markdown(f"**By {book_data['author']}**")
            
            # Star Rating
            rating = book_data.get('rating', 0)
            full_stars = int(rating)
            has_half = rating % 1 >= 0.5
            stars = "⭐" * full_stars + ("½" if has_half else "")
            st.markdown(f"<span class='star-rating'>{stars}</span> ({rating}/5.0)", unsafe_allow_html=True)
            
            st.markdown(f"_{book_data['genre']} • {book_data.get('year', 'N/A')} • {book_data.get('pages', 0)} pages_")
            
            with st.expander("📖 Read Description"):
                st.write(book_data.get('description', 'No description available.'))
            
            if score:
                st.progress(score)
                st.caption(f"Relevance: {int(score*100)}%")
            
            # Actions
            c1, c2 = st.columns(2)
            with c1:
                # Toggle Bookmark
                is_bookmarked = book_data['id'] in st.session_state.bookmarks
                btn_label = "❌ Remove Bookmark" if is_bookmarked else "🔖 Bookmark"
                if st.button(btn_label, key=f"bk_{book_data['id']}"):
                    if is_bookmarked:
                        st.session_state.bookmarks.remove(book_data['id'])
                    else:
                        st.session_state.bookmarks.append(book_data['id'])
                    st.rerun()

def search_page():
    """Main Search Interface"""
    st.title("📚 PageTurner AI")
    st.markdown("_Your intelligent literary companion_")
    
    # Search Bar
    with st.container():
        col1, col2 = st.columns([3, 1])
        with col1:
            query = st.text_input("What are you looking for?", placeholder="Search by title, author, or topic...", key="search_query")
        with col2:
            sort_by = st.selectbox("Sort By", ["Relevance", "Rating", "Newest"])
    
    # Filters
    with st.expander("⚙️ Refine Criteria", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            genres = st.session_state.recommender.get_available_genres()
            selected_genres = st.multiselect("Genres", genres)
        with c2:
            rating_range = st.slider("Min Rating", 1.0, 5.0, (3.5, 5.0))
        with c3:
            pages_range = st.slider("Page Count", 0, 1000, (100, 800))
    
    # Search Action
    if st.button("🔍 Find Books", type="primary") or (query and not st.session_state.search_results):
        with st.spinner("Curating your reading list..."):
            results = st.session_state.recommender.search(
                query=query,
                genres=selected_genres,
                rating_range=rating_range,
                pages_range=pages_range,
                k=15
            )
            st.session_state.search_results = results
            if query:
                st.session_state.search_history.append({"query": query, "time": time.time()})
    
    # Results Display
    st.divider()
    if st.session_state.search_results:
        st.markdown(f"**Found {len(st.session_state.search_results)} books matching your criteria**")
        for book, score in st.session_state.search_results:
            display_book_card(book, score)
            st.divider()

def analytics_page():
    """Analytics and visualizations"""
    st.title("📊 Music Analytics")
    
    if st.session_state.recommender:
        # Genre distribution
        st.subheader("🎸 Genre Distribution")
        genre_data = st.session_state.recommender.get_genre_distribution()
        
        fig = px.pie(
            values=list(genre_data.values()),
            names=list(genre_data.keys()),
            title="Songs by Genre",
            color_discrete_sequence=px.colors.sequential.Plasma
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Favorites analysis
        if st.session_state.favorites:
            st.subheader("❤️ Your Favorite Songs Analysis")
            
            fav_data = st.session_state.recommender.analyze_favorites(
                st.session_state.favorites
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Favorite genres
                fig = px.bar(
                    x=list(fav_data['genres'].keys()),
                    y=list(fav_data['genres'].values()),
                    title="Your Favorite Genres",
                    labels={'x': 'Genre', 'y': 'Count'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Audio features radar
                features = ['Energy', 'Valence', 'Danceability', 'Acousticness']
                values = [
                    fav_data['avg_energy'],
                    fav_data['avg_valence'],
                    fav_data['avg_danceability'],
                    fav_data['avg_acousticness']
                ]
                
                fig = go.Figure(data=go.Scatterpolar(
                    r=values,
                    theta=features,
                    fill='toself'
                ))
                fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    showlegend=False,
                    title="Your Music Taste Profile"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Search history
        if st.session_state.search_history:
            st.subheader("🔍 Recent Searches")
            history_df = pd.DataFrame(st.session_state.search_history)
            st.dataframe(history_df, use_container_width=True)

def dashboard_page():
    """Analytics Dashboard"""
    st.title("📊 Literary Analytics")
    
    stats = st.session_state.recommender.get_genre_distribution()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Collection Composition")
        fig = px.pie(
            values=list(stats.values()),
            names=list(stats.keys()),
            color_discrete_sequence=px.colors.sequential.RdBu,
            hole=0.4
        )
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.subheader("Library Stats")
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value">{sum(stats.values())}</div>
            <div class="metric-label">Total Books</div>
        </div>
        <br>
        <div class="metric-container">
            <div class="metric-value">{len(stats)}</div>
            <div class="metric-label">Unique Genres</div>
        </div>
        """, unsafe_allow_html=True)

def main():
    if init_system():
        # Sidebar Navigation
        with st.sidebar:
            st.title("PageTurner AI")
            page = st.radio("Navigation", ["Search", "Dashboard", "My Bookmarks"])
            
            st.divider()
            st.markdown("### About")
            st.info("Powered by SrvDB Vector Database.\nDiscover your next great read.")

        if page == "Search":
            search_page()
        elif page == "Dashboard":
            dashboard_page()
        elif page == "My Bookmarks":
            st.title("🔖 My Bookmarks")
            if st.session_state.bookmarks:
                st.write("Your bookmarked books will appear here.")
                # Simple list for now as full retrieval needs ID lookup
                for i, book_id in enumerate(st.session_state.bookmarks):
                    st.markdown(f"{i+1}. **{book_id}** (Full details require ID lookup)")
            else:
                st.write("You haven't bookmarked any books yet.")

if __name__ == "__main__":
    main()