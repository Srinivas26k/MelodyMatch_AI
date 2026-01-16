"""
MelodyMatch AI - Entertainment Recommendation System
Main Streamlit Application
Supports Books, Music, and Movies
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from srvdb_manager import SrvDBManager
from recommender import MediaRecommender
from bookmark_manager import BookmarkManager
import time

# --- Setup & Config ---
st.set_page_config(
    page_title="MelodyMatch AI",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .metric-container {
        border: 1px solid #333;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        background-color: #0e1117;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #ff4b4b;
    }
    .metric-label {
        font-size: 1rem;
        color: #888;
    }
    .star-rating {
        color: #ffd700;
        font-size: 1.2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Application State
if 'recommenders' not in st.session_state:
    st.session_state.recommenders = {}
    st.session_state.bookmark_manager = None 
    st.session_state.bookmarks = []
    st.session_state.search_history = []
    st.session_state.search_results = None
    st.session_state.active_media = "Books" # Default

def init_system():
    """Initialize the recommendation system"""
    # Force re-init if recommenders are empty
    if not st.session_state.recommenders:
        try:
            with st.spinner("🎵 Initializing MelodyMatch AI..."):
                # Initialize managers for each type
                managers = {
                    "Books": "book",
                    "Music": "music",
                    "Movies": "movie"
                }
                
                for label, key in managers.items():
                    db_path = f"./db/{key}_vectors"
                    db_manager = SrvDBManager(db_path=db_path)
                    st.session_state.recommenders[label] = MediaRecommender(db_manager, media_type=key)
                
                # Bookmarks
                st.session_state.bookmark_manager = BookmarkManager()
                st.session_state.bookmarks = st.session_state.bookmark_manager.get_bookmarks()
                
            st.success("✅ System initialized successfully!")
            return True
        except Exception as e:
            st.error(f"❌ Failed to initialize system: {e}")
            return False
    return True

def display_media_card(item_data, score=None, media_type="Books"):
    """Display a professional media card"""
    if not item_data:
        return

    with st.container():
        col1, col2 = st.columns([1, 4])
        
        with col1:
            st.image(item_data.get('cover_url', 'https://via.placeholder.com/150'), use_column_width=True)
            
        with col2:
            st.markdown(f"### {item_data['title']}")
            
            # Dynamic Subtitle based on type
            subtitle = ""
            if media_type == "Books":
                subtitle = f"**By {item_data.get('author', 'Unknown')}**"
            elif media_type == "Music":
                subtitle = f"**Artist: {item_data.get('artist', 'Unknown')}**"
            elif media_type == "Movies":
                subtitle = f"**Directed by {item_data.get('director', 'Unknown')}**"
                
            st.markdown(subtitle)
            
            # Star Rating
            rating = item_data.get('rating', 0)
            full_stars = int(rating)
            has_half = rating % 1 >= 0.5
            stars = "⭐" * full_stars + ("½" if has_half else "")
            st.markdown(f"<span class='star-rating'>{stars}</span> ({rating}/5.0)", unsafe_allow_html=True)
            
            # Meta Info
            extra_info = []
            extra_info.append(item_data.get('genre', 'Unknown Genre'))
            extra_info.append(str(item_data.get('year', 'N/A')))
            
            if 'pages' in item_data:
                extra_info.append(f"{item_data['pages']} pages")
            if 'duration' in item_data:
                 val = item_data['duration']
                 if val > 60: # minutes?
                     extra_info.append(f"{val} mins")
                 else:
                     extra_info.append(f"{val}s") # seconds for songs usually?
            
            st.markdown(f"_{' • '.join(extra_info)}_")
            
            with st.expander(f"📖 Read Description"):
                st.write(item_data.get('description', 'No description available.'))
            
            if score:
                st.progress(min(score, 1.0))
                st.caption(f"Relevance: {int(min(score, 1.0)*100)}%")
            
            # Actions
            c1, c2 = st.columns(2)
            with c1:
                # Toggle Bookmark
                item_id = item_data['id']
                if st.session_state.bookmark_manager:
                    is_bookmarked = st.session_state.bookmark_manager.is_bookmarked(item_id)
                    btn_label = "❌ Remove Bookmark" if is_bookmarked else "🔖 Bookmark"
                    
                    if st.button(btn_label, key=f"bk_{item_id}_{media_type}"): # Unique key per tab context
                        if is_bookmarked:
                            st.session_state.bookmark_manager.remove_bookmark(item_id)
                        else:
                            st.session_state.bookmark_manager.add_bookmark(item_id)
                            # Index external item if it's new
                            if item_data.get('external'):
                                st.toast("Saving to local library...")
                                st.session_state.recommenders[media_type].add_external_item(item_data)
                        st.rerun()

def search_page():
    """Main Search Interface"""
    st.title("🎵 MelodyMatch AI")
    st.markdown("_Your intelligent entertainment companion_")
    
    # Media Type Selector
    media_options = ["Books", "Music", "Movies"]
    active_tab = st.selectbox("Select Media Type", media_options, index=0)
    
    # Store active tab
    st.session_state.active_media = active_tab
    recommender = st.session_state.recommenders.get(active_tab)
    
    if not recommender:
        st.error("Recommender not initialized.")
        return

    # Search Bar
    with st.container():
        col1, col2 = st.columns([3, 1])
        with col1:
            query = st.text_input(f"Search {active_tab}...", placeholder="Title, Artist, Genre...", key=f"search_{active_tab}")
        with col2:
            sort_by = st.selectbox("Sort By", ["Relevance", "Rating", "Newest"], key=f"sort_{active_tab}")
    
    # Filters
    with st.expander("⚙️ Refine Criteria", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            genres = recommender.get_available_genres()
            selected_genres = st.multiselect("Genres", genres, key=f"genre_{active_tab}")
        with c2:
            rating_range = st.slider("Min Rating", 1.0, 5.0, (1.0, 5.0), key=f"rating_{active_tab}")
        with c3:
            # Context specific Slider
            if active_tab == "Books":
                sec_label = "Page Count"
                sec_min, sec_max = 0, 1000
            elif active_tab == "Movies":
                sec_label = "Duration (mins)"
                sec_min, sec_max = 60, 240
            else: # Music
                sec_label = "Year" # Duration in seconds is hard to guess range
                sec_min, sec_max = 1960, 2024
                
            secondary_range = st.slider(sec_label, sec_min, sec_max, (sec_min, sec_max), key=f"sec_{active_tab}")
    
    # Check for filter changes
    current_filters = {
        'genres': selected_genres,
        'rating': rating_range,
        'secondary': secondary_range,
        'sort': sort_by,
        'media': active_tab
    }
    
    filters_changed = False
    if 'last_filters' not in st.session_state:
        st.session_state.last_filters = current_filters
    elif st.session_state.last_filters != current_filters:
        filters_changed = True
        st.session_state.last_filters = current_filters

    # Search Action
    do_search = st.button("🔍 Find", type="primary", key=f"btn_{active_tab}")
    
    if do_search or filters_changed or (query and st.session_state.search_results is None):
        with st.spinner(f"Curating your {active_tab} list..."):
            # 1. Local Search
            results = recommender.search(
                query=query,
                genres=selected_genres,
                rating_range=rating_range,
                secondary_range=secondary_range,
                k=15
            )
            
            is_external = False
            # 2. External Fallback (Books Only currently)
            if not results and query and active_tab == "Books":
                 with st.status("Searching global libraries (OpenLibrary)..."):
                    results = recommender.search_external(query)
                    is_external = True
            
            st.session_state.search_results = results
            st.session_state.is_external_results = is_external
            
            if query and do_search:
                st.session_state.search_history.append({"query": query, "type": active_tab, "time": time.time()})
    
    # Results Display
    st.divider()
    if st.session_state.search_results:
        if getattr(st.session_state, 'is_external_results', False):
             st.info("🌐 Item not found in local library. Showing results from external source.")
             
        st.markdown(f"**Found {len(st.session_state.search_results)} results**")
        for item, score in st.session_state.search_results:
            display_media_card(item, score, media_type=active_tab)
            st.divider()
    elif st.session_state.search_results is not None:
        st.warning("No matches found. Try a broader search.")

def dashboard_page():
    """Analytics Dashboard"""
    st.title("📊 Consumption Analytics")
    
    media_type = st.selectbox("Select Media Type", ["Books", "Music", "Movies"])
    recommender = st.session_state.recommenders.get(media_type)
    
    if not recommender:
        st.warning("Initializing...")
        return

    stats = recommender.get_genre_distribution()
    
    if not stats:
        st.info("Library is empty.")
        return

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Genre Composition")
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
            <div class="metric-label">Total {media_type}</div>
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
            st.title("MelodyMatch AI")
            page = st.radio("Navigation", ["Search", "Dashboard", "My Bookmarks"])
            
            st.divider()
            st.markdown("### About")
            st.info("Multi-modal Entertainment Recommendation System.\nPowered by SrvDB.")

        if page == "Search":
            search_page()
        elif page == "Dashboard":
            dashboard_page()
        elif page == "My Bookmarks":
            st.title("🔖 My Bookmarks")
            
            bookmarks = st.session_state.bookmark_manager.get_bookmarks()
            
            if bookmarks:
                st.write(f"You have {len(bookmarks)} bookmarks.")
                st.divider()
                
                # Group by type for display
                for item_id in bookmarks:
                    # Determine type from ID prefix
                    if item_id.startswith("book_"):
                         m_type = "Books"
                    elif item_id.startswith("music_"):
                         m_type = "Music"
                    elif item_id.startswith("movie_"):
                         m_type = "Movies"
                    elif item_id.startswith("ol_"): # External books
                         m_type = "Books"
                    else:
                         m_type = "Books" # Default fallback
                    
                    rec = st.session_state.recommenders.get(m_type)
                    if rec:
                        data = rec.get_item(item_id)
                        if data:
                            display_media_card(data, media_type=m_type)
                            st.divider()
                        else:
                            # Try fetching external if it's external ID (naive check, real app would persist external meta better)
                            st.warning(f"Item {item_id} not found in local DB.")
            else:
                st.info("No bookmarks yet.")
                if st.button("Go to Search"):
                    st.rerun()

if __name__ == "__main__":
    main()