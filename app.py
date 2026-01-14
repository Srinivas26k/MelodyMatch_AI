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
from bookmark_manager import BookmarkManager
import time

# ... (Configuration stays same)

# Application State
if 'recommender' not in st.session_state:
    st.session_state.recommender = None
    st.session_state.bookmark_manager = None 
    st.session_state.bookmarks = []
    st.session_state.search_history = []
    st.session_state.search_results = None
    st.session_state.is_external_results = False

def init_system():
    """Initialize the recommendation system"""
    # Force re-init if recommender is stale (missing new methods)
    if (st.session_state.recommender is None or 
        not hasattr(st.session_state.recommender, 'search_external')):
        try:
            with st.spinner("📚 Initializing PageTurner AI..."):
                # Database
                db_manager = SrvDBManager(db_path="./db/book_vectors")
                
                if db_manager.db.count() == 0:
                     # Warn or attempt to regenerate? For now just load
                     pass
                     
                st.session_state.recommender = BookRecommender(db_manager)
                
                # Bookmarks
                st.session_state.bookmark_manager = BookmarkManager()
                # Load existing bookmarks
                st.session_state.bookmarks = st.session_state.bookmark_manager.get_bookmarks()
                
            st.success("✅ System initialized successfully!")
            return True
        except Exception as e:
            st.error(f"❌ Failed to initialize system: {e}")
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
                st.progress(min(score, 1.0))
                st.caption(f"Relevance: {int(min(score, 1.0)*100)}%")
            
            # Actions
            c1, c2 = st.columns(2)
            with c1:
                # Toggle Bookmark
                book_id = book_data['id']
                if st.session_state.bookmark_manager:
                    is_bookmarked = st.session_state.bookmark_manager.is_bookmarked(book_id)
                    btn_label = "❌ Remove Bookmark" if is_bookmarked else "🔖 Bookmark"
                    
                    if st.button(btn_label, key=f"bk_{book_id}"):
                        if is_bookmarked:
                            st.session_state.bookmark_manager.remove_bookmark(book_id)
                        else:
                            st.session_state.bookmark_manager.add_bookmark(book_id)
                            # Index external book if it's new
                            if book_data.get('external'):
                                st.toast("Saving to local library...")
                                st.session_state.recommender.add_external_book(book_data)
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
            rating_range = st.slider("Min Rating", 1.0, 5.0, (1.0, 5.0))
        with c3:
            pages_range = st.slider("Page Count", 0, 1000, (0, 1000))
    
    # Check for filter changes
    current_filters = {
        'genres': selected_genres,
        'rating': rating_range,
        'pages': pages_range,
        'sort': sort_by
    }
    
    filters_changed = False
    if 'last_filters' not in st.session_state:
        st.session_state.last_filters = current_filters
    elif st.session_state.last_filters != current_filters:
        filters_changed = True
        st.session_state.last_filters = current_filters

    # Search Action
    # Trigger if: Button clicked OR Filters changed OR (Query present and no results)
    do_search = st.button("🔍 Find Books", type="primary")
    
    if do_search or filters_changed or (query and st.session_state.search_results is None):
        with st.spinner("Curating your reading list..."):
            # 1. Local Search
            results = st.session_state.recommender.search(
                query=query,
                genres=selected_genres,
                rating_range=rating_range,
                pages_range=pages_range,
                k=15
            )
            
            is_external = False
            # 2. External Fallback (only if query is distinct)
            if not results and query:
                 with st.status("Searching global libraries (OpenLibrary)..."):
                    results = st.session_state.recommender.search_external(query)
                    is_external = True
            
            st.session_state.search_results = results
            st.session_state.is_external_results = is_external
            
            if query and do_search: # Only log history on explicit search
                st.session_state.search_history.append({"query": query, "time": time.time()})
    
    # Results Display
    st.divider()
    if st.session_state.search_results:
        if getattr(st.session_state, 'is_external_results', False):
             st.info("🌐 Book not found in local library. Showing results from OpenLibrary.")
             
        st.markdown(f"**Found {len(st.session_state.search_results)} books matching your criteria**")
        for book, score in st.session_state.search_results:
            display_book_card(book, score)
            st.divider()
    elif st.session_state.search_results is not None:
        st.warning("No books found. Try a broader search.")

def dashboard_page():
    """Analytics Dashboard"""
    st.title("📊 Literary Analytics")
    
    stats = st.session_state.recommender.get_genre_distribution()
    
    if not stats:
        st.info("Library is empty. Generate data or bookmark books to see stats.")
        return

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
            
            # Refresh bookmarks from manager
            bookmarks = st.session_state.bookmark_manager.get_bookmarks()
            
            if bookmarks:
                st.write(f"You have {len(bookmarks)} bookmarked books.")
                st.divider()
                
                for book_id in bookmarks:
                    # Get full metadata
                    book_data = st.session_state.recommender.get_book(book_id)
                    
                    if book_data:
                        display_book_card(book_data)
                        st.divider()
                    else:
                        # Fallback if book not in local DB (shouldn't happen with lazy indexing, but good for safety)
                        st.warning(f"Metadata missing for ID: {book_id}")
            else:
                st.info("You haven't bookmarked any books yet. Go to Search to find some!")
                if st.button("Go to Search"):
                    st.rerun()

if __name__ == "__main__":
    main()