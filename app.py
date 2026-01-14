"""
MelodyMatch AI - Music Recommendation System
Main Streamlit Application

College Project using SrvDB Vector Database
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json
import time
from datetime import datetime

# Custom modules
from srvdb_manager import SrvDBManager
from recommender import MusicRecommender
from data_processor import AudioProcessor

# Page configuration
st.set_page_config(
    page_title="MelodyMatch AI",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 10px;
        font-weight: bold;
    }
    .song-card {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    .metric-card {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.2) 0%, rgba(118, 75, 162, 0.2) 100%);
        border-radius: 10px;
        padding: 15px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'db_manager' not in st.session_state:
    st.session_state.db_manager = None
    st.session_state.recommender = None
    st.session_state.favorites = []
    st.session_state.search_history = []
    st.session_state.current_song = None

def init_system():
    """Initialize the recommendation system"""
    try:
        with st.spinner("🚀 Initializing MelodyMatch AI..."):
            st.session_state.db_manager = SrvDBManager(
                db_path="./db/music_vectors",
                dimension=128
            )
            st.session_state.recommender = MusicRecommender(
                st.session_state.db_manager
            )
        st.success("✅ System initialized successfully!")
        return True
    except Exception as e:
        st.error(f"❌ Error initializing system: {str(e)}")
        st.info("💡 Please run `python data_processor.py --build-database` first")
        return False

def display_metrics():
    """Display system metrics"""
    if st.session_state.db_manager:
        stats = st.session_state.db_manager.get_stats()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>🎵 Total Songs</h3>
                <h2>{stats['total_songs']:,}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>🎸 Genres</h3>
                <h2>{stats['total_genres']}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h3>⚡ Avg Search Time</h3>
                <h2>{stats['avg_search_time']:.2f}ms</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <h3>❤️ Favorites</h3>
                <h2>{len(st.session_state.favorites)}</h2>
            </div>
            """, unsafe_allow_html=True)

def display_song_card(song_data, similarity_score=None):
    """Display a song card with details"""
    col1, col2 = st.columns([1, 3])
    
    with col1:
        # Placeholder for album art (could be enhanced with Spotify API)
        st.markdown(f"""
        <div style="width: 100px; height: 100px; background: linear-gradient(135deg, #667eea, #764ba2); 
                    border-radius: 10px; display: flex; align-items: center; justify-content: center;">
            <span style="font-size: 40px;">🎵</span>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"### {song_data['title']}")
        st.markdown(f"**Artist:** {song_data['artist']}")
        st.markdown(f"**Genre:** {song_data['genre']}")
        
        if similarity_score is not None:
            st.progress(similarity_score)
            st.caption(f"Match Score: {similarity_score*100:.1f}%")
        
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            if st.button("▶️ Play", key=f"play_{song_data['id']}"):
                st.session_state.current_song = song_data
                st.rerun()
        
        with col_b:
            if st.button("❤️ Like", key=f"like_{song_data['id']}"):
                if song_data['id'] not in st.session_state.favorites:
                    st.session_state.favorites.append(song_data['id'])
                    st.success("Added to favorites!")
        
        with col_c:
            if st.button("🔍 Similar", key=f"similar_{song_data['id']}"):
                st.session_state.current_song = song_data
                st.rerun()

def search_page():
    """Main search page"""
    st.title("🎵 MelodyMatch AI")
    st.subheader("Discover Your Next Favorite Song")
    
    # Search section
    col1, col2 = st.columns([3, 1])
    
    with col1:
        search_query = st.text_input(
            "Search for songs, artists, or describe a mood...",
            placeholder="e.g., 'upbeat electronic music' or 'sad piano ballad'",
            key="search_input"
        )
    
    with col2:
        search_type = st.selectbox(
            "Search Type",
            ["All", "By Name", "By Genre", "By Mood"]
        )
    
    # Genre filter
    if st.session_state.recommender:
        genres = st.session_state.recommender.get_available_genres()
        selected_genres = st.multiselect(
            "Filter by Genre",
            genres,
            default=None
        )
    
    # Advanced filters
    with st.expander("⚙️ Advanced Filters"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            tempo_range = st.slider("Tempo (BPM)", 60, 200, (80, 160))
        
        with col2:
            energy_level = st.slider("Energy Level", 0.0, 1.0, (0.0, 1.0))
        
        with col3:
            valence = st.slider("Mood (Valence)", 0.0, 1.0, (0.0, 1.0))
    
    # Search button
    if st.button("🔍 Search", type="primary"):
        if search_query or selected_genres:
            with st.spinner("Searching through 25,000+ tracks..."):
                results = st.session_state.recommender.search(
                    query=search_query,
                    genres=selected_genres,
                    tempo_range=tempo_range,
                    energy_range=energy_level,
                    valence_range=valence,
                    k=20
                )
                
                if results:
                    st.success(f"Found {len(results)} matching songs!")
                    
                    # Display results
                    for i, (song_data, score) in enumerate(results):
                        with st.container():
                            st.markdown(f"#### {i+1}. Result")
                            display_song_card(song_data, score)
                            st.divider()
                else:
                    st.warning("No results found. Try different search terms!")
        else:
            st.warning("Please enter a search query or select a genre")

def recommendations_page():
    """Recommendations based on current song or favorites"""
    st.title("🎯 Personalized Recommendations")
    
    # Recommendation sources
    rec_source = st.radio(
        "Get recommendations based on:",
        ["Current Song", "My Favorites", "Genre Mix", "Upload Audio"]
    )
    
    if rec_source == "Current Song" and st.session_state.current_song:
        st.subheader(f"Songs similar to: {st.session_state.current_song['title']}")
        
        num_recs = st.slider("Number of recommendations", 5, 50, 10)
        
        if st.button("🎲 Get Recommendations"):
            with st.spinner("Finding similar songs..."):
                recs = st.session_state.recommender.get_similar_songs(
                    st.session_state.current_song['id'],
                    k=num_recs
                )
                
                for i, (song_data, score) in enumerate(recs):
                    with st.container():
                        st.markdown(f"#### {i+1}. Recommendation")
                        display_song_card(song_data, score)
                        st.divider()
    
    elif rec_source == "My Favorites":
        if not st.session_state.favorites:
            st.info("❤️ You haven't liked any songs yet! Start exploring to build your taste profile.")
        else:
            st.subheader(f"Based on your {len(st.session_state.favorites)} favorite songs")
            
            num_recs = st.slider("Number of recommendations", 5, 50, 10)
            
            if st.button("🎲 Get Recommendations"):
                with st.spinner("Analyzing your taste..."):
                    recs = st.session_state.recommender.get_recommendations_from_favorites(
                        st.session_state.favorites,
                        k=num_recs
                    )
                    
                    for i, (song_data, score) in enumerate(recs):
                        with st.container():
                            st.markdown(f"#### {i+1}. Recommendation")
                            display_song_card(song_data, score)
                            st.divider()
    
    elif rec_source == "Genre Mix":
        st.subheader("Create a custom genre mix")
        
        genres = st.session_state.recommender.get_available_genres()
        selected_genres = st.multiselect(
            "Select genres to mix",
            genres,
            max_selections=3
        )
        
        num_recs = st.slider("Number of recommendations", 5, 50, 10)
        
        if st.button("🎲 Generate Playlist") and selected_genres:
            with st.spinner("Creating your mix..."):
                recs = st.session_state.recommender.get_genre_mix(
                    selected_genres,
                    k=num_recs
                )
                
                for i, (song_data, score) in enumerate(recs):
                    with st.container():
                        st.markdown(f"#### {i+1}. Track")
                        display_song_card(song_data, score)
                        st.divider()
    
    elif rec_source == "Upload Audio":
        st.subheader("Upload your own audio file")
        st.info("🎵 Upload a song and we'll find similar tracks!")
        
        uploaded_file = st.file_uploader(
            "Choose an audio file",
            type=['mp3', 'wav', 'ogg', 'flac']
        )
        
        if uploaded_file is not None:
            st.audio(uploaded_file)
            
            if st.button("🔍 Find Similar Songs"):
                with st.spinner("Analyzing audio and searching..."):
                    # Process uploaded audio
                    processor = AudioProcessor()
                    features = processor.extract_features_from_file(uploaded_file)
                    
                    # Get recommendations
                    recs = st.session_state.recommender.get_similar_by_features(
                        features,
                        k=10
                    )
                    
                    st.success(f"Found {len(recs)} similar songs!")
                    
                    for i, (song_data, score) in enumerate(recs):
                        with st.container():
                            st.markdown(f"#### {i+1}. Similar Track")
                            display_song_card(song_data, score)
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

def settings_page():
    """Settings and database management"""
    st.title("⚙️ Settings & Database")
    
    # Database stats
    st.subheader("📊 Database Statistics")
    if st.session_state.db_manager:
        stats = st.session_state.db_manager.get_detailed_stats()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Vectors", f"{stats['vector_count']:,}")
            st.metric("Memory Usage", f"{stats['memory_mb']:.2f} MB")
        
        with col2:
            st.metric("Search Mode", stats['mode'])
            st.metric("Dimension", stats['dimension'])
        
        with col3:
            st.metric("Compression", f"{stats['compression_ratio']:.1f}x")
            st.metric("Avg Query Time", f"{stats['avg_query_ms']:.2f} ms")
    
    st.divider()
    
    # Database management
    st.subheader("🔧 Database Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Rebuild Database"):
            st.warning("This will reprocess all audio files. Continue?")
            if st.button("✅ Yes, Rebuild"):
                with st.spinner("Rebuilding database..."):
                    # Rebuild logic here
                    pass
    
    with col2:
        if st.button("💾 Export Data"):
            # Export functionality
            st.download_button(
                label="Download as CSV",
                data="",  # CSV data
                file_name="music_data.csv",
                mime="text/csv"
            )
    
    st.divider()
    
    # Performance tuning
    st.subheader("⚡ Performance Tuning")
    
    search_mode = st.selectbox(
        "Vector Search Mode",
        ["HNSW (Fast)", "Flat (Accurate)", "Auto"],
        help="HNSW for speed, Flat for accuracy"
    )
    
    cache_size = st.slider("Cache Size (MB)", 50, 500, 100)
    
    if st.button("Apply Settings"):
        st.success("Settings applied!")

def main():
    """Main application"""
    
    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/150x150/667eea/ffffff?text=🎵", use_container_width=True)
        st.title("MelodyMatch AI")
        st.caption("Powered by SrvDB")
        
        st.divider()
        
        # Navigation
        page = st.radio(
            "Navigation",
            ["🔍 Search", "🎯 Recommendations", "📊 Analytics", "⚙️ Settings"]
        )
        
        st.divider()
        
        # Quick stats
        st.subheader("Quick Stats")
        if st.session_state.db_manager:
            display_metrics()
        
        st.divider()
        
        # About
        with st.expander("ℹ️ About"):
            st.markdown("""
            **MelodyMatch AI** is a music recommendation system using:
            - 🎵 25,000+ songs from FMA dataset
            - 🚀 SrvDB vector database
            - 🤖 Advanced audio analysis
            - 💨 Sub-100ms recommendations
            
            Built for educational purposes.
            """)
    
    # Initialize system
    if st.session_state.db_manager is None:
        if not init_system():
            return
    
    # Route to pages
    if page == "🔍 Search":
        search_page()
    elif page == "🎯 Recommendations":
        recommendations_page()
    elif page == "📊 Analytics":
        analytics_page()
    elif page == "⚙️ Settings":
        settings_page()

if __name__ == "__main__":
    main()