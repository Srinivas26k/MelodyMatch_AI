"""
Music Recommendation Engine
Implements various recommendation strategies using SrvDB
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import Counter
import random


class MusicRecommender:
    """Music recommendation engine powered by SrvDB"""
    
    def __init__(self, db_manager):
        """
        Initialize recommender
        
        Args:
            db_manager: SrvDBManager instance
        """
        self.db = db_manager
        
    def get_similar_songs(self, 
                         song_id: str, 
                         k: int = 10) -> List[Tuple[Dict, float]]:
        """
        Get songs similar to a given song
        
        Args:
            song_id: Source song ID
            k: Number of recommendations
        
        Returns:
            List of (song_metadata, similarity_score) tuples
        """
        results = self.db.search_by_id(song_id, k=k)
        return [(meta, score) for _, score, meta in results]
    
    def search(self,
               query: str = None,
               genres: List[str] = None,
               tempo_range: Tuple[float, float] = None,
               energy_range: Tuple[float, float] = None,
               valence_range: Tuple[float, float] = None,
               k: int = 20) -> List[Tuple[Dict, float]]:
        """
        Multi-criteria search
        
        Args:
            query: Text search query (song name, artist, mood)
            genres: List of genres to filter
            tempo_range: (min, max) tempo in BPM
            energy_range: (min, max) energy level (0-1)
            valence_range: (min, max) valence/mood (0-1)
            k: Number of results
        
        Returns:
            List of (song_metadata, relevance_score) tuples
        """
        # Start with all songs or genre-filtered
        candidates = []
        
        if genres:
            # Get songs from selected genres
            for genre in genres:
                genre_songs = self.db.get_by_genre(genre, k=1000)
                candidates.extend(genre_songs)
        else:
            # Get all songs from cache
            candidates = list(self.db.metadata_cache.values())
        
        # Text search filtering
        if query:
            query_lower = query.lower()
            candidates = [
                song for song in candidates
                if query_lower in song.get('title', '').lower() or
                   query_lower in song.get('artist', '').lower() or
                   query_lower in song.get('genre', '').lower()
            ]
        
        # Apply filters (if song has those fields)
        if tempo_range:
            candidates = [
                song for song in candidates
                if 'tempo' in song and tempo_range[0] <= song['tempo'] <= tempo_range[1]
            ]
        
        if energy_range:
            candidates = [
                song for song in candidates
                if 'energy' in song and energy_range[0] <= song['energy'] <= energy_range[1]
            ]
        
        if valence_range:
            candidates = [
                song for song in candidates
                if 'valence' in song and valence_range[0] <= song['valence'] <= valence_range[1]
            ]
        
        # Score candidates
        scored_candidates = []
        for song in candidates[:k*2]:  # Consider more for ranking
            # Base score
            score = 0.8  # Default relevance
            
            # Boost exact matches
            if query:
                if query.lower() in song.get('title', '').lower():
                    score += 0.15
                if query.lower() in song.get('artist', '').lower():
                    score += 0.1
            
            scored_candidates.append((song, score))
        
        # Sort by score and return top k
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        return scored_candidates[:k]
    
    def get_recommendations_from_favorites(self,
                                          favorite_ids: List[str],
                                          k: int = 10,
                                          diversity: float = 0.3) -> List[Tuple[Dict, float]]:
        """
        Get recommendations based on multiple favorite songs
        
        Args:
            favorite_ids: List of favorite song IDs
            k: Number of recommendations
            diversity: 0-1, higher = more diverse (less similar to favorites)
        
        Returns:
            List of (song_metadata, relevance_score) tuples
        """
        if not favorite_ids:
            return []
        
        # Get embeddings for favorites
        favorite_embeddings = []
        for song_id in favorite_ids:
            metadata = self.db._get_metadata(song_id)
            if metadata and 'embedding' in metadata:
                favorite_embeddings.append(np.array(metadata['embedding']))
        
        if not favorite_embeddings:
            return []
        
        # Create centroid (average taste profile)
        centroid = np.mean(favorite_embeddings, axis=0)
        
        # Search using centroid
        results = self.db.search_similar(
            centroid,
            k=k * 3,  # Get more for diversity filtering
            exclude_ids=favorite_ids
        )
        
        # Apply diversity filter
        if diversity > 0:
            results = self._apply_diversity_filter(results, diversity, k)
        
        return [(meta, score) for _, score, meta in results[:k]]
    
    def get_genre_mix(self,
                     genres: List[str],
                     k: int = 30,
                     balance: str = 'equal') -> List[Tuple[Dict, float]]:
        """
        Create a mixed playlist from multiple genres
        
        Args:
            genres: List of genre names
            k: Total number of songs
            balance: 'equal' or 'proportional'
        
        Returns:
            List of (song_metadata, score) tuples
        """
        if not genres:
            return []
        
        # Determine songs per genre
        if balance == 'equal':
            songs_per_genre = k // len(genres)
            remainder = k % len(genres)
        else:
            # Proportional based on genre size
            genre_counts = {g: len(self.db.get_by_genre(g)) for g in genres}
            total = sum(genre_counts.values())
            songs_per_genre = {
                g: int((count / total) * k)
                for g, count in genre_counts.items()
            }
        
        # Collect songs
        playlist = []
        for i, genre in enumerate(genres):
            count = songs_per_genre if balance == 'equal' else songs_per_genre[genre]
            if balance == 'equal' and i < remainder:
                count += 1
            
            genre_songs = self.db.get_by_genre(genre, k=count * 2)
            
            # Random sample with slight bias to "better" songs
            # (assuming songs are somewhat ordered by popularity)
            sampled = random.sample(
                genre_songs[:min(len(genre_songs), count * 2)],
                min(count, len(genre_songs))
            )
            
            for song in sampled:
                playlist.append((song, 0.8))  # Neutral score
        
        # Shuffle for variety
        random.shuffle(playlist)
        
        return playlist[:k]
    
    def get_similar_by_features(self,
                               features: np.ndarray,
                               k: int = 10) -> List[Tuple[Dict, float]]:
        """
        Get similar songs based on audio features
        (For uploaded audio files)
        
        Args:
            features: Audio feature vector
            k: Number of results
        
        Returns:
            List of (song_metadata, similarity_score) tuples
        """
        results = self.db.search_similar(features, k=k)
        return [(meta, score) for _, score, meta in results]
    
    def get_mood_based_recommendations(self,
                                      mood: str,
                                      k: int = 20) -> List[Tuple[Dict, float]]:
        """
        Get songs matching a mood
        
        Args:
            mood: Mood keyword (happy, sad, energetic, calm, etc.)
            k: Number of results
        
        Returns:
            List of (song_metadata, relevance_score) tuples
        """
        # Map moods to genres and characteristics
        mood_mappings = {
            'happy': {'genres': ['Pop', 'Electronic'], 'valence': (0.6, 1.0)},
            'sad': {'genres': ['Blues', 'Folk'], 'valence': (0.0, 0.4)},
            'energetic': {'genres': ['Rock', 'Electronic'], 'energy': (0.7, 1.0)},
            'calm': {'genres': ['Classical', 'Folk'], 'energy': (0.0, 0.4)},
            'party': {'genres': ['Electronic', 'Hip-Hop'], 'energy': (0.7, 1.0)},
            'focus': {'genres': ['Classical', 'Instrumental'], 'energy': (0.3, 0.6)},
        }
        
        mood_lower = mood.lower()
        if mood_lower in mood_mappings:
            mapping = mood_mappings[mood_lower]
            return self.search(
                genres=mapping.get('genres'),
                energy_range=mapping.get('energy'),
                valence_range=mapping.get('valence'),
                k=k
            )
        else:
            # Fallback: text search
            return self.search(query=mood, k=k)
    
    def get_available_genres(self) -> List[str]:
        """Get list of all available genres"""
        return self.db.get_all_genres()
    
    def get_genre_distribution(self) -> Dict[str, int]:
        """Get count of songs per genre"""
        genre_counts = Counter()
        
        for metadata in self.db.metadata_cache.values():
            genre = metadata.get('genre', 'Unknown')
            genre_counts[genre] += 1
        
        return dict(genre_counts)
    
    def analyze_favorites(self, favorite_ids: List[str]) -> Dict:
        """
        Analyze user's favorite songs
        
        Args:
            favorite_ids: List of favorite song IDs
        
        Returns:
            Analysis dictionary with genre distribution, avg features, etc.
        """
        if not favorite_ids:
            return {}
        
        # Collect metadata
        favorites_meta = [
            self.db._get_metadata(song_id)
            for song_id in favorite_ids
        ]
        favorites_meta = [m for m in favorites_meta if m]
        
        # Genre analysis
        genres = Counter(m.get('genre', 'Unknown') for m in favorites_meta)
        
        # Average features (if available)
        energy_vals = [m.get('energy', 0.5) for m in favorites_meta if 'energy' in m]
        valence_vals = [m.get('valence', 0.5) for m in favorites_meta if 'valence' in m]
        danceability_vals = [m.get('danceability', 0.5) for m in favorites_meta if 'danceability' in m]
        acousticness_vals = [m.get('acousticness', 0.5) for m in favorites_meta if 'acousticness' in m]
        
        return {
            'total_favorites': len(favorites_meta),
            'genres': dict(genres),
            'avg_energy': np.mean(energy_vals) if energy_vals else 0.5,
            'avg_valence': np.mean(valence_vals) if valence_vals else 0.5,
            'avg_danceability': np.mean(danceability_vals) if danceability_vals else 0.5,
            'avg_acousticness': np.mean(acousticness_vals) if acousticness_vals else 0.5,
            'top_genre': genres.most_common(1)[0][0] if genres else 'Unknown'
        }
    
    def create_playlist(self,
                       name: str,
                       seed_songs: List[str] = None,
                       genres: List[str] = None,
                       length: int = 30,
                       style: str = 'balanced') -> List[Dict]:
        """
        Create a curated playlist
        
        Args:
            name: Playlist name
            seed_songs: Optional seed song IDs
            genres: Optional genre filters
            length: Number of songs
            style: 'balanced', 'similar', or 'diverse'
        
        Returns:
            List of song metadata
        """
        playlist = []
        
        if seed_songs:
            # Start with recommendations from seeds
            if style == 'similar':
                # Very similar songs
                recs = self.get_recommendations_from_favorites(
                    seed_songs,
                    k=length,
                    diversity=0.1
                )
            elif style == 'diverse':
                # More variety
                recs = self.get_recommendations_from_favorites(
                    seed_songs,
                    k=length,
                    diversity=0.7
                )
            else:  # balanced
                recs = self.get_recommendations_from_favorites(
                    seed_songs,
                    k=length,
                    diversity=0.4
                )
            
            playlist = [song for song, _ in recs]
        
        elif genres:
            # Genre-based playlist
            recs = self.get_genre_mix(genres, k=length)
            playlist = [song for song, _ in recs]
        
        else:
            # Random diverse playlist
            all_genres = self.get_available_genres()
            selected_genres = random.sample(all_genres, min(3, len(all_genres)))
            recs = self.get_genre_mix(selected_genres, k=length)
            playlist = [song for song, _ in recs]
        
        # Add playlist metadata
        for song in playlist:
            song['playlist'] = name
        
        return playlist
    
    def _apply_diversity_filter(self,
                               results: List[Tuple[str, float, Dict]],
                               diversity: float,
                               k: int) -> List[Tuple[str, float, Dict]]:
        """
        Apply diversity filtering to results
        
        Args:
            results: Search results
            diversity: Diversity factor (0-1)
            k: Target number of results
        
        Returns:
            Filtered results
        """
        if not results or diversity == 0:
            return results
        
        # Maximal Marginal Relevance (MMR) approach
        selected = []
        candidates = list(results)
        
        # Start with top result
        selected.append(candidates.pop(0))
        
        while len(selected) < k and candidates:
            best_score = -float('inf')
            best_idx = 0
            
            for idx, candidate in enumerate(candidates):
                # Relevance score
                relevance = candidate[1]
                
                # Diversity penalty (similarity to already selected)
                diversity_penalty = 0
                for selected_item in selected:
                    # Compare genre, artist, etc.
                    if candidate[2].get('genre') == selected_item[2].get('genre'):
                        diversity_penalty += 0.2
                    if candidate[2].get('artist') == selected_item[2].get('artist'):
                        diversity_penalty += 0.3
                
                # MMR score
                mmr_score = (1 - diversity) * relevance - diversity * diversity_penalty
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx
            
            selected.append(candidates.pop(best_idx))
        
        return selected


# Example usage
if __name__ == "__main__":
    from srvdb_manager import SrvDBManager
    
    # Initialize
    db_manager = SrvDBManager()
    recommender = MusicRecommender(db_manager)
    
    # Get available genres
    genres = recommender.get_available_genres()
    print(f"Available genres: {genres}")
    
    # Create a mixed playlist
    playlist = recommender.get_genre_mix(['Rock', 'Pop'], k=10)
    print(f"\nMixed playlist ({len(playlist)} songs):")
    for i, (song, score) in enumerate(playlist):
        print(f"{i+1}. {song['title']} - {song['artist']}")