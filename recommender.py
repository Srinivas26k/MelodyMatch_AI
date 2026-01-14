"""
Book Recommendation Engine
Implements various recommendation strategies using SrvDB
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import Counter
import random


class BookRecommender:
    """Book recommendation engine powered by SrvDB"""
    
    def __init__(self, db_manager):
        """
        Initialize recommender
        
        Args:
            db_manager: SrvDBManager instance
        """
        self.db = db_manager
        
    def get_similar_books(self, 
                         book_id: str, 
                         k: int = 10) -> List[Tuple[Dict, float]]:
        """Get books similar to a given book"""
        results = self.db.search_by_id(book_id, k=k)
        return [(meta, score) for _, score, meta in results]
    
    def search(self,
               query: str = None,
               genres: List[str] = None,
               rating_range: Tuple[float, float] = None,
               pages_range: Tuple[int, int] = None,
               k: int = 20) -> List[Tuple[Dict, float]]:
        """
        Multi-criteria book search
        
        Args:
            query: Text search query (title, author, genre)
            genres: List of genres to filter
            rating_range: (min, max) rating (0-5)
            pages_range: (min, max) page count
            k: Number of results
        
        Returns:
            List of (book_metadata, relevance_score) tuples
        """
        # Start with all books or genre-filtered
        candidates = []
        
        if genres:
            # Get books from selected genres
            for genre in genres:
                genre_books = self.db.get_by_genre(genre, k=1000)
                candidates.extend(genre_books)
        else:
            # Get all books from cache
            candidates = list(self.db.metadata_cache.values())
        
        # Text search filtering
        if query:
            query_lower = query.lower()
            candidates = [
                book for book in candidates
                if query_lower in book.get('title', '').lower() or
                   query_lower in book.get('author', '').lower() or
                   query_lower in book.get('genre', '').lower()
            ]
        
        # Apply filters
        if rating_range:
            candidates = [
                book for book in candidates
                if 'rating' in book and rating_range[0] <= book['rating'] <= rating_range[1]
            ]
            
        if pages_range:
            candidates = [
                book for book in candidates
                if 'pages' in book and pages_range[0] <= book['pages'] <= pages_range[1]
            ]
        
        # Score candidates
        scored_candidates = []
        for book in candidates[:k*3]:  # Consider more for ranking
            # Base score
            score = 0.8
            
            # Boost exact matches
            if query:
                if query.lower() in book.get('title', '').lower():
                    score += 0.2
                if query.lower() in book.get('author', '').lower():
                    score += 0.15
            
            # Boost high ratings
            if 'rating' in book:
                score += (book['rating'] - 3.0) * 0.05
            
            scored_candidates.append((book, score))
        
        # Sort by score
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        return scored_candidates[:k]
    
    def get_recommendations_from_favorites(self,
                                          favorite_ids: List[str],
                                          k: int = 10) -> List[Tuple[Dict, float]]:
        """Get recommendations based on favorite books"""
        if not favorite_ids:
            return []
        
        # Get embeddings for favorites
        favorite_embeddings = []
        for book_id in favorite_ids:
            metadata = self.db._get_metadata(book_id)
            if metadata and 'embedding' in metadata:
                favorite_embeddings.append(np.array(metadata['embedding']))
        
        if not favorite_embeddings:
            return []
        
        # Create centroid
        centroid = np.mean(favorite_embeddings, axis=0)
        
        # Search using centroid
        results = self.db.search_similar(
            centroid,
            k=k,
            exclude_ids=favorite_ids
        )
        
        return [(meta, score) for _, score, meta in results]
    
    def get_available_genres(self) -> List[str]:
        """Get list of all available genres"""
        return self.db.get_all_genres()
    
    def get_genre_distribution(self) -> Dict[str, int]:
        """Get count of books per genre"""
        genre_counts = Counter()
        for metadata in self.db.metadata_cache.values():
            genre = metadata.get('genre', 'Unknown')
            genre_counts[genre] += 1
        return dict(genre_counts)


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