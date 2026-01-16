"""
Media Recommendation Engine
Implements various recommendation strategies using SrvDB for Books, Music, and Movies.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import Counter
import random

class MediaRecommender:
    """Generic media recommendation engine powered by SrvDB"""
    
    def __init__(self, db_manager, media_type: str = "book"):
        """
        Initialize recommender
        
        Args:
            db_manager: SrvDBManager instance
            media_type: Type of media ('book', 'music', 'movie')
        """
        self.db = db_manager
        self.media_type = media_type
        
    def get_similar_items(self, 
                         item_id: str, 
                         k: int = 10) -> List[Tuple[Dict, float]]:
        """Get items similar to a given item"""
        results = self.db.search_by_id(item_id, k=k)
        return [(meta, score) for _, score, meta in results]
    
    def search(self,
               query: str = None,
               genres: List[str] = None,
               rating_range: Tuple[float, float] = None,
               secondary_range: Tuple[int, int] = None, # pages or duration or year
               k: int = 20) -> List[Tuple[Dict, float]]:
        """
        Multi-criteria search
        
        Args:
            query: Text search query
            genres: List of genres to filter
            rating_range: (min, max) rating
            secondary_range: (min, max) for pages/duration/year depending on type
            k: Number of results
        
        Returns:
            List of (metadata, relevance_score) tuples
        """
        # Start with all items or genre-filtered
        candidates = []
        
        if genres:
            # Get items from selected genres
            for genre in genres:
                genre_items = self.db.get_by_genre(genre, k=1000)
                candidates.extend(genre_items)
        else:
            # Get all items from cache
            candidates = list(self.db.metadata_cache.values())
        
        # Text search filtering
        if query:
            query_lower = query.lower()
            candidates = [
                item for item in candidates
                if query_lower in item.get('title', '').lower() or
                   query_lower in item.get('author', '').lower() or
                   query_lower in item.get('artist', '').lower() or
                   query_lower in item.get('director', '').lower() or
                   query_lower in item.get('genre', '').lower()
            ]
        
        # Apply filters
        if rating_range:
            candidates = [
                item for item in candidates
                if 'rating' in item and rating_range[0] <= item['rating'] <= rating_range[1]
            ]
            
        if secondary_range:
            key = 'pages' if self.media_type == 'book' else 'year' # Simplified for now, can be duration
            candidates = [
                item for item in candidates
                if key in item and secondary_range[0] <= item[key] <= secondary_range[1]
            ]
        
        # Score candidates
        scored_candidates = []
        for item in candidates[:k*3]:  # Consider more for ranking
            # Base score
            score = 0.8
            
            # Boost exact matches
            if query:
                q = query.lower()
                if q in item.get('title', '').lower():
                    score += 0.2
                if q in item.get('author', '').lower() or q in item.get('artist', '').lower() or q in item.get('director', '').lower():
                    score += 0.15
            
            # Boost high ratings
            if 'rating' in item:
                score += (item['rating'] - 3.0) * 0.05
            
            scored_candidates.append((item, score))
        
        # Sort by score
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        return scored_candidates[:k]
    
    def get_recommendations_from_favorites(self,
                                          favorite_ids: List[str],
                                          k: int = 10) -> List[Tuple[Dict, float]]:
        """Get recommendations based on favorite items"""
        if not favorite_ids:
            return []
        
        # Get embeddings for favorites
        favorite_embeddings = []
        valid_ids = []
        for item_id in favorite_ids:
            # Validate metadata exists in this DB instance
            metadata = self.db._get_metadata(item_id)
            if metadata and 'embedding' in metadata:
                favorite_embeddings.append(np.array(metadata['embedding']))
                valid_ids.append(item_id)
        
        if not favorite_embeddings:
            return []
        
        # Create centroid
        centroid = np.mean(favorite_embeddings, axis=0)
        
        # Search using centroid
        results = self.db.search_similar(
            centroid,
            k=k,
            exclude_ids=valid_ids
        )
        
        return [(meta, score) for _, score, meta in results]
    
    def get_available_genres(self) -> List[str]:
        """Get list of all available genres"""
        return self.db.get_all_genres()
    
    def get_genre_distribution(self) -> Dict[str, int]:
        """Get count of items per genre"""
        genre_counts = Counter()
        for metadata in self.db.metadata_cache.values():
            genre = metadata.get('genre', 'Unknown')
            genre_counts[genre] += 1
        return dict(genre_counts)

    def search_external(self, query: str, k: int = 5) -> List[Tuple[Dict, float]]:
        """
        Search for external items (OpenLibrary for Books, placeholder for others)
        """
        if self.media_type != 'book':
            return [] # Only books supported currently for external
            
        import requests
        
        results = []
        try:
            # OpenLibrary Search API
            url = f"https://openlibrary.org/search.json?q={query}&limit={k}"
            response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                
                for doc in data.get('docs', []):
                    # Extract fields
                    title = doc.get('title', 'Unknown Title')
                    author = doc.get('author_name', ['Unknown Author'])[0]
                    # Genre/Subjects
                    subjects = doc.get('subject', [])
                    genre = subjects[0] if subjects else "General"
                    
                    # Year
                    year = doc.get('first_publish_year', None)
                    
                    # Pages (estimate if missing)
                    pages = doc.get('number_of_pages_median', 300)
                    
                    # Cover
                    cover_i = doc.get('cover_i')
                    if cover_i:
                        cover_url = f"https://covers.openlibrary.org/b/id/{cover_i}-M.jpg"
                    else:
                        cover_url = "https://via.placeholder.com/150x200?text=No+Cover"
                    
                    # Create metadata dict matching our internal format
                    item_data = {
                        'id': f"ol_{doc.get('key', '').split('/')[-1]}", # Use OL key as unique ID
                        'title': title,
                        'author': author,
                        'genre': genre,
                        'rating': 4.0, # Default rating for external items
                        'pages': pages,
                        'year': year,
                        'description': f"Retrieved from OpenLibrary. First published in {year}.",
                        'cover_url': cover_url,
                        'external': True
                    }
                    
                    # Relevance score (from API order)
                    results.append((item_data, 0.9))
        
        except Exception as e:
            print(f"External API Error: {e}")
            
        return results

    def get_item(self, item_id: str) -> Optional[Dict]:
        """Get metadata for a single item by ID"""
        # First check cache
        if item_id in self.db.metadata_cache:
            return self.db.metadata_cache[item_id]
        return None

    def add_external_item(self, item_data: Dict) -> bool:
        """
        Add an external item to the local database (Lazy Indexing)
        Generates a synthetic embedding based on genre/metadata
        """
        # Check if already exists
        if item_data['id'] in self.db.metadata_cache:
            return True
            
        try:
            # Generate synthetic embedding (128D) similar to generator
            genre = item_data.get('genre', 'General')
            vector = np.random.randn(128)
            
            # Simple genre bias
            genre_hash = sum(ord(c) for c in genre)
            vector[genre_hash % 10] += 1.5 
            vector[(genre_hash + 1) % 10] += 0.5
            
            # Normalize
            vector = vector / np.linalg.norm(vector)
            
            # Ensure ID is set
            if 'id' not in item_data:
                return False
                
            # Add to DB
            self.db.add_books([vector], [item_data]) # This method name in manager might need alias but works
            return True
        except Exception as e:
            print(f"Error adding external item: {e}")
            return False