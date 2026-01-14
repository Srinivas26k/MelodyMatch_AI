"""
SrvDB Manager - Interface for vector database operations
Handles all interactions with SrvDB for music recommendation system
"""

import srvdb
import numpy as np
import json
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional


class SrvDBManager:
    """Manages SrvDB vector database for music embeddings"""
    
    def __init__(self, db_path: str = "./db/music_vectors", dimension: int = 128):
        """
        Initialize SrvDB Manager
        
        Args:
            db_path: Path to database directory
            dimension: Vector dimension (128 for audio features)
        """
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.dimension = dimension
        
        # Initialize database with HNSW for fast search
        print(f"🚀 Initializing SrvDB at {db_path}")
        self.db = srvdb.SrvDBPython.new_hnsw(
            path=str(self.db_path),
            dimension=dimension,
            m=16,  # Good balance of speed and accuracy
            ef_construction=200,
            ef_search=50
        )
        
        # Metadata cache for quick access
        self.metadata_cache = {}
        self.query_times = []
        
        print(f"✅ Database initialized with {self.db.count()} vectors")
    
    def add_songs(self, 
                  embeddings: List[np.ndarray], 
                  metadata: List[Dict]) -> List[str]:
        """
        Add songs to the database
        
        Args:
            embeddings: List of audio feature vectors
            metadata: List of song metadata dictionaries
        
        Returns:
            List of song IDs
        """
        if len(embeddings) != len(metadata):
            raise ValueError("Embeddings and metadata must have same length")
        
        # Generate IDs
        ids = [meta['id'] for meta in metadata]
        
        # Convert embeddings to list format
        embeddings_list = [emb.tolist() for emb in embeddings]
        
        # Convert metadata to JSON strings
        metadata_json = [json.dumps(meta) for meta in metadata]
        
        # Batch insert
        print(f"📝 Adding {len(ids)} songs to database...")
        start_time = time.time()
        
        self.db.add(
            ids=ids,
            embeddings=embeddings_list,
            metadatas=metadata_json
        )
        
        elapsed = time.time() - start_time
        print(f"✅ Added {len(ids)} songs in {elapsed:.2f}s ({len(ids)/elapsed:.0f} songs/s)")
        
        # Update metadata cache
        for song_id, meta in zip(ids, metadata):
            self.metadata_cache[song_id] = meta
        
        # Persist to disk
        self.db.persist()
        
        return ids
    
    def search_similar(self, 
                       query_embedding: np.ndarray, 
                       k: int = 10,
                       exclude_ids: Optional[List[str]] = None) -> List[Tuple[str, float, Dict]]:
        """
        Search for similar songs
        
        Args:
            query_embedding: Query vector
            k: Number of results
            exclude_ids: Song IDs to exclude from results
        
        Returns:
            List of (song_id, similarity_score, metadata) tuples
        """
        start_time = time.time()
        
        # Perform search
        results = self.db.search(
            query=query_embedding.tolist(),
            k=k * 2 if exclude_ids else k  # Get extra if filtering
        )
        
        # Record query time
        query_time = (time.time() - start_time) * 1000  # ms
        self.query_times.append(query_time)
        
        # Filter and enrich results
        enriched_results = []
        for song_id, score in results:
            # Skip excluded IDs
            if exclude_ids and song_id in exclude_ids:
                continue
            
            # Get metadata
            metadata = self._get_metadata(song_id)
            enriched_results.append((song_id, score, metadata))
            
            if len(enriched_results) >= k:
                break
        
        return enriched_results
    
    def search_by_id(self, song_id: str, k: int = 10) -> List[Tuple[str, float, Dict]]:
        """
        Find similar songs to a given song ID
        
        Args:
            song_id: Source song ID
            k: Number of results
        
        Returns:
            List of (song_id, similarity_score, metadata) tuples
        """
        # Get the song's embedding from metadata
        metadata = self._get_metadata(song_id)
        if not metadata or 'embedding' not in metadata:
            raise ValueError(f"Song {song_id} not found or has no embedding")
        
        embedding = np.array(metadata['embedding'])
        
        # Search, excluding the source song
        return self.search_similar(embedding, k=k, exclude_ids=[song_id])
    
    def batch_search(self, 
                     query_embeddings: List[np.ndarray], 
                     k: int = 10) -> List[List[Tuple[str, float, Dict]]]:
        """
        Batch search for multiple queries
        
        Args:
            query_embeddings: List of query vectors
            k: Number of results per query
        
        Returns:
            List of result lists
        """
        embeddings_list = [emb.tolist() for emb in query_embeddings]
        
        start_time = time.time()
        batch_results = self.db.search_batch(queries=embeddings_list, k=k)
        query_time = (time.time() - start_time) * 1000  # ms
        
        self.query_times.append(query_time / len(query_embeddings))
        
        # Enrich all results
        enriched_batch = []
        for results in batch_results:
            enriched = [
                (song_id, score, self._get_metadata(song_id))
                for song_id, score in results
            ]
            enriched_batch.append(enriched)
        
        return enriched_batch
    
    def get_by_genre(self, genre: str, k: int = 100) -> List[Dict]:
        """
        Get songs by genre (cached metadata filtering)
        
        Args:
            genre: Genre name
            k: Maximum number of songs
        
        Returns:
            List of song metadata
        """
        # Filter from cache
        genre_songs = [
            meta for meta in self.metadata_cache.values()
            if meta.get('genre', '').lower() == genre.lower()
        ]
        
        return genre_songs[:k]
    
    def get_all_genres(self) -> List[str]:
        """Get list of all unique genres"""
        genres = set()
        for meta in self.metadata_cache.values():
            if 'genre' in meta:
                genres.add(meta['genre'])
        return sorted(list(genres))
    
    def get_stats(self) -> Dict:
        """Get database statistics"""
        avg_query_time = np.mean(self.query_times) if self.query_times else 0
        
        return {
            'total_songs': self.db.count(),
            'total_genres': len(self.get_all_genres()),
            'avg_search_time': avg_query_time,
            'cache_size': len(self.metadata_cache),
            'dimension': self.dimension
        }
    
    def get_detailed_stats(self) -> Dict:
        """Get detailed database statistics"""
        stats = self.get_stats()
        
        # Try to get SrvDB internal stats
        try:
            db_info = self.db.info()
            stats['db_info'] = db_info
        except:
            pass
        
        stats.update({
            'mode': 'HNSW',
            'compression_ratio': 1.0,  # No compression for full precision
            'avg_query_ms': stats['avg_search_time'],
            'memory_mb': stats['total_songs'] * self.dimension * 4 / (1024 * 1024)  # Approx
        })
        
        return stats
    
    def _get_metadata(self, song_id: str) -> Dict:
        """
        Get metadata for a song
        
        Args:
            song_id: Song ID
        
        Returns:
            Metadata dictionary
        """
        # Check cache first
        if song_id in self.metadata_cache:
            return self.metadata_cache[song_id]
        
        # Fetch from database
        metadata_json = self.db.get(song_id)
        if metadata_json:
            metadata = json.loads(metadata_json)
            self.metadata_cache[song_id] = metadata
            return metadata
        
        return {}
    
    def optimize_for_search_speed(self):
        """Optimize database for faster search (increase ef_search)"""
        print("⚡ Optimizing for search speed...")
        self.db.set_ef_search(100)  # Higher = better recall, slightly slower
        print("✅ Optimization complete")
    
    def optimize_for_accuracy(self):
        """Optimize database for better accuracy (increase ef_search)"""
        print("🎯 Optimizing for accuracy...")
        self.db.set_ef_search(200)  # Higher = better recall
        print("✅ Optimization complete")
    
    def save_metadata_cache(self, filepath: str = None):
        """Save metadata cache to disk"""
        if filepath is None:
            filepath = self.db_path / "metadata_cache.json"
        
        with open(filepath, 'w') as f:
            json.dump(self.metadata_cache, f)
        
        print(f"💾 Saved metadata cache to {filepath}")
    
    def load_metadata_cache(self, filepath: str = None):
        """Load metadata cache from disk"""
        if filepath is None:
            filepath = self.db_path / "metadata_cache.json"
        
        if Path(filepath).exists():
            with open(filepath, 'r') as f:
                self.metadata_cache = json.load(f)
            print(f"📂 Loaded {len(self.metadata_cache)} metadata entries from cache")
        else:
            print("⚠️ No metadata cache found")
    
    def export_database(self, output_path: str):
        """Export database to JSON for analysis"""
        export_data = {
            'total_songs': self.db.count(),
            'dimension': self.dimension,
            'songs': []
        }
        
        for song_id, metadata in self.metadata_cache.items():
            export_data['songs'].append({
                'id': song_id,
                **metadata
            })
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"📤 Exported database to {output_path}")


# Example usage
if __name__ == "__main__":
    # Initialize database
    db_manager = SrvDBManager()
    
    # Create test embeddings
    test_embeddings = [np.random.randn(128) for _ in range(10)]
    test_metadata = [
        {
            'id': f'song_{i}',
            'title': f'Test Song {i}',
            'artist': f'Artist {i}',
            'genre': 'Rock' if i % 2 == 0 else 'Pop',
            'embedding': test_embeddings[i].tolist()
        }
        for i in range(10)
    ]
    
    # Add songs
    ids = db_manager.add_songs(test_embeddings, test_metadata)
    print(f"Added {len(ids)} test songs")
    
    # Search
    query = np.random.randn(128)
    results = db_manager.search_similar(query, k=5)
    
    print("\n🔍 Search Results:")
    for i, (song_id, score, meta) in enumerate(results):
        print(f"{i+1}. {meta['title']} - {meta['artist']} (Score: {score:.3f})")
    
    # Get stats
    stats = db_manager.get_stats()
    print(f"\n📊 Database Stats: {stats}")