"""
Synthetic Data Generator for MelodyMatch AI
Generates realistic data for Books, Music, and Movies using srvdb.
"""

import numpy as np
import json
from pathlib import Path
from srvdb_manager import SrvDBManager
import random
import shutil

# --- DATASETS ---

GUTENBERG_BOOKS = [
    ("Pride and Prejudice", "Jane Austen", "Romance"),
    ("Moby Dick", "Herman Melville", "Adventure"),
    ("Frankenstein", "Mary Shelley", "Horror"),
    ("The Great Gatsby", "F. Scott Fitzgerald", "Fiction"),
    ("1984", "George Orwell", "Sci-Fi"),
    ("The Prophet", "Kahlil Gibran", "Poetry"),
    ("Alice's Adventures in Wonderland", "Lewis Carroll", "Fantasy"),
    ("The Adventures of Sherlock Holmes", "Arthur Conan Doyle", "Mystery"),
    ("The Picture of Dorian Gray", "Oscar Wilde", "Fiction"),
    ("Metamorphosis", "Franz Kafka", "Fiction"),
    ("Dracula", "Bram Stoker", "Horror"),
    ("Jane Eyre", "Charlotte Brontë", "Romance"),
    ("The Odyssey", "Homer", "Classics"),
    ("The Brothers Karamazov", "Fyodor Dostoevsky", "Classics"),
    ("War and Peace", "Leo Tolstoy", "Historical"),
    ("Don Quixote", "Miguel de Cervantes", "Classics"),
    ("The Count of Monte Cristo", "Alexandre Dumas", "Adventure"),
    ("The Iliad", "Homer", "Classics"),
    ("Divine Comedy", "Dante Alighieri", "Poetry"),
    ("The Republic", "Plato", "Philosophy")
]

CLASSIC_MUSIC = [
    ("Bohemian Rhapsody", "Queen", "Rock"),
    ("Imagine", "John Lennon", "Pop"),
    ("Smells Like Teen Spirit", "Nirvana", "Grunge"),
    ("Billie Jean", "Michael Jackson", "Pop"),
    ("Hotel California", "Eagles", "Rock"),
    ("Purple Haze", "Jimi Hendrix", "Rock"),
    ("Like a Rolling Stone", "Bob Dylan", "Folk Rock"),
    ("I Will Always Love You", "Whitney Houston", "R&B"),
    ("Hey Jude", "The Beatles", "Rock"),
    ("Respect", "Aretha Franklin", "Soul"),
    ("What's Going On", "Marvin Gaye", "Soul"),
    ("Born to Run", "Bruce Springsteen", "Rock"),
    ("Stairway to Heaven", "Led Zeppelin", "Rock"),
    ("Heroes", "David Bowie", "Rock"),
    ("Thriller", "Michael Jackson", "Pop"),
    ("Vogue", "Madonna", "Pop"),
    ("Wonderwall", "Oasis", "Britpop"),
    ("Creep", "Radiohead", "Alt Rock"),
    ("Lose Yourself", "Eminem", "Hip Hop"),
    ("Hallelujah", "Jeff Buckley", "Alt Rock")
]

CLASSIC_MOVIES = [
    ("The Godfather", "Francis Ford Coppola", "Crime"),
    ("The Shawshank Redemption", "Frank Darabont", "Drama"),
    ("Pulp Fiction", "Quentin Tarantino", "Crime"),
    ("The Dark Knight", "Christopher Nolan", "Action"),
    ("Schindler's List", "Steven Spielberg", "Biography"),
    ("12 Angry Men", "Sidney Lumet", "Drama"),
    ("Forrest Gump", "Robert Zemeckis", "Drama"),
    ("Inception", "Christopher Nolan", "Sci-Fi"),
    ("The Matrix", "Lana Wachowski", "Sci-Fi"),
    ("Fight Club", "David Fincher", "Drama"),
    ("Goodfellas", "Martin Scorsese", "Biography"),
    ("Star Wars: A New Hope", "George Lucas", "Sci-Fi"),
    ("Parasite", "Bong Joon Ho", "Thriller"),
    ("Casablanca", "Michael Curtiz", "Romance"),
    ("Rear Window", "Alfred Hitchcock", "Mystery"),
    ("The Silence of the Lambs", "Jonathan Demme", "Thriller"),
    ("Se7en", "David Fincher", "Crime"),
    ("Interstellar", "Christopher Nolan", "Sci-Fi"),
    ("Spirited Away", "Hayao Miyazaki", "Animation"),
    ("The Lion King", "Roger Allers", "Animation")
]

def generate_dataset(media_type, source_data, num_items=500, db_path="./db/vectors"):
    """Generate synthetic dataset for a specific media type"""
    
    print(f"\n🚀 Generating {num_items} {media_type} items at {db_path}...")
    
    # Clean up existing to avoid duplicates during regen
    path = Path(db_path)
    if path.exists():
        try:
            shutil.rmtree(path)
            print(f"   Cleared existing DB at {db_path}")
        except Exception as e:
            print(f"   Warning: Could not clear DB: {e}")

    embeddings = []
    metadata_list = []
    
    for i in range(num_items):
        # Pick base item
        base_title, creator, genre = random.choice(source_data)
        
        # Add variation
        if i > len(source_data):
            # cycle through
            variation = i // len(source_data) + 1
            if media_type == 'music':
                title = f"{base_title} (Remix {variation})"
            elif media_type == 'movie':
                title = f"{base_title} {variation}"
            else:
                title = f"{base_title} (Ed. {variation})"
        else:
            title = base_title
            
        # Stats
        rating = round(random.triangular(3.0, 5.0, 4.2), 1)
        
        # Type specific attributes
        extra_meta = {}
        desc_tmpl = ""
        
        if media_type == 'book':
            pages = random.randint(150, 900)
            year = random.randint(1800, 1950)
            extra_meta = {'pages': pages, 'year': year, 'author': creator}
            desc_tmpl = f"A classic {genre} novel by {creator}."
            
        elif media_type == 'music':
            duration = random.randint(180, 400) # seconds
            year = random.randint(1960, 2023)
            extra_meta = {'duration': duration, 'year': year, 'artist': creator}
            desc_tmpl = f"A hit {genre} song by {creator}."

        elif media_type == 'movie':
            duration_min = random.randint(80, 180) # minutes
            year = random.randint(1940, 2023)
            extra_meta = {'duration': duration_min, 'year': year, 'director': creator}
            desc_tmpl = f"A masterpiece {genre} film directed by {creator}."

        # Description
        desc = f"{desc_tmpl} Widely acclaimed and highly rated. {random.choice(['Must experience.', 'Critics choice.', 'Fan favorite.'])}"
        
        # Cover URL (Placeholder)
        seed = hash(title + media_type)
        cover_url = f"https://picsum.photos/seed/{seed}/200/300"
        
        # Generate embedding (128D) with genre clustering
        base_vector = np.random.randn(128)
        
        # Genre bias for clustering
        genre_hash = sum(ord(c) for c in genre)
        base_vector[genre_hash % 10] += 1.5 
        base_vector[(genre_hash + 1) % 10] += 0.5
        
        # Normalize
        vector = base_vector / np.linalg.norm(base_vector)
        
        embeddings.append(vector)
        metadata_list.append({
            'id': f'{media_type}_{i}',
            'title': title,
            'genre': genre,
            'rating': rating,
            'description': desc,
            'cover_url': cover_url,
            'embedding': vector.tolist(),
            **extra_meta
        })

    # Initialize and Populate
    db_manager = SrvDBManager(db_path=db_path, dimension=128)
    db_manager.add_books(embeddings, metadata_list) # method name is generic enough internally despite name
    print(f"✅ {media_type.capitalize()} DB populated! Count: {db_manager.db.count()}")

def main():
    # Generate all 3 datasets
    generate_dataset('book', GUTENBERG_BOOKS, num_items=200, db_path="./db/book_vectors")
    generate_dataset('music', CLASSIC_MUSIC, num_items=200, db_path="./db/music_vectors")
    generate_dataset('movie', CLASSIC_MOVIES, num_items=200, db_path="./db/movie_vectors")
    
    print("\n🎉 All synthetic data generated successfully!")

if __name__ == "__main__":
    main()
