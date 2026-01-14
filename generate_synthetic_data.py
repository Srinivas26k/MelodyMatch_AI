"""
Synthetic Data Generator for PageTurner AI (Book Recommendation System)
Uses Project Gutenberg metadata (simulated with real classic titles) to generate realistic book data.
"""

import numpy as np
import json
from pathlib import Path
from srvdb_manager import SrvDBManager
import random

# Real Gutenberg Classics Data (Sample of 100 for variety, expanded via variation)
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
    ("Crime and Punishment", "Fyodor Dostoevsky", "Classics"),
    ("War and Peace", "Leo Tolstoy", "Historical"),
    ("Anna Karenina", "Leo Tolstoy", "Romance"),
    ("Les Misérables", "Victor Hugo", "Historical"),
    ("Great Expectations", "Charles Dickens", "Fiction"),
    ("A Tale of Two Cities", "Charles Dickens", "Historical"),
    ("Ulysses", "James Joyce", "Classics"),
    ("Don Quixote", "Miguel de Cervantes", "Classics"),
    ("The Count of Monte Cristo", "Alexandre Dumas", "Adventure"),
    ("The Iliad", "Homer", "Classics"),
    ("Divine Comedy", "Dante Alighieri", "Poetry"),
    ("The Republic", "Plato", "Philosophy"),
    ("Meditations", "Marcus Aurelius", "Philosophy"),
    ("The Prince", "Niccolò Machiavelli", "Philosophy"),
    ("Walden", "Henry David Thoreau", "Philosophy"),
    ("Leaves of Grass", "Walt Whitman", "Poetry"),
    ("Heart of Darkness", "Joseph Conrad", "Fiction"),
    ("The Call of the Wild", "Jack London", "Adventure"),
    ("The Time Machine", "H.G. Wells", "Sci-Fi"),
    ("The War of the Worlds", "H.G. Wells", "Sci-Fi"),
    ("20,000 Leagues Under the Sea", "Jules Verne", "Adventure"),
    ("Around the World in Eighty Days", "Jules Verne", "Adventure"),
    ("Little Women", "Louisa May Alcott", "Fiction"),
    ("Wuthering Heights", "Emily Brontë", "Romance"),
    ("Sense and Sensibility", "Jane Austen", "Romance"),
    ("Emma", "Jane Austen", "Romance"),
    ("Treasure Island", "Robert Louis Stevenson", "Adventure"),
    ("Strange Case of Dr Jekyll and Mr Hyde", "Robert Louis Stevenson", "Horror"),
    ("Gulliver's Travels", "Jonathan Swift", "Satire"),
    ("Robinson Crusoe", "Daniel Defoe", "Adventure"),
    ("The Scarlet Letter", "Nathaniel Hawthorne", "Historical"),
    ("The Importance of Being Earnest", "Oscar Wilde", "Play"),
    ("Romeo and Juliet", "William Shakespeare", "Play"),
    ("Hamlet", "William Shakespeare", "Play"),
    ("Macbeth", "William Shakespeare", "Play"),
    ("A Midsummer Night's Dream", "William Shakespeare", "Play"),
    ("Paradise Lost", "John Milton", "Poetry"),
    ("The Canterbury Tales", "Geoffrey Chaucer", "Poetry"),
    ("Beowulf", "Unknown", "Classics"),
    ("The Art of War", "Sun Tzu", "Philosophy"),
    ("Tao Te Ching", "Laozi", "Philosophy"),
    ("Siddhartha", "Hermann Hesse", "Fiction"),
    ("The Stranger", "Albert Camus", "Philosophy"),
    ("Candide", "Voltaire", "Satire"),
    ("The Three Musketeers", "Alexandre Dumas", "Adventure"),
    ("Notes from Underground", "Fyodor Dostoevsky", "Philosophy"),
    ("Dead Souls", "Nikolai Gogol", "Satire"),
    ("Fathers and Sons", "Ivan Turgenev", "Fiction"),
    ("Madam Bovary", "Gustave Flaubert", "Fiction"),
    ("The Trial", "Franz Kafka", "Fiction"),
    ("The Castle", "Franz Kafka", "Fiction"),
    ("Dubliners", "James Joyce", "Short Stories"),
    ("Middlemarch", "George Eliot", "Historical"),
    ("Vanity Fair", "William Makepeace Thackeray", "Satire"),
    ("Tess of the d'Urbervilles", "Thomas Hardy", "Fiction"),
    ("Far from the Madding Crowd", "Thomas Hardy", "Romance"),
    ("The Jungle Book", "Rudyard Kipling", "Fiction"),
    ("Kim", "Rudyard Kipling", "Adventure"),
    ("White Fang", "Jack London", "Adventure"),
    ("The Sea-Wolf", "Jack London", "Adventure"),
    ("Of Human Bondage", "W. Somerset Maugham", "Fiction"),
    ("The Moon and Sixpence", "W. Somerset Maugham", "Fiction"),
    ("A Room with a View", "E.M. Forster", "Romance"),
    ("Howards End", "E.M. Forster", "Fiction"),
    ("A Passage to India", "E.M. Forster", "Historical"),
    ("Lord Jim", "Joseph Conrad", "Adventure"),
    ("Nostromo", "Joseph Conrad", "Politics"),
    ("The Secret Agent", "Joseph Conrad", "Thriller"),
    ("Sons and Lovers", "D.H. Lawrence", "Fiction"),
    ("Women in Love", "D.H. Lawrence", "Fiction"),
    ("Lady Chatterley's Lover", "D.H. Lawrence", "Romance"),
    ("The Good Soldier", "Ford Madox Ford", "Fiction"),
    ("To the Lighthouse", "Virginia Woolf", "Fiction"),
    ("Mrs Dalloway", "Virginia Woolf", "Fiction"),
    ("Orlando", "Virginia Woolf", "Historical"),
    ("Brave New World", "Aldous Huxley", "Sci-Fi"),
    ("The Grapes of Wrath", "John Steinbeck", "Historical"),
    ("Of Mice and Men", "John Steinbeck", "Fiction"),
    ("The Sound and the Fury", "William Faulkner", "Fiction"),
    ("As I Lay Dying", "William Faulkner", "Fiction"),
    ("The Sun Also Rises", "Ernest Hemingway", "Fiction"),
    ("A Farewell to Arms", "Ernest Hemingway", "War"),
    ("For Whom the Bell Tolls", "Ernest Hemingway", "War"),
    ("The Old Man and the Sea", "Ernest Hemingway", "Fiction")
]

def generate_synthetic_data(num_books=1000, db_path="./db/book_vectors"):
    """Generate realistic book data using Gutenberg classics"""
    
    print(f"🚀 Generating {num_books} books from Gutenberg Classics dataset...")
    
    embeddings = []
    metadata_list = []
    
    # We will loop and create variations to hit 1000 if needed, or simply pick random
    
    for i in range(num_books):
        # Pick a base book
        base_title, author, genre = random.choice(GUTENBERG_BOOKS)
        
        # Add variation to title to simulate different editions or volumes if we exceed unique books
        if i > len(GUTENBERG_BOOKS):
             title = f"{base_title} (Vol. {random.randint(1, 5)})"
        else:
             title = base_title
        
        # Generate varied stats
        rating = round(random.triangular(3.0, 5.0, 4.2), 1) # Good books skew higher
        pages = random.randint(150, 900)
        year = random.randint(1800, 1950) # Classics range
        
        # Description
        desc = f"A classic {genre} work by {author}. This edition of '{title}' brings you the timeless text that has captivated readers for generations. {random.choice(['A must-read.', 'A masterpiece.', 'Profound and moving.', 'An essential addition to any library.'])}"
        
        # Cover URL - Use a consistent seed based on title hash for stability
        seed = hash(title)
        cover_url = f"https://picsum.photos/seed/{seed}/200/300"
        
        # Generate embedding (128D) with genre clustering
        base_vector = np.random.randn(128)
        
        # Genre bias for clustering
        # Simple hash of genre string to indices
        genre_hash = sum(ord(c) for c in genre)
        base_vector[genre_hash % 10] += 1.5 
        base_vector[(genre_hash + 1) % 10] += 0.5
        
        # Normalize
        vector = base_vector / np.linalg.norm(base_vector)
        
        embeddings.append(vector)
        metadata_list.append({
            'id': f'book_{i}',
            'title': title,
            'author': author,
            'genre': genre,
            'rating': rating,
            'pages': pages,
            'year': year,
            'description': desc,
            'cover_url': cover_url,
            'track_id': i, # internal ID
            'embedding': vector.tolist()
        })

    print(f"💾 Initializing Database at {db_path}...")
    db_manager = SrvDBManager(db_path=db_path, dimension=128)
    
    # Add to database
    print("📝 Populating library...")
    db_manager.add_songs(embeddings, metadata_list)
    
    print("\n✅ Library populated with Gutenberg Classics!")
    print(f"Total books in DB: {db_manager.db.count()}")

if __name__ == "__main__":
    generate_synthetic_data()
