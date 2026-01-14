import json
from pathlib import Path
from typing import List, Dict

class BookmarkManager:
    """Manages persistent bookmarks using a JSON file"""
    
    def __init__(self, filepath: str = "bookmarks.json"):
        self.filepath = Path(filepath)
        self.bookmarks = self._load_bookmarks()
        
    def _load_bookmarks(self) -> List[str]:
        """Load bookmarks from file"""
        if self.filepath.exists():
            try:
                with open(self.filepath, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return []
        return []
        
    def save_bookmarks(self):
        """Save current bookmarks to file"""
        with open(self.filepath, 'w') as f:
            json.dump(self.bookmarks, f)
            
    def add_bookmark(self, book_id: str):
        """Add a book ID to bookmarks if not present"""
        if book_id not in self.bookmarks:
            self.bookmarks.append(book_id)
            self.save_bookmarks()
            
    def remove_bookmark(self, book_id: str):
        """Remove a book ID from bookmarks"""
        if book_id in self.bookmarks:
            self.bookmarks.remove(book_id)
            self.save_bookmarks()
            
    def get_bookmarks(self) -> List[str]:
        """Get list of bookmarked IDs"""
        return self.bookmarks
    
    def is_bookmarked(self, book_id: str) -> bool:
        """Check if a book is bookmarked"""
        return book_id in self.bookmarks
