# 📚 PageTurner AI

**PageTurner AI** is an intelligent book recommendation engine that combines local vector search with global API retrieval to help you discover your next great read.

![PageTurner AI](https://via.placeholder.com/800x400?text=PageTurner+AI+Dashboard)

## 🌟 Features

-   **Hybrid Search**: 
    -   **Local Library**: Instant vector-based semantic search over a curated collection of 1,000+ classic books.
    -   **Global Fallback**: Automatically searches the **OpenLibrary API** if a book isn't in your local database.
-   **Lazy Indexing**: Bookmarking an external book automatically adds it to your local SrvDB vector index, growing your personal library over time.
-   **Smart Filtering**: Filter by Genre, Rating, and Page Count with real-time updates.
-   **Persistent Bookmarks**: Your reading list is saved locally and persists across sessions.
-   **Literary Analytics**: Visualize your collection's genre distribution and statistics.

## 🛠️ Technology Stack

-   **Frontend**: [Streamlit](https://streamlit.io/)
-   **Database**: SrvDB (Custom Vector Database with HNSW Indexing)
-   **Data Source**: Project Gutenberg (Local) & OpenLibrary API (External)
-   **Visualization**: Plotly

## 🚀 Getting Started

### Prerequisites

-   Python 3.8+
-   pip

### Installation

1.  **Clone the repository**
    ```bash
    git clone https://github.com/yourusername/PageTurner-AI.git
    cd PageTurner-AI
    ```

2.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the application**
    ```bash
    streamlit run app.py
    ```

4.  **Open in Browser**
    The app will open automatically at `http://localhost:8501`.

## 📂 Project Structure

-   `app.py`: Main Streamlit application and UI logic.
-   `recommender.py`: Recommendation engine handling search and API integration.
-   `srvdb_manager.py`: Interface for the SrvDB vector database.
-   `bookmark_manager.py`: Handles persistent storage of user bookmarks.
-   `generate_synthetic_data.py`: Script to generate/reset the local book database.
-   `bookmarks.json`: JSON store for user bookmarks.
-   `db/`: Directory containing the vector database files (GitIgnored).

## 🤝 contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.