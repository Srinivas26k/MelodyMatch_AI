"""
Audio Data Processor - Extract features from music files
Processes FMA dataset and creates embeddings for SrvDB
"""

import librosa
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import json
from tqdm import tqdm
import argparse
import warnings
warnings.filterwarnings('ignore')

from srvdb_manager import SrvDBManager


class AudioProcessor:
    """Extract audio features from music files"""
    
    def __init__(self, sample_rate: int = 22050, duration: int = 30):
        """
        Initialize Audio Processor
        
        Args:
            sample_rate: Audio sample rate (Hz)
            duration: Duration to analyze (seconds)
        """
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_mfcc = 20
        self.n_chroma = 12
        self.n_contrast = 7
        
    def extract_features(self, audio_path: str) -> np.ndarray:
        """
        Extract comprehensive audio features
        
        Args:
            audio_path: Path to audio file
        
        Returns:
            Feature vector (128D)
        """
        try:
            # Load audio
            y, sr = librosa.load(
                audio_path,
                sr=self.sample_rate,
                duration=self.duration
            )
            
            # Extract features
            features = []
            
            # 1. MFCC (20 features) - Timbre
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
            mfcc_mean = np.mean(mfcc, axis=1)
            features.extend(mfcc_mean)
            
            # 2. Chroma (12 features) - Pitch content
            chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=self.n_chroma)
            chroma_mean = np.mean(chroma, axis=1)
            features.extend(chroma_mean)
            
            # 3. Spectral Contrast (7 features) - Texture
            contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=self.n_contrast-1)
            contrast_mean = np.mean(contrast, axis=1)
            features.extend(contrast_mean)
            
            # 4. Spectral Features (8 features)
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
            spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
            spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
            zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y))
            
            features.extend([
                spectral_centroid,
                spectral_bandwidth,
                spectral_rolloff,
                zero_crossing_rate
            ])
            
            # 5. Rhythm Features (4 features)
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            features.append(tempo)
            
            # Beat strength
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            features.append(np.mean(onset_env))
            features.append(np.std(onset_env))
            features.append(np.max(onset_env))
            
            # 6. Harmonic/Percussive Separation (2 features)
            y_harmonic, y_percussive = librosa.effects.hpss(y)
            harmonic_ratio = np.sum(np.abs(y_harmonic)) / (np.sum(np.abs(y)) + 1e-6)
            percussive_ratio = np.sum(np.abs(y_percussive)) / (np.sum(np.abs(y)) + 1e-6)
            features.extend([harmonic_ratio, percussive_ratio])
            
            # 7. Tonnetz (6 features) - Tonal space
            tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
            tonnetz_mean = np.mean(tonnetz, axis=1)
            features.extend(tonnetz_mean)
            
            # 8. Energy and Dynamics (8 features)
            rms = librosa.feature.rms(y=y)
            features.append(np.mean(rms))
            features.append(np.std(rms))
            features.append(np.max(rms))
            features.append(np.min(rms))
            
            # Dynamic range
            features.append(np.max(rms) - np.min(rms))
            
            # Loudness distribution
            features.append(np.percentile(rms, 25))
            features.append(np.percentile(rms, 75))
            features.append(np.median(rms))
            
            # 9. Mel Spectrogram Statistics (10 features)
            mel = librosa.feature.melspectrogram(y=y, sr=sr)
            mel_db = librosa.power_to_db(mel, ref=np.max)
            
            features.append(np.mean(mel_db))
            features.append(np.std(mel_db))
            features.append(np.max(mel_db))
            features.append(np.min(mel_db))
            features.append(np.median(mel_db))
            
            # Spectral flux
            spectral_flux = np.sum(np.diff(mel, axis=1)**2, axis=0)
            features.append(np.mean(spectral_flux))
            features.append(np.std(spectral_flux))
            features.append(np.max(spectral_flux))
            
            # Spectral spread
            features.append(np.mean(np.std(mel, axis=0)))
            features.append(np.std(np.std(mel, axis=0)))
            
            # 10. Additional Spectral Features (11 features to reach 128)
            spectral_flatness = np.mean(librosa.feature.spectral_flatness(y=y))
            features.append(spectral_flatness)
            
            # Poly features
            poly_features = librosa.feature.poly_features(y=y, sr=sr, order=1)
            features.extend(np.mean(poly_features, axis=1))
            
            # Pad or truncate to exactly 128 dimensions
            feature_vector = np.array(features)
            if len(feature_vector) < 128:
                feature_vector = np.pad(feature_vector, (0, 128 - len(feature_vector)))
            else:
                feature_vector = feature_vector[:128]
            
            # Normalize
            feature_vector = (feature_vector - np.mean(feature_vector)) / (np.std(feature_vector) + 1e-6)
            
            return feature_vector
            
        except Exception as e:
            print(f"Error processing {audio_path}: {str(e)}")
            return None
    
    def extract_features_from_file(self, uploaded_file) -> np.ndarray:
        """Extract features from uploaded file (Streamlit)"""
        # Save temporarily
        temp_path = Path("temp_audio.mp3")
        with open(temp_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        
        features = self.extract_features(str(temp_path))
        
        # Clean up
        temp_path.unlink()
        
        return features
    
    def process_fma_dataset(self, 
                           fma_path: str,
                           metadata_path: str,
                           output_path: str = "./data/processed",
                           max_songs: int = None):
        """
        Process FMA dataset and create embeddings
        
        Args:
            fma_path: Path to fma_medium directory
            metadata_path: Path to tracks.csv
            output_path: Where to save processed data
            max_songs: Maximum number of songs to process
        """
        print("🎵 Starting FMA Dataset Processing")
        print("=" * 60)
        
        # Load metadata
        print("📂 Loading metadata...")
        tracks_df = pd.read_csv(metadata_path, header=[0, 1], low_memory=False)
        
        # Flatten column names
        tracks_df.columns = ['_'.join(col).strip() for col in tracks_df.columns.values]
        
        # Get relevant columns
        track_ids = tracks_df.index.tolist()
        
        if max_songs:
            track_ids = track_ids[:max_songs]
        
        print(f"📊 Processing {len(track_ids)} tracks")
        
        # Process songs
        embeddings = []
        metadata_list = []
        failed_tracks = []
        
        fma_path = Path(fma_path)
        
        for track_id in tqdm(track_ids, desc="🎼 Extracting features"):
            # Construct file path (FMA structure: fma_medium/XXX/XXXXXX.mp3)
            track_id_str = str(track_id).zfill(6)
            subfolder = track_id_str[:3]
            audio_path = fma_path / subfolder / f"{track_id_str}.mp3"
            
            if not audio_path.exists():
                failed_tracks.append(track_id)
                continue
            
            # Extract features
            features = self.extract_features(str(audio_path))
            
            if features is None:
                failed_tracks.append(track_id)
                continue
            
            # Get metadata
            try:
                track_meta = tracks_df.loc[track_id]
                
                # Extract key metadata fields (adjust based on actual FMA structure)
                metadata = {
                    'id': f'track_{track_id}',
                    'title': str(track_meta.get('track_title', f'Track {track_id}')),
                    'artist': str(track_meta.get('artist_name', 'Unknown')),
                    'album': str(track_meta.get('album_title', 'Unknown')),
                    'genre': str(track_meta.get('track_genre_top', 'Unknown')),
                    'duration': float(track_meta.get('track_duration', 0)),
                    'track_id': track_id,
                    'embedding': features.tolist()
                }
                
                embeddings.append(features)
                metadata_list.append(metadata)
                
            except Exception as e:
                print(f"Error getting metadata for track {track_id}: {e}")
                failed_tracks.append(track_id)
        
        print(f"\n✅ Successfully processed {len(embeddings)} tracks")
        print(f"❌ Failed tracks: {len(failed_tracks)}")
        
        # Save processed data
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print("\n💾 Saving processed data...")
        
        # Save embeddings
        np.save(output_path / "embeddings.npy", np.array(embeddings))
        
        # Save metadata
        with open(output_path / "metadata.json", 'w') as f:
            json.dump(metadata_list, f, indent=2)
        
        # Save failed tracks
        with open(output_path / "failed_tracks.txt", 'w') as f:
            f.write('\n'.join(map(str, failed_tracks)))
        
        print(f"✅ Saved to {output_path}")
        
        return embeddings, metadata_list
    
    def build_srvdb_database(self,
                            processed_data_path: str = "./data/processed",
                            db_path: str = "./db/music_vectors"):
        """
        Build SrvDB database from processed data
        
        Args:
            processed_data_path: Path to processed embeddings and metadata
            db_path: Path for SrvDB database
        """
        print("🚀 Building SrvDB Database")
        print("=" * 60)
        
        processed_path = Path(processed_data_path)
        
        # Load processed data
        print("📂 Loading processed data...")
        embeddings = np.load(processed_path / "embeddings.npy")
        
        with open(processed_path / "metadata.json", 'r') as f:
            metadata = json.load(f)
        
        print(f"📊 Loaded {len(embeddings)} embeddings")
        
        # Initialize database
        print("\n🗄️ Initializing SrvDB...")
        db_manager = SrvDBManager(db_path=db_path, dimension=128)
        
        # Add songs in batches
        batch_size = 1000
        total_batches = (len(embeddings) + batch_size - 1) // batch_size
        
        for i in range(0, len(embeddings), batch_size):
            batch_end = min(i + batch_size, len(embeddings))
            batch_embeddings = embeddings[i:batch_end]
            batch_metadata = metadata[i:batch_end]
            
            print(f"📝 Adding batch {i//batch_size + 1}/{total_batches}...")
            db_manager.add_songs(batch_embeddings, batch_metadata)
        
        # Save metadata cache
        db_manager.save_metadata_cache()
        
        print("\n✅ Database built successfully!")
        print(f"📊 Total songs: {db_manager.db.count()}")
        
        # Test search
        print("\n🔍 Testing search...")
        test_query = embeddings[0]
        results = db_manager.search_similar(test_query, k=5)
        
        print("Top 5 results:")
        for i, (song_id, score, meta) in enumerate(results):
            print(f"{i+1}. {meta['title']} - {meta['artist']} (Score: {score:.3f})")


def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(description="Process FMA dataset for music recommendation")
    
    parser.add_argument('--fma-path', type=str, default='./data/fma_medium',
                       help='Path to fma_medium directory')
    parser.add_argument('--metadata-path', type=str, default='./data/fma_metadata/tracks.csv',
                       help='Path to tracks.csv')
    parser.add_argument('--output-path', type=str, default='./data/processed',
                       help='Output path for processed data')
    parser.add_argument('--db-path', type=str, default='./db/music_vectors',
                       help='Path for SrvDB database')
    parser.add_argument('--max-songs', type=int, default=None,
                       help='Maximum number of songs to process')
    parser.add_argument('--build-database', action='store_true',
                       help='Build SrvDB database from processed data')
    parser.add_argument('--process-only', action='store_true',
                       help='Only process audio, don\'t build database')
    
    args = parser.parse_args()
    
    processor = AudioProcessor()
    
    if not args.build_database or not args.process_only:
        # Process FMA dataset
        embeddings, metadata = processor.process_fma_dataset(
            fma_path=args.fma_path,
            metadata_path=args.metadata_path,
            output_path=args.output_path,
            max_songs=args.max_songs
        )
    
    if args.build_database and not args.process_only:
        # Build database
        processor.build_srvdb_database(
            processed_data_path=args.output_path,
            db_path=args.db_path
        )


if __name__ == "__main__":
    main()