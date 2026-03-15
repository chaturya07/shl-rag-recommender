import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
import json
from pathlib import Path

class EmbeddingGenerator:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Initialize embedding generator
        
        Model options:
        - 'all-MiniLM-L6-v2': Fast, lightweight (recommended)
        - 'all-mpnet-base-v2': More accurate but slower
        - 'multi-qa-mpnet-base-dot-v1': Optimized for Q&A/search
        """
        print(f"\n{'='*70}")
        print("🤖 INITIALIZING EMBEDDING MODEL")
        print(f"{'='*70}")
        print(f"📦 Loading model: {model_name}")
        
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        
        print(f"✅ Model loaded successfully")
        print(f"📊 Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
    
    def generate_embeddings(self, input_file='data/shl_assessments_cleaned.csv',
                          output_dir='data/embeddings'):
        """
        Generate embeddings for all assessments
        """
        print(f"\n{'='*70}")
        print("🔤 GENERATING EMBEDDINGS")
        print(f"{'='*70}")
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Load cleaned data
        print(f"📂 Loading data from: {input_file}")
        df = pd.read_csv(input_file)
        print(f"✅ Loaded {len(df)} assessments")
        
        # Generate embeddings
        print(f"\n🔄 Generating embeddings...")
        print(f"   This may take 1-2 minutes...")
        
        texts = df['text_for_embedding'].tolist()
        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            batch_size=32,
            convert_to_numpy=True
        )
        
        print(f"\n✅ Generated embeddings")
        print(f"   Shape: {embeddings.shape}")
        print(f"   Size: {embeddings.nbytes / 1024 / 1024:.2f} MB")
        
        # Save embeddings
        embeddings_file = f"{output_dir}/embeddings.npy"
        np.save(embeddings_file, embeddings)
        print(f"\n💾 Saved embeddings: {embeddings_file}")
        
        # Save metadata
        metadata = {
            'assessment_names': df['Assessment Name'].tolist(),
            'descriptions': df['Short Description'].tolist(),
            'urls': df['URL'].tolist(),
            'types': df['assessment_type'].tolist(),
            'texts': texts,
            'model_name': self.model_name,
            'embedding_dim': embeddings.shape[1],
            'num_assessments': len(df)
        }
        
        metadata_file = f"{output_dir}/metadata.pkl"
        with open(metadata_file, 'wb') as f:
            pickle.dump(metadata, f)
        print(f"💾 Saved metadata: {metadata_file}")
        
        # Save config
        config = {
            'model_name': self.model_name,
            'embedding_dimension': int(embeddings.shape[1]),
            'num_assessments': len(df),
            'input_file': input_file,
            'embeddings_file': embeddings_file,
            'metadata_file': metadata_file
        }
        
        config_file = f"{output_dir}/config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"💾 Saved config: {config_file}")
        
        # Verify embeddings
        print(f"\n{'='*70}")
        print("✅ EMBEDDING GENERATION COMPLETE")
        print(f"{'='*70}")
        print(f"📊 Statistics:")
        print(f"   - Total assessments: {len(df)}")
        print(f"   - Embedding dimension: {embeddings.shape[1]}")
        print(f"   - Total embeddings size: {embeddings.nbytes / 1024 / 1024:.2f} MB")
        print(f"   - Model used: {self.model_name}")
        
        # Show sample similarities
        self._show_sample_similarities(embeddings, df['Assessment Name'].tolist())
        
        return embeddings, metadata
    
    def _show_sample_similarities(self, embeddings, names, num_samples=3):
        """Show sample similarity scores to verify embeddings"""
        print(f"\n{'='*70}")
        print(f"🔍 SAMPLE SIMILARITY CHECKS")
        print(f"{'='*70}")
        
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Pick a few random assessments
        sample_indices = np.random.choice(len(embeddings), min(num_samples, len(embeddings)), replace=False)
        
        for idx in sample_indices:
            query_embedding = embeddings[idx:idx+1]
            similarities = cosine_similarity(query_embedding, embeddings)[0]
            
            # Get top 5 similar (excluding itself)
            top_indices = np.argsort(similarities)[-6:-1][::-1]
            
            print(f"\n📌 Query: {names[idx]}")
            print(f"   Top 5 similar assessments:")
            for i, similar_idx in enumerate(top_indices, 1):
                print(f"   {i}. {names[similar_idx]} (similarity: {similarities[similar_idx]:.3f})")


def test_search(query_text, embeddings_dir='data/embeddings', top_k=5):
    """
    Test the embeddings by searching for similar assessments
    """
    print(f"\n{'='*70}")
    print(f"🔍 TESTING SEARCH")
    print(f"{'='*70}")
    print(f"Query: '{query_text}'")
    
    # Load model
    with open(f"{embeddings_dir}/config.json", 'r') as f:
        config = json.load(f)
    
    model = SentenceTransformer(config['model_name'])
    
    # Load embeddings and metadata
    embeddings = np.load(f"{embeddings_dir}/embeddings.npy")
    with open(f"{embeddings_dir}/metadata.pkl", 'rb') as f:
        metadata = pickle.load(f)
    
    # Generate query embedding
    query_embedding = model.encode([query_text], convert_to_numpy=True)
    
    # Calculate similarities
    from sklearn.metrics.pairwise import cosine_similarity
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    
    # Get top K
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    print(f"\n📊 Top {top_k} Results:")
    print(f"{'='*70}")
    
    for i, idx in enumerate(top_indices, 1):
        print(f"\n{i}. {metadata['assessment_names'][idx]}")
        print(f"   Type: {metadata['types'][idx]}")
        print(f"   Similarity: {similarities[idx]:.3f}")
        print(f"   URL: {metadata['urls'][idx]}")


def main():
    """Main execution"""
    # Generate embeddings
    generator = EmbeddingGenerator(model_name='all-MiniLM-L6-v2')
    embeddings, metadata = generator.generate_embeddings()
    
    # Test with sample queries
    print(f"\n{'='*70}")
    print(f"🧪 TESTING WITH SAMPLE QUERIES")
    print(f"{'='*70}")
    
    sample_queries = [
        "Python programming skills assessment",
        "Personality test for sales role",
        "Cognitive ability test"
    ]
    
    for query in sample_queries:
        test_search(query, top_k=3)
    
    print(f"\n{'='*70}")
    print(f"✅ ALL DONE!")
    print(f"{'='*70}")
    print(f"\n📁 Generated files:")
    print(f"   - data/embeddings/embeddings.npy")
    print(f"   - data/embeddings/metadata.pkl")
    print(f"   - data/embeddings/config.json")
    print(f"\n🚀 Next steps:")
    print(f"   1. ✅ Embeddings generated")
    print(f"   2. 📊 Next: Build vector search with FAISS")
    print(f"   3. 🤖 Next: Add LLM-based reranking")
    print(f"   4. 🌐 Next: Create API endpoint")


if __name__ == "__main__":
    main()