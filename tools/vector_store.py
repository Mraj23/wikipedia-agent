"""
Vector store for MaterialQA agent with RAG capabilities.
"""

import hashlib
from typing import List, Dict, Optional, Tuple
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
import os
import pickle
import shutil

# Fix tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class MaterialQAVectorStore:
    """In-memory vector store for Wikipedia articles with RAG capabilities."""
    
    def __init__(self, persist_path: Optional[str] = None):
        """Initialize the vector store with embeddings model."""
        
        # Use a better performing embedding model
        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",  # Better performance than MiniLM
            model_kwargs={'device': 'cpu'}
        )
        
        # Text splitter for chunking articles - larger chunks for better context
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,  # Larger chunks for better context
            chunk_overlap=200,  # More overlap for continuity
            separators=["\n\n", "\n", ". ", " "]
        )
        
        # Vector store (starts empty)
        self.vector_store: Optional[FAISS] = None
        self.persist_path = persist_path
        
        # Track what articles we've indexed
        self.indexed_urls = set()
        
        # Load existing index if available
        if persist_path and os.path.exists(persist_path):
            self._load_index()
    
    def add_articles(self, articles: List[Dict[str, str]]) -> int:
        """
        Add Wikipedia articles to the vector store.
        
        Args:
            articles: List of article dicts with 'title', 'content', 'url'
            
        Returns:
            Number of new chunks added
        """
        new_chunks = 0
        documents = []
        
        for article in articles:
            article_url = article['url']
            
            # Skip if already indexed
            if article_url in self.indexed_urls:
                continue
            
            # Get full content (not truncated)
            full_content = article.get('full_content', article.get('content', ''))
            if not full_content:
                continue
            
            # Split article into chunks
            chunks = self.text_splitter.split_text(full_content)
            
            # Create documents with metadata
            for i, chunk in enumerate(chunks):
                doc = Document(
                    page_content=chunk,
                    metadata={
                        'title': article['title'],
                        'url': article_url,
                        'chunk_id': i,
                        'source': f"{article['title']} (chunk {i+1})"
                    }
                )
                documents.append(doc)
                new_chunks += 1
            
            self.indexed_urls.add(article_url)
            print(f"📄 Indexed: {article['title']} ({len(chunks)} chunks)")
        
        if documents:
            # Add to vector store
            if self.vector_store is None:
                self.vector_store = FAISS.from_documents(documents, self.embeddings)
            else:
                self.vector_store.add_documents(documents)
            
            # Persist if path provided
            if self.persist_path:
                self._save_index()
        
        return new_chunks
    
    def similarity_search(self, query: str, k: int = 5, score_threshold: float = 0.3) -> List[Dict]:
        """
        Search for relevant chunks using semantic similarity.
        
        Args:
            query: Search query
            k: Number of results to return
            score_threshold: Minimum similarity score (0-1)
            
        Returns:
            List of relevant chunks with metadata and scores
        """
        if self.vector_store is None:
            return []
        
        # Get documents with scores
        docs_with_scores = self.vector_store.similarity_search_with_score(query, k=k*2)
        
        # Filter by score threshold and format results
        results = []
        seen_sources = set()
        
        for doc, score in docs_with_scores:
            # FAISS returns L2 distance, convert to similarity (lower distance = higher similarity)
            # Use a more forgiving normalization for better retrieval
            similarity = max(0, 1 - score/4)  # More forgiving conversion
            
            if similarity >= score_threshold and len(results) < k:
                source_key = f"{doc.metadata['url']}_{doc.metadata['chunk_id']}"
                
                # Avoid duplicate chunks
                if source_key not in seen_sources:
                    results.append({
                        'content': doc.page_content,
                        'title': doc.metadata['title'],
                        'url': doc.metadata['url'],
                        'chunk_id': doc.metadata['chunk_id'],
                        'source': doc.metadata['source'],
                        'similarity': similarity
                    })
                    seen_sources.add(source_key)
        
        return results
    
    def get_stats(self) -> Dict:
        """Get statistics about the vector store."""
        if self.vector_store is None:
            return {
                'total_chunks': 0,
                'total_articles': 0,
                'indexed_urls': []
            }
        
        return {
            'total_chunks': self.vector_store.index.ntotal,
            'total_articles': len(self.indexed_urls),
            'indexed_urls': list(self.indexed_urls)
        }
    
    def clear(self):
        """Clear the vector store and all persisted data."""
        self.vector_store = None
        self.indexed_urls.clear()
        
        if self.persist_path:
            try:
                # Remove FAISS directory if it exists
                if os.path.exists(self.persist_path) and os.path.isdir(self.persist_path):
                    shutil.rmtree(self.persist_path)
                    print(f"🗑️ Removed FAISS directory: {self.persist_path}")
                
                # Remove metadata file if it exists
                metadata_path = f"{self.persist_path}_metadata.pkl"
                if os.path.exists(metadata_path):
                    os.remove(metadata_path)
                    print(f"🗑️ Removed metadata file: {metadata_path}")
                    
            except Exception as e:
                print(f"⚠️ Error clearing persisted data: {e}")
        
        print("🗑️ Vector store cleared")
    
    def _save_index(self):
        """Save the vector store to disk."""
        try:
            if self.vector_store and self.persist_path:
                # Save FAISS index
                self.vector_store.save_local(self.persist_path)
                
                # Save metadata
                metadata_path = f"{self.persist_path}_metadata.pkl"
                with open(metadata_path, 'wb') as f:
                    pickle.dump({
                        'indexed_urls': self.indexed_urls
                    }, f)
        except Exception as e:
            print(f"Warning: Could not save vector store: {e}")
    
    def _load_index(self):
        """Load the vector store from disk."""
        try:
            if os.path.exists(self.persist_path):
                # Load FAISS index
                self.vector_store = FAISS.load_local(
                    self.persist_path, 
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                
                # Load metadata
                metadata_path = f"{self.persist_path}_metadata.pkl"
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'rb') as f:
                        metadata = pickle.load(f)
                        self.indexed_urls = metadata.get('indexed_urls', set())
                
                print(f"📚 Loaded vector store: {len(self.indexed_urls)} articles indexed")
        except Exception as e:
            print(f"Warning: Could not load vector store: {e}")
            self.vector_store = None
            self.indexed_urls = set()
