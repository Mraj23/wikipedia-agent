"""
Simple Wikipedia Q&A Agent
Clean workflow: Query -> Rewrite -> Search -> Filter -> Embed -> Retrieve -> Answer
"""

from typing import List, Dict, Optional
from langchain_anthropic import ChatAnthropic
from langchain.schema import HumanMessage, SystemMessage
import config
from tools.wikipedia_search import WikipediaSearcher
from tools.vector_store import MaterialQAVectorStore


class SimpleWikipediaAgent:
    def __init__(self):
        self.llm = ChatAnthropic(
            model="claude-3-5-sonnet-20241022",
            api_key=config.ANTHROPIC_API_KEY,
            temperature=0.1
        )
        self.wikipedia_searcher = WikipediaSearcher()
        self.vector_store = MaterialQAVectorStore(persist_path="vector_store")
        # Note: URL deduplication now handled by vector_store.indexed_urls
    
    def _format_conversation_history(self, conversation_history: Optional[List[Dict]] = None) -> str:
        """Format conversation history for inclusion in prompts."""
        if not conversation_history:
            return ""
        
        # Take last 4 human/AI message pairs (8 messages total)
        recent_messages = conversation_history[-8:] if len(conversation_history) > 8 else conversation_history
        
        if not recent_messages:
            return ""
        
        formatted_history = "\n\nRecent conversation context:\n"
        for msg in recent_messages:
            role = "Human" if msg.get("role") == "user" else "Assistant"
            content = msg.get("content", "").strip()
            if content:
                formatted_history += f"{role}: {content}\n"
        
        return formatted_history
    
    def can_answer_from_vector_store(self, query: str, conversation_history: Optional[List[Dict]] = None) -> tuple[bool, List[Dict]]:
        """Check if we can answer the query from existing vector store."""
        chunks = self.vector_store.similarity_search(query, k=7, score_threshold=0.3)
        
        if not chunks:
            return False, []
        
        # Use LLM to check if chunks are sufficient
        chunks_text = "\n\n".join([
            f"From {chunk['source']}:\n{chunk['content'][:400]}"
            for chunk in chunks
        ])
        
        # Get context about what we already know
        vector_store_stats = self.vector_store.get_stats()
        existing_titles = [url.split('/')[-1].replace('_', ' ') for url in vector_store_stats.get('indexed_urls', [])]
        context_info = f"\n\nKnowledge base contains articles about: {', '.join(existing_titles[:5])}" if existing_titles else ""
        
        # Add conversation history context
        conversation_context = self._format_conversation_history(conversation_history)
        
        prompt = f"""Question: {query}{conversation_context}

Available information:
{chunks_text}{context_info}

Can this question be answered comprehensively with the available information above?
Consider the conversation context and if the information is specific enough to cover the key aspects of the question.
Respond with only "YES" or "NO":"""

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            can_answer = "YES" in response.content.upper()
            print(f"📊 Vector store check: {'✅ Sufficient' if can_answer else '❌ Need more info'}")
            return can_answer, chunks
        except Exception as e:
            print(f"Error checking vector store sufficiency: {e}")
            return False, chunks
    
    def generate_answer(self, query: str, chunks: List[Dict], conversation_history: Optional[List[Dict]] = None) -> str:
        """Generate answer using retrieved chunks."""
        if not chunks:
            return "I don't have enough information to answer this question."
        
        context = "\n\n".join([
            f"From {chunk['source']}:\n{chunk['content']}"
            for chunk in chunks
        ])
        
        # Add conversation history context
        conversation_context = self._format_conversation_history(conversation_history)
        
        system_prompt = """You are a helpful assistant that answers questions using provided Wikipedia content. 
        
Instructions:
- Use only the provided information to answer
- Be specific and detailed
- Consider the conversation context for follow-up questions and references
- If the information is insufficient, say so
- Include relevant details and explanations"""
        
        user_prompt = f"""Question: {query}{conversation_context}

Information from Wikipedia:
{context}

Please provide a comprehensive answer based on the information above, considering the conversation context."""

        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            return f"Error generating answer: {e}"
    
    def search_new_articles(self, query: str, conversation_history: Optional[List[Dict]] = None) -> List[Dict]:
        """Search for new Wikipedia articles with context awareness."""
        print("🔍 Searching for new articles...")
        
        # Get existing articles for context
        vector_store_stats = self.vector_store.get_stats()
        already_indexed = set(vector_store_stats.get('indexed_urls', []))
        
        # Extract article titles from URLs for context
        existing_titles = []
        for url in vector_store_stats.get('indexed_urls', []):
            # Extract title from Wikipedia URL
            title = url.split('/')[-1].replace('_', ' ')
            existing_titles.append(title)
        
        # Get articles using the simple search workflow with context
        articles = self.wikipedia_searcher.search_and_filter(
            query, 
            max_articles=3, 
            existing_articles=existing_titles,
            conversation_history=conversation_history
        )
        
        new_articles = []
        for article in articles:
            if article['url'] not in already_indexed:
                new_articles.append(article)
                print(f"  + NEW: {article['title']}")
            else:
                print(f"  ↻ SKIP: {article['title']} (already in knowledge base)")
        
        return new_articles
    
    def add_articles_to_vector_store(self, articles: List[Dict]) -> int:
        """Add articles to vector store and return number of chunks added."""
        if not articles:
            return 0
        
        chunks_added = self.vector_store.add_articles(articles)
        print(f"📚 Added {chunks_added} chunks to vector store")
        return chunks_added
    
    def ask(self, query: str, conversation_history: Optional[List[Dict]] = None) -> Dict:
        """Main method to ask a question."""
        print(f"\n❓ Question: {query}")
        print("=" * 60)
        
        # Step 1: Try to answer from existing vector store
        can_answer, chunks = self.can_answer_from_vector_store(query, conversation_history)
        
        if not can_answer:
            # Step 2: Search for new articles
            new_articles = self.search_new_articles(query, conversation_history)
            
            if new_articles:
                # Step 3: Add to vector store
                self.add_articles_to_vector_store(new_articles)
                
                # Step 4: Try vector store again
                can_answer, chunks = self.can_answer_from_vector_store(query, conversation_history)
            else:
                print("❌ No new relevant articles found")
        
        # Step 5: Generate answer
        answer = self.generate_answer(query, chunks, conversation_history)
        
        # Prepare sources - deduplicate by URL to avoid showing same article multiple times
        sources = []
        seen_urls = set()
        for chunk in chunks:
            url = chunk.get('url', '')
            if url and url not in seen_urls:
                source_info = {
                    'title': chunk.get('title', 'Unknown'),
                    'url': url,
                    'chunk_info': chunk.get('source', '')
                }
                sources.append(source_info)
                seen_urls.add(url)
        
        result = {
            'question': query,
            'answer': answer,
            'sources': sources,
            'chunks_used': len(chunks)
        }
        
        print("=" * 60)
        print("✅ Answer generated!")
        print(f"📊 Used {len(chunks)} chunks from {len(sources)} sources")
        
        return result
    
    def clear_memory(self):
        """Clear vector store and all indexed URLs."""
        self.vector_store.clear()
        print("🗑️ Memory cleared")
    
    def get_stats(self):
        """Get current statistics."""
        return self.vector_store.get_stats()


def main():
    """Interactive demo."""
    agent = SimpleWikipediaAgent()
    
    print("🤖 Simple Wikipedia Q&A Agent")
    print("Commands: 'quit' to exit, 'clear' to clear memory, 'stats' for statistics")
    print("-" * 60)
    
    while True:
        try:
            query = input("\n❓ Ask a question: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                break
            elif query.lower() == 'clear':
                agent.clear_memory()
                continue
            elif query.lower() == 'stats':
                stats = agent.get_stats()
                print(f"📊 Stats: {stats['total_articles']} articles, {stats['total_chunks']} chunks")
                continue
            elif not query:
                continue
            
            result = agent.ask(query)
            print(f"\n💬 Answer:\n{result['answer']}")
            
            if result['sources']:
                print(f"\n📚 Sources:")
                for i, source in enumerate(result['sources'], 1):
                    print(f"  {i}. {source['title']}")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
    
    print("\n👋 Goodbye!")


if __name__ == "__main__":
    main()
