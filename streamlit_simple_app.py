#!/usr/bin/env python3
"""
Simple Streamlit UI for the Simple Wikipedia Agent
Run with: streamlit run streamlit_simple_app.py
"""

import streamlit as st
import time
from simple_agent import SimpleWikipediaAgent

# Page config
st.set_page_config(
    page_title="Simple Wikipedia Q&A",
    page_icon="🤖",
    layout="wide"
)

# Initialize the agent
@st.cache_resource
def get_agent():
    """Initialize and cache the simple agent."""
    try:
        return SimpleWikipediaAgent()
    except Exception as e:
        st.error(f"Failed to initialize agent: {e}")
        return None

def stream_agent_process(agent, query, conversation_history=None):
    """Show the agent's process step by step."""
    status_placeholder = st.empty()
    progress_bar = st.progress(0)
    
    # Step 1: Check vector store
    status_placeholder.info("🔍 Checking existing knowledge...")
    progress_bar.progress(10)
    time.sleep(0.5)
    
    can_answer, chunks = agent.can_answer_from_vector_store(query, conversation_history)
    
    if not can_answer:
        # Step 2: Search new articles
        status_placeholder.info("🌐 Searching Wikipedia for new articles...")
        progress_bar.progress(30)
        time.sleep(0.5)
        
        new_articles = agent.search_new_articles(query, conversation_history)
        
        if new_articles:
            status_placeholder.success(f"📚 Found {len(new_articles)} relevant articles")
            progress_bar.progress(60)
            time.sleep(0.5)
            
            # Step 3: Add to vector store
            status_placeholder.info("💾 Adding articles to knowledge base...")
            progress_bar.progress(80)
            agent.add_articles_to_vector_store(new_articles)
            time.sleep(0.5)
            
            # Step 4: Re-check vector store
            can_answer, chunks = agent.can_answer_from_vector_store(query, conversation_history)
        else:
            status_placeholder.warning("❌ No relevant articles found")
    
    # Step 5: Generate answer
    status_placeholder.info("🤔 Generating answer...")
    progress_bar.progress(90)
    time.sleep(0.5)
    
    answer = agent.generate_answer(query, chunks, conversation_history)
    
    # Complete
    status_placeholder.success("✅ Answer ready!")
    progress_bar.progress(100)
    time.sleep(0.5)
    
    # Clear status
    status_placeholder.empty()
    progress_bar.empty()
    
    return answer, chunks

def main():
    st.title("🤖 Simple Wikipedia Q&A Agent")
    st.markdown("*Ask any question and get answers from Wikipedia with full transparency*")
    
    # Check if agent can be initialized
    agent = get_agent()
    if not agent:
        st.error("❌ Could not initialize the agent. Please check your API key in the .env file.")
        st.stop()
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Create layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("💬 Chat")
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message["role"] == "assistant" and "sources" in message:
                    with st.expander(f"📚 Sources ({len(message['sources'])})"):
                        for i, source in enumerate(message["sources"], 1):
                            st.markdown(f"{i}. **{source['title']}**")
                            if source.get('url'):
                                st.markdown(f"   🔗 [{source['url']}]({source['url']})")
        
        # Chat input
        if prompt := st.chat_input("Ask any question (e.g., 'What is aluminum?' or 'How does steel compare to iron?')"):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Get assistant response
            with st.chat_message("assistant"):
                try:
                    # Show the process
                    answer, chunks = stream_agent_process(agent, prompt, st.session_state.messages)
                    
                    # Display answer
                    st.markdown(answer)
                    
                    # Prepare sources - deduplicate by URL
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
                    
                    # Show sources in expandable section
                    if sources:
                        with st.expander(f"📚 Sources ({len(sources)})"):
                            for i, source in enumerate(sources, 1):
                                st.markdown(f"{i}. **{source['title']}**")
                                if source.get('url'):
                                    st.markdown(f"   🔗 [{source['url']}]({source['url']})")
                    
                    # Add to session state
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer,
                        "sources": sources
                    })
                    
                except Exception as e:
                    error_msg = f"❌ Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": error_msg
                    })
    
    with col2:
        st.subheader("📊 Agent Stats")
        
        # Get and display stats
        stats = agent.get_stats()
        
        st.metric("Articles", stats['total_articles'])
        st.metric("Knowledge Chunks", stats['total_chunks'])
        st.metric("URLs Indexed", len(stats.get('indexed_urls', [])))
        
        # Show some indexed articles
        if stats['indexed_urls']:
            with st.expander("📚 Knowledge Base"):
                for i, url in enumerate(stats['indexed_urls'][:5], 1):
                    title = url.split('/')[-1].replace('_', ' ')
                    st.markdown(f"{i}. [{title}]({url})")
                if len(stats['indexed_urls']) > 5:
                    st.markdown(f"... and {len(stats['indexed_urls']) - 5} more")
        
        st.markdown("---")
        
        st.subheader("🔧 Controls")
        
        if st.button("🗑️ Clear Memory", type="secondary"):
            agent.clear_memory()
            st.session_state.messages = []
            st.success("Memory cleared!")
            st.rerun()
        
        if st.button("📊 Refresh Stats", type="secondary"):
            st.rerun()
        

if __name__ == "__main__":
    main()
