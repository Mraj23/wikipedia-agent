# Simple Wikipedia Q&A Agent

A clean, simple Wikipedia-powered question answering agent. No over-engineering, just a straightforward workflow that works.

## Workflow

```
Query → Rewrite Query → Search Wikipedia → Filter Articles → Create Embeddings → Retrieve Chunks → Answer Question
                                                                                      ↓
                                                                              Can't Answer? → Search More
```

## Features

- 🔍 **Smart Query Rewriting**: Converts questions into Wikipedia search terms
- 📚 **Relevance Filtering**: Uses AI to filter out irrelevant articles  
- 🧠 **Vector Store**: Embeddings-based retrieval for better context
- 🔄 **Iterative Search**: Searches more if initial results insufficient
- 📊 **URL Deduplication**: Never processes the same article twice

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set API Key**:
   Create a `.env` file:
   ```
   ANTHROPIC_API_KEY=your_api_key_here
   ```

3. **Run the Agent**:
   ```bash
   python simple_agent.py
   ```

## Usage

```bash
$ python simple_agent.py

🤖 Simple Wikipedia Q&A Agent
Commands: 'quit' to exit, 'clear' to clear memory, 'stats' for statistics
------------------------------------------------------------

❓ Ask a question: How does aluminum compare to steel in terms of strength?

🔍 Original query: How does aluminum compare to steel in terms of strength?
📝 Search terms: ['Aluminum', 'Steel', 'Material strength comparison']
📚 Found 6 unique articles
📄 Retrieved 5 full articles
✅ Keeping: Aluminium
✅ Keeping: Steel
❌ Filtering out: Iron
✅ Keeping: Ultimate tensile strength
✅ Filtered to 3 relevant articles
📚 Added 45 chunks to vector store
📊 Vector store check: ✅ Sufficient

============================================================
✅ Answer generated!
📊 Used 5 chunks from 3 sources

💬 Answer:
[Detailed comparison of aluminum vs steel strength properties...]

📚 Sources:
  1. Aluminium
  2. Steel  
  3. Ultimate tensile strength
```

## Commands

- Type any question to get an answer
- `stats` - Show vector store statistics
- `clear` - Clear memory and start fresh
- `quit` - Exit the program

## Project Structure

```
wikipedia-agent/
├── simple_agent.py           # Main simple agent
├── config.py                 # Configuration
├── requirements.txt          # Dependencies
├── tools/
│   ├── wikipedia_search.py   # Simple Wikipedia search
│   └── vector_store.py       # Vector embeddings store
└── Readme.md                 # This file
```

## How It Works

1. **Query Rewriting**: Uses Claude to convert user questions into Wikipedia search terms
2. **Wikipedia Search**: Searches for articles using Wikipedia's API
3. **Relevance Filtering**: Uses Claude to filter out irrelevant articles
4. **Vector Embeddings**: Creates embeddings from article content for semantic search
5. **Retrieval**: Finds most relevant chunks using similarity search
6. **Answer Generation**: Uses Claude to generate answers from retrieved chunks
7. **Iterative Search**: If answer is insufficient, searches for more articles

## Why Simple?

- **No complex workflows**: Straight-forward linear process
- **No over-engineering**: Removed link extraction, complex graphs, etc.
- **No unnecessary abstractions**: Direct, readable code
- **Easy to understand**: Clear workflow, minimal dependencies
- **Easy to modify**: Simple structure makes changes easy

The previous system was overly complex with link extraction, complex relevance checking, and multiple agents. This version does exactly what's needed and nothing more.
