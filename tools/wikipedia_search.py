import wikipediaapi
import requests
import time
from typing import List, Dict, Optional
from langchain_anthropic import ChatAnthropic
from langchain.schema import HumanMessage
import config


class WikipediaSearcher:
    def __init__(self):
        self.wiki = wikipediaapi.Wikipedia(
            language='en',
            extract_format=wikipediaapi.ExtractFormat.WIKI,
            user_agent='MaterialAgentQ/1.0 (rajmehta@wikipedia-agent)'
        )
        # Use session for better connection handling (like the demo code)
        self.session = requests.Session()
        # Set User-Agent header for the session to avoid 403 errors
        self.session.headers.update({
            'User-Agent': 'MaterialQA/1.0 (https://github.com/mraj23/wikipedia-agent; contact@example.com)'
        })
        self.search_api_url = "https://en.wikipedia.org/w/api.php"
        
        # Initialize LLM for query rewriting and relevance filtering
        self.llm = ChatAnthropic(
            model="claude-3-5-sonnet-20241022",
            api_key=config.ANTHROPIC_API_KEY,
            temperature=0.1
        )
    
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
    
    def rewrite_query_for_wikipedia(self, query: str, existing_articles: List[str] = None, conversation_history: Optional[List[Dict]] = None) -> List[str]:
        """Rewrite user query into Wikipedia search terms, considering existing knowledge."""
        
        # Build context about existing articles
        context_info = ""
        if existing_articles:
            context_info = f"""

Already indexed articles:
{chr(10).join([f"- {article}" for article in existing_articles[:10]])}
{"... and more" if len(existing_articles) > 10 else ""}

Generate search terms that will find NEW, complementary articles not already covered."""

        # Add conversation history context
        conversation_context = self._format_conversation_history(conversation_history)

        prompt = f"""Rewrite this question into 2-3 specific Wikipedia search terms that would find relevant articles:

Question: {query}{conversation_context}{context_info}

Return only the search terms, one per line, no explanations:"""

        response = self.llm.invoke([HumanMessage(content=prompt)])
        
        # Parse search terms
        search_terms = []
        for line in response.content.strip().split('\n'):
            term = line.strip()
            if term and not term.startswith('-') and not term.startswith('•'):
                search_terms.append(term)
        
        return search_terms[:3]  # Max 3 terms
    
    def search_wikipedia_titles(self, search_term: str, limit: int = 5) -> List[str]:
        """Search Wikipedia for article titles using session-based approach."""
        try:
            # Simple parameters like the demo code
            params = {
                "action": "query",
                "format": "json",
                "list": "search",
                "srsearch": search_term,
                "srlimit": limit,
                "srnamespace": 0
            }
            
            # Use session.get() like the demo
            response = self.session.get(url=self.search_api_url, params=params)
            data = response.json()
            
            titles = []
            if 'query' in data and 'search' in data['query']:
                for result in data['query']['search']:
                    titles.append(result['title'])
            
            print(f"🔍 Found {len(titles)} titles for '{search_term}'")
            return titles
            
        except Exception as e:
            print(f"❌ Search error for '{search_term}': {e}")
            # Fallback: try direct page lookup
            try:
                page = self.wiki.page(search_term)
                if page.exists():
                    print(f"📄 Found via direct lookup: {page.title}")
                    return [page.title]
            except:
                pass
            return []
    
    def get_articles(self, titles: List[str]) -> List[Dict[str, str]]:
        """Get full Wikipedia articles from titles."""
        articles = []
        
        for title in titles:
            try:
                page = self.wiki.page(title)
                if page.exists() and len(page.text) > 100:  # Skip stubs
                    articles.append({
                        'title': page.title,
                        'summary': page.summary,
                        'content': page.text,
                        'url': page.fullurl
                    })
            except Exception as e:
                print(f"Error getting article {title}: {e}")
                continue
        
        return articles
    
    def filter_relevant_articles(self, articles: List[Dict[str, str]], original_query: str, conversation_history: Optional[List[Dict]] = None) -> List[Dict[str, str]]:
        """Filter articles based on relevance to original query."""
        if not articles:
            return []
        
        relevant_articles = []
        
        # Add conversation history context
        conversation_context = self._format_conversation_history(conversation_history)
        
        for article in articles:
            # Create relevance check prompt
            prompt = f"""Original Question: {original_query}{conversation_context}

Article Title: {article['title']}
Article Summary: {article['summary'][:300]}

Is this article relevant for answering the original question, considering the conversation context?
Respond with only "RELEVANT" or "NOT_RELEVANT":"""

            try:
                response = self.llm.invoke([HumanMessage(content=prompt)])
                if "RELEVANT" in response.content.upper():
                    relevant_articles.append(article)
                    print(f"✅ Keeping: {article['title']}")
                else:
                    print(f"❌ Filtering out: {article['title']}")
            except Exception as e:
                print(f"Error checking relevance for {article['title']}: {e}")
                # If error, keep the article to be safe
                relevant_articles.append(article)
        
        return relevant_articles
    
    def fallback_search_by_keywords(self, query: str, max_articles: int = 3) -> List[Dict[str, str]]:
        """Fallback search using direct Wikipedia page lookup when API fails."""
        print("🔄 Using fallback search method...")
        
        # Extract potential article names from the query
        keywords = [
            "aluminum", "aluminium", "steel", "iron", "carbon", "titanium", 
            "copper", "bronze", "brass", "zinc", "nickel", "chromium",
            "material", "metal", "alloy", "properties", "strength"
        ]
        
        # Find keywords in the query
        query_lower = query.lower()
        found_keywords = [kw for kw in keywords if kw in query_lower]
        
        # Try to get articles for found keywords
        articles = []
        for keyword in found_keywords[:max_articles]:
            try:
                # Try different variations
                variations = [
                    keyword.capitalize(),
                    keyword.upper() if len(keyword) <= 3 else keyword.title(),
                    f"{keyword.title()} alloy" if keyword in ["aluminum", "steel", "titanium"] else None
                ]
                
                for variation in variations:
                    if not variation:
                        continue
                        
                    try:
                        page = self.wiki.page(variation)
                        if page.exists() and len(page.text) > 100:
                            articles.append({
                                'title': page.title,
                                'summary': page.summary,
                                'content': page.text,
                                'url': page.fullurl
                            })
                            print(f"📄 Found via fallback: {page.title}")
                            break  # Found one, move to next keyword
                    except:
                        continue
            except:
                continue
        
        return articles[:max_articles]
    
    def search_and_filter(self, query: str, max_articles: int = 3, existing_articles: List[str] = None, conversation_history: Optional[List[Dict]] = None) -> List[Dict[str, str]]:
        """Complete search workflow: rewrite -> search -> filter."""
        print(f"🔍 Original query: {query}")
        
        # Step 1: Rewrite query for Wikipedia search with context
        search_terms = self.rewrite_query_for_wikipedia(query, existing_articles, conversation_history)
        print(f"📝 Search terms: {search_terms}")
        if existing_articles:
            print(f"🧠 Context: {len(existing_articles)} existing articles considered")
        
        # Step 2: Search Wikipedia for each term
        all_titles = []
        api_working = True
        
        for term in search_terms:
            titles = self.search_wikipedia_titles(term, limit=3)
            if not titles and api_working:
                # If we get no results and haven't tried fallback yet
                print(f"⚠️ No results for '{term}', API might be blocked")
                api_working = False
            all_titles.extend(titles)
        
        # If API search failed completely, try fallback
        if not all_titles:
            print("🔄 API search failed, trying fallback method...")
            fallback_articles = self.fallback_search_by_keywords(query, max_articles)
            if fallback_articles:
                # Filter fallback articles for relevance
                relevant_articles = self.filter_relevant_articles(fallback_articles, query, conversation_history)
                print(f"✅ Fallback found {len(relevant_articles)} relevant articles")
                return relevant_articles
            else:
                print("❌ Both API and fallback search failed")
                return []
        
        # Remove duplicates while preserving order
        unique_titles = []
        seen = set()
        for title in all_titles:
            if title not in seen:
                unique_titles.append(title)
                seen.add(title)
        
        print(f"📚 Found {len(unique_titles)} unique articles")
        
        # Step 3: Get full articles
        articles = self.get_articles(unique_titles[:max_articles])
        print(f"📄 Retrieved {len(articles)} full articles")
        
        # Step 4: Filter for relevance
        relevant_articles = self.filter_relevant_articles(articles, query, conversation_history)
        print(f"✅ Filtered to {len(relevant_articles)} relevant articles")
        
        return relevant_articles
