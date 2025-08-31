"""
Test suite for the Simple Wikipedia Agent
"""

import unittest
from unittest.mock import Mock, patch
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from simple_agent import SimpleWikipediaAgent
from tools.wikipedia_search import WikipediaSearcher


class TestWikipediaSearcher(unittest.TestCase):
    def setUp(self):
        self.searcher = WikipediaSearcher()
    
    def test_rewrite_query_for_wikipedia(self):
        """Test query rewriting functionality."""
        query = "How strong is aluminum compared to steel?"
        
        with patch.object(self.searcher, 'llm') as mock_llm:
            mock_response = Mock()
            mock_response.content = "Aluminum\nSteel\nMaterial strength"
            mock_llm.return_value = mock_response
            
            search_terms = self.searcher.rewrite_query_for_wikipedia(query)
            
            self.assertIsInstance(search_terms, list)
            self.assertLessEqual(len(search_terms), 3)
            mock_llm.assert_called()
    
    def test_search_wikipedia_titles(self):
        """Test Wikipedia title search."""
        # This is an integration test that hits the actual Wikipedia API
        titles = self.searcher.search_wikipedia_titles("Aluminum", limit=3)
        
        self.assertIsInstance(titles, list)
        self.assertLessEqual(len(titles), 3)
        # Should find at least one result for "Aluminum"
        if titles:
            self.assertTrue(any("aluminum" in title.lower() or "aluminium" in title.lower() for title in titles))
    
    def test_get_articles(self):
        """Test article retrieval."""
        test_titles = ["Aluminium", "Steel"]
        articles = self.searcher.get_articles(test_titles)
        
        self.assertIsInstance(articles, list)
        for article in articles:
            self.assertIn('title', article)
            self.assertIn('summary', article)
            self.assertIn('content', article)
            self.assertIn('url', article)
            # Content should be substantial (not a stub)
            self.assertGreater(len(article['content']), 100)
    
    def test_filter_relevant_articles(self):
        """Test relevance filtering."""
        mock_articles = [
            {
                'title': 'Aluminium',
                'summary': 'Aluminium is a chemical element with symbol Al and atomic number 13.',
                'content': 'Long content about aluminum...',
                'url': 'https://en.wikipedia.org/wiki/Aluminium'
            },
            {
                'title': 'Cat',
                'summary': 'The domestic cat is a small, typically furry, carnivorous mammal.',
                'content': 'Long content about cats...',
                'url': 'https://en.wikipedia.org/wiki/Cat'
            }
        ]
        
        query = "What are the properties of aluminum?"
        
        with patch.object(self.searcher, 'llm') as mock_llm:
            # Mock responses: first call returns RELEVANT, second returns NOT_RELEVANT
            mock_llm.side_effect = [
                Mock(content="RELEVANT"),
                Mock(content="NOT_RELEVANT")
            ]
            
            relevant_articles = self.searcher.filter_relevant_articles(mock_articles, query)
            
            self.assertEqual(len(relevant_articles), 1)
            self.assertEqual(relevant_articles[0]['title'], 'Aluminium')
            self.assertEqual(mock_llm.call_count, 2)


class TestSimpleWikipediaAgent(unittest.TestCase):
    def setUp(self):
        self.agent = SimpleWikipediaAgent()
    
    def test_initialization(self):
        """Test agent initialization."""
        self.assertIsNotNone(self.agent.llm)
        self.assertIsNotNone(self.agent.wikipedia_searcher)
        self.assertIsNotNone(self.agent.vector_store)
        self.assertEqual(len(self.agent.retrieved_urls), 0)
    
    def test_can_answer_from_vector_store_empty(self):
        """Test vector store check when empty."""
        can_answer, chunks = self.agent.can_answer_from_vector_store("What is aluminum?")
        
        self.assertFalse(can_answer)
        self.assertEqual(len(chunks), 0)
    
    def test_generate_answer_no_chunks(self):
        """Test answer generation with no chunks."""
        answer = self.agent.generate_answer("What is aluminum?", [])
        
        self.assertIn("don't have enough information", answer.lower())
    
    def test_generate_answer_with_chunks(self):
        """Test answer generation with chunks."""
        mock_chunks = [
            {
                'content': 'Aluminum is a lightweight metal with excellent corrosion resistance.',
                'source': 'Aluminium (chunk 1)',
                'title': 'Aluminium',
                'url': 'https://en.wikipedia.org/wiki/Aluminium'
            }
        ]
        
        with patch.object(self.agent, 'llm') as mock_llm:
            mock_llm.return_value.content = "Aluminum is a lightweight metal known for its excellent corrosion resistance and wide range of applications."
            
            answer = self.agent.generate_answer("What is aluminum?", mock_chunks)
            
            self.assertIsInstance(answer, str)
            self.assertGreater(len(answer), 10)
            mock_llm.assert_called()
    
    def test_add_articles_to_vector_store(self):
        """Test adding articles to vector store."""
        mock_articles = [
            {
                'title': 'Test Article',
                'summary': 'Test summary',
                'content': 'Test content for the article',
                'url': 'https://test.com'
            }
        ]
        
        with patch.object(self.agent.vector_store, 'add_articles') as mock_add:
            mock_add.return_value = 5
            
            chunks_added = self.agent.add_articles_to_vector_store(mock_articles)
            
            self.assertEqual(chunks_added, 5)
            mock_add.assert_called_once_with(mock_articles)
    
    def test_clear_memory(self):
        """Test memory clearing."""
        # Add some URLs to simulate usage
        self.agent.retrieved_urls.add("https://test1.com")
        self.agent.retrieved_urls.add("https://test2.com")
        
        with patch.object(self.agent.vector_store, 'clear') as mock_clear:
            self.agent.clear_memory()
            
            self.assertEqual(len(self.agent.retrieved_urls), 0)
            mock_clear.assert_called_once()
    
    def test_get_stats(self):
        """Test statistics retrieval."""
        with patch.object(self.agent.vector_store, 'get_stats') as mock_stats:
            mock_stats.return_value = {
                'total_articles': 5,
                'total_chunks': 25,
                'indexed_urls': ['url1', 'url2']
            }
            
            stats = self.agent.get_stats()
            
            self.assertIn('total_articles', stats)
            self.assertIn('total_chunks', stats)
            self.assertIn('retrieved_urls_count', stats)
            self.assertEqual(stats['retrieved_urls_count'], len(self.agent.retrieved_urls))


class TestIntegration(unittest.TestCase):
    """Integration tests that test the full workflow."""
    
    def setUp(self):
        self.agent = SimpleWikipediaAgent()
    
    def test_full_workflow_simple_query(self):
        """Test the complete workflow with a simple query."""
        # This is a real integration test - it will make actual API calls
        # Skip if no API key is available
        try:
            import config
            if not config.ANTHROPIC_API_KEY:
                self.skipTest("No API key available")
        except:
            self.skipTest("No API key available")
        
        query = "What is aluminum?"
        
        # Clear any existing state
        self.agent.clear_memory()
        
        # Run the query
        result = self.agent.ask(query)
        
        # Check the result structure
        self.assertIn('question', result)
        self.assertIn('answer', result)
        self.assertIn('sources', result)
        self.assertIn('chunks_used', result)
        
        self.assertEqual(result['question'], query)
        self.assertIsInstance(result['answer'], str)
        self.assertGreater(len(result['answer']), 50)  # Should be a substantial answer
        self.assertIsInstance(result['sources'], list)
        self.assertGreater(result['chunks_used'], 0)
        
        # Should have found some sources
        if result['sources']:
            for source in result['sources']:
                self.assertIn('title', source)
                self.assertIn('url', source)


def run_quick_test():
    """Run a quick test to verify the system works."""
    print("🧪 Running Quick Test of Simple Wikipedia Agent")
    print("=" * 60)
    
    try:
        # Test basic imports
        print("✅ Testing imports...")
        agent = SimpleWikipediaAgent()
        searcher = WikipediaSearcher()
        print("✅ Imports successful")
        
        # Test query rewriting
        print("✅ Testing query rewriting...")
        with patch.object(searcher, 'llm') as mock_llm:
            mock_response = Mock()
            mock_response.content = "Aluminum\nSteel\nMaterials"
            mock_llm.return_value = mock_response
            terms = searcher.rewrite_query_for_wikipedia("How strong is aluminum?")
            print(f"✅ Query rewriting works: {terms}")
        
        # Test Wikipedia search (real API call)
        print("✅ Testing Wikipedia search...")
        titles = searcher.search_wikipedia_titles("Aluminum", limit=2)
        print(f"✅ Wikipedia search works: Found {len(titles)} titles")
        
        # Test vector store
        print("✅ Testing vector store...")
        stats = agent.get_stats()
        print(f"✅ Vector store works: {stats}")
        
        print("\n🎉 All quick tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Quick test failed: {e}")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test the Simple Wikipedia Agent')
    parser.add_argument('--quick', action='store_true', help='Run quick test only')
    parser.add_argument('--integration', action='store_true', help='Run integration tests (requires API key)')
    
    args = parser.parse_args()
    
    if args.quick:
        run_quick_test()
    else:
        # Run full test suite
        if args.integration:
            # Run all tests including integration
            unittest.main(argv=[''], exit=False)
        else:
            # Run unit tests only (skip integration tests)
            loader = unittest.TestLoader()
            suite = unittest.TestSuite()
            
            # Add unit test classes
            suite.addTests(loader.loadTestsFromTestCase(TestWikipediaSearcher))
            suite.addTests(loader.loadTestsFromTestCase(TestSimpleWikipediaAgent))
            
            runner = unittest.TextTestRunner(verbosity=2)
            runner.run(suite)
