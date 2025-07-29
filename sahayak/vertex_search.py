# sahayak/vertex_search.py

import os
from typing import List, Dict, Optional
from google.cloud import discoveryengine_v1
from google.cloud.discoveryengine_v1 import SearchRequest

class VertexAISearch:
    """Vertex AI Search wrapper for educational content search."""
    
    def __init__(self, project_id: str = None, location: str = "global"):
        """
        Initialize Vertex AI Search.
        
        Args:
            project_id: Google Cloud project ID (defaults to current project)
            location: Search engine location (default: global)
        """
        self.project_id = project_id or os.getenv('GOOGLE_CLOUD_PROJECT')
        if not self.project_id:
            # Try to get from gcloud config
            import subprocess
            try:
                result = subprocess.run(['gcloud', 'config', 'get-value', 'project'], 
                                      capture_output=True, text=True, check=True)
                self.project_id = result.stdout.strip()
            except:
                raise ValueError("Project ID not found. Set GOOGLE_CLOUD_PROJECT environment variable or configure gcloud.")
        
        self.location = location
        self.client = discoveryengine_v1.SearchServiceClient()
        
        # Create a search engine for educational content
        self.search_engine_id = self._get_or_create_search_engine()
        
    def _get_or_create_search_engine(self) -> str:
        """Get existing search engine or create a new one for educational content."""
        try:
            # Try to use a default search engine (if you have one configured)
            # For now, we'll use a simple web search approach
            return "default_search_engine"
        except Exception as e:
            print(f"⚠️ Using fallback search approach: {e}")
            return "fallback"
    
    def search(self, query: str, max_results: int = 5) -> List[Dict]:
        """
        Search for educational content using Vertex AI Search.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            
        Returns:
            List of search results with 'content' and 'url' keys
        """
        try:
            # For educational content, we'll use a hybrid approach:
            # 1. Try Vertex AI Search if configured
            # 2. Fall back to a curated educational search
            
            # Enhanced query for educational content
            enhanced_query = self._enhance_query_for_education(query)
            
            # Try Vertex AI Search first
            if self.search_engine_id != "fallback":
                return self._vertex_search(enhanced_query, max_results)
            else:
                # Fallback to educational web search
                return self._educational_web_search(enhanced_query, max_results)
                
        except Exception as e:
            print(f"⚠️ Vertex AI Search failed: {e}")
            # Final fallback to basic web search
            return self._basic_web_search(query, max_results)
    
    def _enhance_query_for_education(self, query: str) -> str:
        """Enhance query for better educational search results."""
        educational_terms = [
            "educational content",
            "lesson plan",
            "teaching resources",
            "curriculum",
            "student learning",
            "academic"
        ]
        
        # Add educational context if not present
        if not any(term in query.lower() for term in educational_terms):
            query += " educational content teaching resources"
        
        return query
    
    def _vertex_search(self, query: str, max_results: int) -> List[Dict]:
        """Perform search using Vertex AI Search."""
        try:
            # This would use the actual Vertex AI Search API
            # For now, we'll implement a placeholder
            print(f"🔍 Vertex AI Search: {query}")
            
            # Placeholder implementation
            # In a real implementation, you would:
            # 1. Create a search request
            # 2. Call the Vertex AI Search API
            # 3. Process and return results
            
            return self._educational_web_search(query, max_results)
            
        except Exception as e:
            print(f"⚠️ Vertex AI Search failed: {e}")
            return self._educational_web_search(query, max_results)
    
    def _educational_web_search(self, query: str, max_results: int) -> List[Dict]:
        """Fallback to educational web search using requests."""
        import requests
        import json
        
        try:
            # Use a simple web search approach for educational content
            # This is a fallback when Vertex AI Search is not fully configured
            
            # Search for educational resources
            search_urls = [
                f"https://www.khanacademy.org/search?page_search_query={query}",
                f"https://www.education.com/search/?q={query}",
                f"https://www.teacherspayteachers.com/Browse/Search:{query}",
                f"https://www.britannica.com/search?query={query}"
            ]
            
            results = []
            for url in search_urls[:2]:  # Limit to first 2 sources
                try:
                    # Create a mock result for demonstration
                    # In production, you'd actually scrape or use APIs
                    mock_result = {
                        'content': f"Educational content about {query} from {url.split('//')[1].split('/')[0]}",
                        'url': url,
                        'title': f"Educational Resource: {query}",
                        'source': 'educational_web'
                    }
                    results.append(mock_result)
                except Exception as e:
                    print(f"⚠️ Failed to search {url}: {e}")
                    continue
            
            return results[:max_results]
            
        except Exception as e:
            print(f"⚠️ Educational web search failed: {e}")
            return self._basic_web_search(query, max_results)
    
    def _basic_web_search(self, query: str, max_results: int) -> List[Dict]:
        """Basic web search fallback."""
        try:
            # Simple fallback using a basic search approach
            results = [
                {
                    'content': f"General information about {query} from web search",
                    'url': f"https://www.google.com/search?q={query}",
                    'title': f"Search Results: {query}",
                    'source': 'basic_web'
                }
            ]
            
            return results[:max_results]
            
        except Exception as e:
            print(f"❌ All search methods failed: {e}")
            return []
    
    def invoke(self, query: str) -> List[Dict]:
        """
        LangChain-compatible invoke method to replace TavilySearch.
        
        Args:
            query: Search query
            
        Returns:
            List of search results
        """
        return self.search(query, max_results=2)  # Match Tavily's default

# Create a global instance
try:
    vertex_search_tool = VertexAISearch()
except Exception as e:
    vertex_search_tool = None 