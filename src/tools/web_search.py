import urllib.request
import urllib.parse
import json

def search_wikipedia(query: str, limit: int = 2) -> str:
    """
    Independent Web Search using Wikipedia's public open API.
    Provides agents with real-world knowledge on demand.
    """
    encoded_query = urllib.parse.quote(query)
    url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={encoded_query}&utf8=&format=json"
    
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'IndependentGenAI/1.0'})
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode('utf-8'))
            
        results = data.get('query', {}).get('search', [])
        if not results:
            return "No results found on Wikipedia."
            
        snippets = []
        for i, res in enumerate(results[:limit]):
            title = res.get('title', '')
            # Strip basic HTML tags from snippet
            snippet = res.get('snippet', '').replace('<span class="searchmatch">', '').replace('</span>', '')
            snippets.append(f"[{title}]: {snippet}")
            
        return " | ".join(snippets)
    except Exception as e:
        return f"Web search failed: {str(e)}"

def search_web(query: str) -> str:
    """Main search entrypoint for Swarm Agents"""
    # For now, routes to Wikipedia as a reliable source of facts.
    return search_wikipedia(query)
