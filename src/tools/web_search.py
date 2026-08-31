import urllib.request
import urllib.parse
import json

def search_wikipedia(query: str, limit: int = 2) -> str:
    """Independent Web Search using Wikipedia's public open API. Provides agents with real-world knowledge on demand."""
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
            snippet = res.get('snippet', '').replace('<span class="searchmatch">', '').replace('</span>', '')
            snippets.append(f"[{title}]: {snippet}")
            
        return " | ".join(snippets)
    except Exception as e:
        return f"Web search failed: {str(e)}"

def search_web(query: str) -> str:
    """Main multi-source web search entrypoint for Swarm Agents."""
    try:
        from src.tools.recon_engine import UnrestrictedAgentReconEngine
        recon = UnrestrictedAgentReconEngine()
        res = recon.autonomous_search(query)
        if res and not res.startswith("No results"):
            return res
    except Exception:
        pass
    return search_wikipedia(query)

