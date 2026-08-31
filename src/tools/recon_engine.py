import urllib.request
import urllib.parse
import urllib.error
import json
import re
import random
import time
import functools

try:
    from playwright.sync_api import sync_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False


def _retry_with_backoff(retries=3, backoff_in_seconds=2):
    """Smart Retry Decorator with Exponential Backoff"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            x = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if isinstance(e, urllib.error.HTTPError) and e.code == 404:
                        raise e
                    if x == retries:
                        return f"Failed after {retries} retries. Error: {str(e)}"
                    sleep_time = (backoff_in_seconds * 2 ** x) + random.uniform(0, 1)
                    time.sleep(sleep_time)
                    x += 1
        return wrapper
    return decorator


class UnrestrictedAgentReconEngine:
    """Autonomous information gathering engine providing live web search, encyclopedic facts, and network reconnaissance."""
    
    USER_AGENTS = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
        "Mozilla/5.0 (X11; Linux x86_64; rv:123.0) Gecko/20100101 Firefox/123.0"
    ]
    
    PROXIES = []

    def _get_headers(self) -> dict:
        """Generates random headers to bypass basic bot/anti-scraping checks."""
        return {
            'User-Agent': random.choice(self.USER_AGENTS),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5'
        }
        
    def _get_opener(self):
        """Returns a URL opener with optional proxy rotation"""
        if self.PROXIES:
            proxy = random.choice(self.PROXIES)
            proxy_handler = urllib.request.ProxyHandler({'http': proxy, 'https': proxy})
            return urllib.request.build_opener(proxy_handler)
        return urllib.request.build_opener()

    def _clean_html(self, raw_html: str) -> str:
        """Strips HTML tags, scripts, and compresses extra whitespace."""
        clean = re.sub(r'<(script|style)[^>]*>.*?</\1>', '', raw_html, flags=re.DOTALL | re.IGNORECASE)
        clean = re.sub(r'<[^>]+>', ' ', clean)
        return re.sub(r'\s+', ' ', clean).strip()

    @_retry_with_backoff(retries=3, backoff_in_seconds=2)
    def search_web(self, query: str, limit: int = 3) -> str:
        """Performs real-time web search for live news, articles, and websites."""
        encoded_query = urllib.parse.quote(query)
        url = f"https://html.duckduckgo.com/html/?q={encoded_query}"
        
        req = urllib.request.Request(url, headers=self._get_headers())
        opener = self._get_opener()
        with opener.open(req, timeout=6) as response:
            html = response.read().decode('utf-8', errors='ignore')
            
        snippets = re.findall(r'<a class="result__snippet[^>]*>(.*?)</a>', html, re.DOTALL)
        titles = re.findall(r'<a class="result__url[^>]*>(.*?)</a>', html, re.DOTALL)
        
        results = []
        for i in range(min(limit, len(snippets))):
            t = self._clean_html(titles[i]) if i < len(titles) else "Link"
            s = self._clean_html(snippets[i])
            results.append(f"[{t}]: {s}")
            
        return " | ".join(results) if results else "No DuckDuckGo results found."

    @_retry_with_backoff(retries=2, backoff_in_seconds=1)
    def search_knowledge_base(self, query: str, limit: int = 2) -> str:
        """Fetches structured encyclopedic facts, history, or technical terms."""
        encoded_query = urllib.parse.quote(query)
        url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={encoded_query}&utf8=&format=json"
        
        req = urllib.request.Request(url, headers=self._get_headers())
        opener = self._get_opener()
        with opener.open(req, timeout=5) as response:
            data = json.loads(response.read().decode('utf-8'))
            
        results = data.get('query', {}).get('search', [])
        if not results:
            return "No Knowledge Base results."
            
        snippets = [f"[{res.get('title')}]: {self._clean_html(res.get('snippet'))}" for res in results[:limit]]
        return " | ".join(snippets)

    @_retry_with_backoff(retries=2, backoff_in_seconds=2)
    def read_full_page(self, url: str, max_chars: int = 3000) -> str:
        """Reads ANY webpage. Falls back to Wayback Machine on 404."""
        try:
            req = urllib.request.Request(url, headers=self._get_headers())
            opener = self._get_opener()
            with opener.open(req, timeout=8) as response:
                html = response.read().decode('utf-8', errors='ignore')
                
            text = self._clean_html(html)
            if len(text) > max_chars:
                return text[:max_chars] + f"\n... [Truncated: Total length {len(text)} chars]"
            return text
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return self.search_wayback_machine(url, max_chars)
            raise e

    def search_wayback_machine(self, url: str, max_chars: int = 3000) -> str:
        """Fetches the last known archived version of a deleted/404 URL."""
        api_url = f"http://archive.org/wayback/available?url={urllib.parse.quote(url)}"
        try:
            req = urllib.request.Request(api_url, headers=self._get_headers())
            opener = self._get_opener()
            with opener.open(req, timeout=8) as response:
                data = json.loads(response.read().decode('utf-8'))
                
            snapshots = data.get("archived_snapshots", {})
            if "closest" in snapshots and snapshots["closest"]["available"]:
                archive_url = snapshots["closest"]["url"]
                
                req_arc = urllib.request.Request(archive_url, headers=self._get_headers())
                with opener.open(req_arc, timeout=10) as arc_resp:
                    html = arc_resp.read().decode('utf-8', errors='ignore')
                    
                text = f"[ARCHIVED VERSION RECOVERED]\n" + self._clean_html(html)
                if len(text) > max_chars:
                    return text[:max_chars] + f"\n... [Truncated]"
                return text
            return "No archived version found on Wayback Machine."
        except Exception as e:
            return f"Wayback Machine query failed: {str(e)}"

    def read_dynamic_page(self, url: str, max_chars: int = 3000) -> str:
        """Uses Playwright to render JavaScript-heavy SPAs."""
        if not PLAYWRIGHT_AVAILABLE:
            return "Playwright is not installed. Run: pip install playwright && playwright install"
            
        try:
            with sync_playwright() as p:
                browser_kwargs = {"headless": True}
                if self.PROXIES:
                    browser_kwargs["proxy"] = {"server": random.choice(self.PROXIES)}
                    
                browser = p.chromium.launch(**browser_kwargs)
                context = browser.new_context(user_agent=random.choice(self.USER_AGENTS))
                page = context.new_page()
                page.goto(url, wait_until="networkidle", timeout=15000)
                
                text = page.evaluate("document.body.innerText")
                browser.close()
                
                if len(text) > max_chars:
                    return text[:max_chars] + f"\n... [Truncated]"
                return text
        except Exception as e:
            return f"Headless browser failed: {str(e)}"

    @_retry_with_backoff(retries=2, backoff_in_seconds=2)
    def search_subdomains(self, domain: str) -> str:
        """Discovers public subdomains using Certificate Transparency logs (crt.sh)."""
        url = f"https://crt.sh/?q=%25.{urllib.parse.quote(domain)}&output=json"
        req = urllib.request.Request(url, headers=self._get_headers())
        opener = self._get_opener()
        with opener.open(req, timeout=8) as response:
            data = json.loads(response.read().decode('utf-8'))
            
        subdomains = set(entry['name_value'] for entry in data[:15])
        return f"Discovered Subdomains for {domain}: " + ", ".join(subdomains)

    @_retry_with_backoff(retries=2, backoff_in_seconds=2)
    def search_github_code(self, query: str) -> str:
        """Searches GitHub public repositories for relevant code/PoCs."""
        encoded_query = urllib.parse.quote(query)
        url = f"https://api.github.com/search/repositories?q={encoded_query}&sort=stars&order=desc"
        req = urllib.request.Request(url, headers=self._get_headers())
        opener = self._get_opener()
        with opener.open(req, timeout=6) as response:
            data = json.loads(response.read().decode('utf-8'))
            
        items = data.get('items', [])[:3]
        results = [f"[{repo['full_name']}]: {repo['description']} (URL: {repo['html_url']})" for repo in items]
        return " | ".join(results) if results else "No GitHub repos found."

    def scan_ports(self, target_ip: str, ports: list = [21, 22, 80, 443, 3306, 8080]) -> str:
        """Executes a real TCP connect scan against the target IP."""
        import socket
        import concurrent.futures
        
        open_ports = []
        
        def scan_single_port(port):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(0.5)
                result = s.connect_ex((target_ip, port))
                if result == 0:
                    open_ports.append(port)
                    
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            executor.map(scan_single_port, ports)
            
        if open_ports:
            return f"Open ports found on {target_ip}: {sorted(open_ports)}"
        return f"No open ports detected on {target_ip} (Filtered or Down)."

    def autonomous_search(self, query: str) -> str:
        """Main interface for the AI Agent. Tries Web Search first -> Falls back to Knowledge Base if empty/failed."""
        web_res = self.search_web(query)
        
        if "Error" in web_res or "Failed" in web_res or "No DuckDuckGo" in web_res:
            wiki_res = self.search_knowledge_base(query)
            return f"[Primary Engine Failed. Fallback Used]: {wiki_res}"
            
        return f"[Live Web Intelligence]: {web_res}"