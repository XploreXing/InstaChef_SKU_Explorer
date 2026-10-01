"""Shared Tavily API client with thread-safe concurrency control.

Provides a singleton TavilyClient and a Semaphore-limited tavily_search()
function. All agent tools (search_web, discover_cuisine) should call
tavily_search() instead of creating their own Tavily clients.
"""

import os
from dotenv import load_dotenv
import threading
from typing import Literal

load_dotenv()

_client_lock=threading.Lock()
_client_instance=None # 这一步是什么意思
_semaphore=threading.Semaphore(3) #control 只有3次的并发concurrency #用于控制一个API的调用速度，防止ratelimit，账号封禁

def get_tavily_client():
    """Thread-safe singleton Tavily client.

    Reads TAVILY_API_KEY from environment (matches config.yaml
    search.api_key_env default). Call this instead of creating
    TavilyClient directly.
    """
    global _client_instance #声明这个client instance是全局变量
    if _client_instance is None:
        with _client_lock:
            if _client_instance is None: #为什么在锁内还要再check一遍？
                from tavily import TavilyClient
                api_key=os.getenv('TAVILY_API_KEY')
                if not api_key:
                    raise ValueError('TAVILY API KEY is not set yet') 

                _client_instance=TavilyClient(api_key=api_key)

    return _client_instance

def tavily_search(query:str,max_results:int=5,search_depth: Literal["basic", "advanced", "fast", "ultra-fast"] = "basic",**kwargs)->list[dict]: #通过

        """Execute a single Tavily search query. Thread-safe.

    Acquires the Semaphore to limit concurrency (max 3 simultaneous
    Tavily API calls across all threads). Returns [] on error instead
    of raising.

    Args:
        query: Search query string.
        max_results: Maximum number of results to return.
        **kwargs: Additional params passed to client.search().

    Returns:
        List of result dicts, each with 'title', 'url', 'content' keys.
        Returns empty list on API errors.
    """
        client=get_tavily_client()
        with _semaphore:
            try:
                response=client.search(query=query,
                                       max_results=max_results,
                                       search_depth=search_depth,**kwargs,)
                return response.get("results",[])
            except Exception as e:
                print(f"Tavily search failed for '{query}': {e}", flush=True)
                return []