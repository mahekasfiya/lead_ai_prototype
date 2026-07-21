from serpapi import GoogleSearch
from config import SERPAPI_KEY
from .models import SearchResult

def search(query, num_results=5):

    params={
        "engine":"google",
        "q": query,
        "api_key": SERPAPI_KEY,
        "num": num_results
    }

    search=GoogleSearch(params)

    data=search.get_dict()

    if "organic_results" not in data:
        print ("No organic search results found")
        return []

    results=[]

    for item in data.get("organic_results",[]):

        result=SearchResult(
            title=item.get("title", ""),
            url=item.get("link",""),
            snippet=item.get("snippet","")
        )

        results.append(result)

    return results