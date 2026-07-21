import requests

def fetch_page(url):

    headers={
        "User-Agent":
        "Mozilla/5.0"
    }

    response=requests.get(
        url,
        headers=headers,
        timeout=10
    )

    response.raise_for_status()

    return response.text