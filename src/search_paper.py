import requests
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("CORE_TOKEN")


RETRYABLE_STATUS = {429, 500, 502, 503, 504}

def core_post_with_retry(url: str, payload: dict, headers: dict, timeout: int = 60,
                         max_retries: int = 8, base_delay: float = 0.8):
    """
    Retries retryable HTTP statuses with exponential backoff + jitter.
    Specifically handles CORE ES overload failures that surface as 500.
    """
    last_err = None

    for attempt in range(max_retries + 1):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=timeout)

            if r.status_code < 400:
                return r

            # Retryable?
            if r.status_code in RETRYABLE_STATUS:
                # CORE sometimes returns useful headers; try to respect them
                retry_after = r.headers.get("X-RateLimit-Retry-After") or r.headers.get("Retry-After")
                if retry_after:
                    delay = float(retry_after)
                else:
                    delay = base_delay * (2 ** attempt)

                # Add jitter so you don't synchronize with others
                delay = delay * (0.7 + 0.6 * random.random())
                time.sleep(delay)
                last_err = RuntimeError(f"{r.status_code} {r.text}")
                continue

            # Non-retryable error: fail fast
            raise RuntimeError(f"{r.status_code} {r.text}")

        except (requests.Timeout, requests.ConnectionError) as e:
            # Treat transient network errors as retryable
            delay = base_delay * (2 ** attempt) * (0.7 + 0.6 * random.random())
            time.sleep(delay)
            last_err = e
            continue

    raise RuntimeError(f"Failed after retries. Last error: {last_err}")


def core_search_works(query: str, limit: int = 10, offset: int = 0):
    url = "https://api.core.ac.uk/v3/search/works"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    payload = {
        "q": query,
        "limit": limit,
        "offset": offset,
        "scroll": False,
        "stats": False,
    }

    r = core_post_with_retry(url, payload, headers=headers, timeout=60)
    return r.json()


if __name__ == "__main__":
    q = '(title:"tectonic rifting" OR abstract:"tectonic rifting" OR title:(geodynamics AND rifting) OR abstract:(geodynamics AND rifting)) AND yearPublished>=2015'
    data = core_search_works(q, limit=10, offset=0)

    # CORE sometimes uses totalHits vs total_hits depending on endpoint version
    print("hits:", data.get("totalHits", data.get("total_hits")))
    for w in data.get("results", []):
        print(w.get("id"), w.get("yearPublished"), w.get("title"))