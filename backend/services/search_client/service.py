"""Search flow: cache, then limiter, then upstream (MSP-21; FR-005, FR-006, FR-008, FR-055).

Only a successful, non-empty result is cached. Timeouts, HTTP errors and empty
responses raise an UpstreamError naming the service and are never cached.
Nothing here is called by the SMILES submit path, so entering SMILES directly
keeps working when PubChem or ChEMBL are down (FR-008).
"""
from backend.services.search_client.cache import SearchCache, get_redis
from backend.services.search_client.chembl import SERVICE as CHEMBL, search_chembl
from backend.services.search_client.errors import UpstreamEmpty
from backend.services.search_client.limiter import SharedLimiter
from backend.services.search_client.pubchem import SERVICE as PUBCHEM, search_pubchem

CLIENTS = {"pubchem": search_pubchem, "chembl": search_chembl}
SERVICE_NAMES = {"pubchem": PUBCHEM, "chembl": CHEMBL}
MAX_QUERY_LEN = 200


class SearchService:
    def __init__(self, cache, limiter, clients=None):
        self.cache, self.limiter = cache, limiter
        self.clients = clients or CLIENTS

    def search(self, query, source):
        cached = self.cache.get(source, query)
        if cached is not None:
            return {**cached, "cached": True}

        name = SERVICE_NAMES[source]
        if source == "pubchem":
            self.limiter.acquire(name)

        results = self.clients[source](query)  # may raise an UpstreamError
        if not results:
            raise UpstreamEmpty(name, f"{name} returned no results for that query.")

        payload = {"source": source, "query": query, "results": results}
        self.cache.set(source, query, payload)  # reached only on success
        return {**payload, "cached": False}


_default = None


def get_default_service():
    global _default  # pylint: disable=global-statement
    if _default is None:
        _default = SearchService(SearchCache(), SharedLimiter(get_redis))
    return _default
