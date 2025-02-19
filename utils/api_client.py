import requests
import logging
from typing import Optional, Dict, Any
import asyncio

logger = logging.getLogger(__name__)


class ApiClient:
    def __init__(self, api_key: Optional[str] = None, host: Optional[str] = None):
        self.api_key = api_key
        self.host = host
        self.session = self._create_session()

    def _create_session(self) -> requests.Session:
        """Create and configure a session for making requests."""
        session = requests.Session()
        session.headers.update({"Content-Type": "application/json"})
        if self.api_key:
            session.headers.update({"Authorization": f"Bearer {self.api_key}"})
        return session

    async def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Make an API request with retry logic."""
        url = f"{self.host}{endpoint}"
        try:
            # Use asyncio.to_thread to make the blocking request non-blocking
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, lambda: self.session.request(method, url, json=data)
            )
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            raise

    async def get(self, endpoint: str) -> Dict[str, Any]:
        """Send a GET request to the specified endpoint."""
        return await self._make_request("GET", endpoint)

    async def post(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Send a POST request to the specified endpoint."""
        return await self._make_request("POST", endpoint, data)