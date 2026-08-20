"""
Immich API client for interacting with the Immich instance.
"""

import time
from typing import Any, Dict, Iterator, List, Optional
import httpx
from .models import Asset, Tag, BulkTagRequest, CreateTagRequest
from .config import settings
from .logging import get_logger
from .performance_monitor import performance_monitor


class ImmichAPIError(Exception):
    """Custom exception for Immich API errors."""
    pass


class ImmichClient:
    """Client for interacting with the Immich API with multi-library support."""
    
    def __init__(self):
        self.base_url = settings.immich_base_url
        self.logger = get_logger("immich_client")
        self.timeout = settings.request_timeout
        self.max_retries = settings.max_retries
        self.retry_delay = settings.retry_delay
        
        # Multi-library support
        self.library_configs = settings.get_library_config()
        self.current_library_index = 0
        self.current_library = self.library_configs[0] if self.library_configs else {"name": "Unknown", "api_key": ""}
        
        # Per-library tag caching
        self._tag_caches: Dict[str, Dict[str, Tag]] = {
            lib['api_key']: {} for lib in self.library_configs
        }
        self._tag_cache_valid: Dict[str, bool] = {
            lib['api_key']: False for lib in self.library_configs
        }
        self._tag_cache_timestamp: Dict[str, float] = {
            lib['api_key']: 0 for lib in self.library_configs
        }
        
        self.logger.info(f"🏛️ Initialized with {len(self.library_configs)} libraries: {[lib['name'] for lib in self.library_configs]}")
        
        # Initialize HTTP client
        self._setup_http_client()
    
    @property
    def api_key(self):
        """Get current API key for backward compatibility."""
        return self.current_library["api_key"]
    
    @property
    def current_library_name(self):
        """Get current library name."""
        return self.current_library["name"]
    
    @property
    def _tag_cache(self):
        """Get tag cache for current library."""
        return self._tag_caches.get(self.api_key, {})
    
    @_tag_cache.setter
    def _tag_cache(self, value):
        """Set tag cache for current library."""
        self._tag_caches[self.api_key] = value
    
    def switch_to_library(self, library_index: int):
        """Switch to a specific library by index."""
        if 0 <= library_index < len(self.library_configs):
            old_name = self.current_library_name
            self.current_library_index = library_index
            self.current_library = self.library_configs[library_index]
            
            # Update HTTP client headers with new API key
            if hasattr(self, 'client'):
                self.client.headers["X-API-Key"] = self.api_key
            
            # Only log if actually switching to a different library
            if old_name != self.current_library_name:
                self.logger.info(f"🔄 Switched from '{old_name}' to '{self.current_library_name}' ({library_index + 1}/{len(self.library_configs)})")
        else:
            raise ValueError(f"Invalid library index: {library_index}")
    
    def _switch_to_library_silent(self, library_index: int):
        """Switch to a specific library by index without logging."""
        if 0 <= library_index < len(self.library_configs):
            self.current_library_index = library_index
            self.current_library = self.library_configs[library_index]
            
            # Update HTTP client headers with new API key
            if hasattr(self, 'client'):
                self.client.headers["X-API-Key"] = self.api_key
        else:
            raise ValueError(f"Invalid library index: {library_index}")
    
    def switch_to_next_library(self):
        """Switch to the next library in rotation."""
        next_index = (self.current_library_index + 1) % len(self.library_configs)
        self.switch_to_library(next_index)
    
    @staticmethod
    def _parse_user_info(user_data: Dict) -> Dict:
        """Return the stable subset of an Immich user response."""
        return {
            "id": user_data.get("id", "unknown"),
            "name": user_data.get("name", "Unknown User"),
            "email": user_data.get("email", "unknown@example.com"),
        }

    def get_current_user_info(self) -> Dict:
        """Get information about the current processing user."""
        try:
            response = self._make_request(
                method="GET",
                endpoint="/api/users/me"
            )
            
            if response.status_code == 200:
                return self._parse_user_info(response.json())
        except Exception as e:
            self.logger.debug(f"Failed to get user info: {e}")
        
        return {"id": "unknown", "name": "Unknown User", "email": "unknown@example.com"}

    def get_user_info_for_api_key(self, api_key: str) -> Dict:
        """Get user information without changing the active library."""
        response = self._make_request_silent(
            method="GET",
            endpoint="/api/users/me",
            api_key=api_key,
        )
        return self._parse_user_info(response.json())
    
    def _get_cache_properties(self):
        """Get cache properties for current library."""
        api_key = self.api_key
        return {
            'valid': self._tag_cache_valid.get(api_key, False),
            'timestamp': self._tag_cache_timestamp.get(api_key, 0),
            'ttl': settings.tag_cache_ttl
        }
    
    def _set_cache_properties(self, valid: bool, timestamp: float = None):
        """Set cache properties for current library."""
        api_key = self.api_key
        self._tag_cache_valid[api_key] = valid
        if timestamp is not None:
            self._tag_cache_timestamp[api_key] = timestamp
    
    def _setup_http_client(self):
        """Setup HTTP client with current library's API key."""
        # HTTP client with retry logic
        self.client = httpx.Client(
            timeout=self.timeout,
            headers={
                "X-API-Key": self.api_key,
                "Content-Type": "application/json",
            }
        )
        self._cache_ttl = settings.tag_cache_ttl
        
        # Silence httpx request logging
        import logging
        logging.getLogger("httpx").setLevel(logging.WARNING)

    @staticmethod
    def _tag_cache_key(tag: Tag) -> str:
        """Build a case-insensitive cache key from a tag's complete path."""
        return tag.path.casefold()
    
    def _make_request(
        self, 
        method: str, 
        endpoint: str, 
        params: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Make an HTTP request with retry logic."""
        url = f"{self.base_url}{endpoint}"
        request_start = time.time()
        
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.request(
                    method=method,
                    url=url,
                    params=params,
                    json=json_data
                )
                response.raise_for_status()
                
                # Record successful API call
                response_time = time.time() - request_start
                performance_monitor.record_api_call(response_time)
                
                return response
                
            except httpx.HTTPStatusError as e:
                if e.response.status_code >= 500 and attempt < self.max_retries:
                    self.logger.warning(
                        f"⚠️  Server error {e.response.status_code}, retrying "
                        f"(attempt {attempt + 1}/{self.max_retries})"
                    )
                    time.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
                    continue
                else:
                    self.logger.error(
                        f"❌ HTTP {method} {url} failed: {e.response.status_code} - {e.response.text}"
                    )
                    raise ImmichAPIError(f"HTTP {e.response.status_code}: {e.response.text}")
                    
            except httpx.RequestError as e:
                if attempt < self.max_retries:
                    self.logger.warning(f"Request error, retrying (attempt {attempt + 1}/{self.max_retries}): {e}")
                    time.sleep(self.retry_delay * (2 ** attempt))
                    continue
                else:
                    self.logger.error(f"❌ Request failed: {str(e)}")
                    raise ImmichAPIError(f"Request failed: {e}")
    
    def _make_request_silent(
        self, 
        method: str, 
        endpoint: str, 
        params: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        api_key: Optional[str] = None,
    ) -> Any:
        """Make an HTTP request without logging (for health checks)."""
        url = f"{self.base_url}{endpoint}"
        headers = {"X-API-Key": api_key} if api_key else None
        
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.request(
                    method=method,
                    url=url,
                    params=params,
                    json=json_data,
                    headers=headers,
                )
                response.raise_for_status()
                return response
                
            except httpx.HTTPStatusError as e:
                if e.response.status_code >= 500 and attempt < self.max_retries:
                    time.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
                    continue
                else:
                    raise ImmichAPIError(f"HTTP {e.response.status_code}: {e.response.text}")
                    
            except httpx.RequestError as e:
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay * (2 ** attempt))
                    continue
                else:
                    raise ImmichAPIError(f"Request failed: {e}")
    
    def _iter_metadata_asset_items(
        self,
        filters: Dict[str, Any],
        page_size: int = 250
    ) -> Iterator[Dict[str, Any]]:
        """Yield every asset item from a paginated metadata search."""
        if page_size < 1:
            raise ImmichAPIError("Metadata search page size must be positive")

        page = 1
        visited_pages = set()

        while True:
            if page in visited_pages:
                raise ImmichAPIError(
                    f"Metadata search returned a repeated page: {page}"
                )
            visited_pages.add(page)

            request_data = dict(filters)
            request_data.update({"page": page, "size": page_size})

            response = self._make_request(
                method="POST",
                endpoint="/api/search/metadata",
                json_data=request_data
            )
            response_data = response.json()

            if not isinstance(response_data, dict):
                raise ImmichAPIError(
                    "Metadata search returned a non-object response"
                )

            assets_section = response_data.get("assets")
            if not isinstance(assets_section, dict):
                raise ImmichAPIError(
                    "Metadata search response is missing its assets section"
                )

            if "items" not in assets_section:
                raise ImmichAPIError(
                    "Metadata search assets section is missing its items"
                )

            assets_list = assets_section["items"]
            if not isinstance(assets_list, list):
                raise ImmichAPIError(
                    "Metadata search assets section contains invalid items"
                )

            if not assets_list:
                return

            for asset_data in assets_list:
                if not isinstance(asset_data, dict):
                    raise ImmichAPIError(
                        "Metadata search returned a non-object asset"
                    )
                yield asset_data

            next_page = assets_section.get("nextPage")
            if next_page in (None, ""):
                return

            try:
                page = int(next_page)
            except (TypeError, ValueError) as e:
                raise ImmichAPIError(
                    f"Metadata search returned an invalid next page: {next_page!r}"
                ) from e

            if page < 1:
                raise ImmichAPIError(
                    f"Metadata search returned an invalid next page: {page}"
                )

    def get_untagged_assets(self, limit: int = 250) -> List[Asset]:
        """Get image assets that have no tags using the metadata search endpoint.
        
        Metadata search pages are followed until the requested limit is reached.
        As we tag assets with 'auto:processed', they disappear from this search automatically.
        Only searches for IMAGE assets since WD14 cannot process videos.
        
        Returns:
            List of untagged image assets, capped by limit.
        """
        if limit < 1:
            raise ImmichAPIError("Asset search limit must be positive")

        library_name = self.current_library_name
        self.logger.debug(f"🔍 Library '{library_name}': Getting untagged image assets via metadata search")
        
        assets = []
        page_size = min(250, limit)

        for asset_data in self._iter_metadata_asset_items(
            {
                "tagIds": None,
                "type": "IMAGE"
            },
            page_size=page_size
        ):
            try:
                assets.append(Asset(**asset_data))
            except Exception as e:
                asset_id = asset_data.get("id", "unknown")
                raise ImmichAPIError(
                    f"Failed to parse asset {asset_id}: {e}"
                ) from e

            if len(assets) >= limit:
                break

        self.logger.info(
            f"✅ Library '{library_name}': "
            f"Found {len(assets)} untagged image assets"
        )
        return assets
    
    def get_unprocessed_assets(
        self,
        processed_tag_id: Optional[str] = None,
        limit: int = 250
    ) -> List[Asset]:
        """
        TARGET_ALBUMS behavior:

          empty
            -> preserve original behavior:
               return assets with no tags at all

          Anime
            -> return untreated assets in Anime

          Anime,Hentai
            -> return untreated assets in Anime OR Hentai

        When album filtering is enabled, an asset is considered treated
        when Immich's metadata search matches it against processed_tag_id.
        Search results do not reliably embed their tags, so processed IDs are
        fetched separately and subtracted from the album assets.
        """

        target_album_names = [
            name.strip()
            for name in settings.target_albums.split(",")
            if name.strip()
        ]

        if limit < 1:
            raise ImmichAPIError("Asset search limit must be positive")

        # Preserve the original tagger behavior when no albums are targeted.
        if not target_album_names:
            return self.get_untagged_assets(limit=limit)

        if not processed_tag_id:
            raise ImmichAPIError("processed_tag_id is required")

        library_name = self.current_library_name

        # Resolve configured album names.
        response = self._make_request(
            method="GET",
            endpoint="/api/albums"
        )

        albums = response.json()
        if not isinstance(albums, list):
            raise ImmichAPIError("Album listing returned a non-list response")

        albums_by_name = {}

        for album in albums:
            if not isinstance(album, dict):
                raise ImmichAPIError("Album listing returned a non-object album")

            name = album.get("albumName", "")
            album_id = album.get("id")

            if name and album_id:
                albums_by_name.setdefault(
                    name.casefold(),
                    []
                ).append(album_id)

        album_ids = []
        missing_albums = []
        resolved_album_names = []

        for requested_name in target_album_names:
            matches = albums_by_name.get(
                requested_name.casefold(),
                []
            )

            try:
                if not matches:
                    raise ImmichAPIError(
                        f"Target album was not found: {requested_name}"
                    )
            except ImmichAPIError as e:
                missing_albums.append(requested_name)
                self.logger.error(
                    f"❌ Library '{library_name}': {e}"
                )
                continue

            album_ids.extend(matches)
            resolved_album_names.append(requested_name)

        if not album_ids:
            raise ImmichAPIError(
                "None of the configured target albums were found: "
                + ", ".join(missing_albums)
            )

        if missing_albums:
            self.logger.warning(
                f"⚠️ Library '{library_name}': Continuing with valid target "
                f"albums; missing: {', '.join(missing_albums)}"
            )

        album_ids = list(dict.fromkeys(album_ids))

        self.logger.info(
            f"🎯 Library '{library_name}': "
            f"Filtering to albums: "
            f"{', '.join(resolved_album_names)}"
        )

        search_filters = {
            "albumIds": album_ids,
            "type": "IMAGE"
        }

        processed_asset_ids = set()
        for asset_data in self._iter_metadata_asset_items(
            {**search_filters, "tagIds": [processed_tag_id]}
        ):
            asset_id = asset_data.get("id")
            if not asset_id:
                raise ImmichAPIError(
                    "Processed asset search returned an asset without an ID"
                )
            processed_asset_ids.add(asset_id)

        assets = []
        page_size = min(250, limit)

        for asset_data in self._iter_metadata_asset_items(
            search_filters,
            page_size=page_size
        ):
            asset_id = asset_data.get("id")
            if not asset_id:
                raise ImmichAPIError(
                    "Album asset search returned an asset without an ID"
                )

            if asset_id in processed_asset_ids:
                continue

            try:
                assets.append(Asset(**asset_data))
            except Exception as e:
                raise ImmichAPIError(
                    f"Failed to parse asset {asset_id}: {e}"
                ) from e

            if len(assets) >= limit:
                break

        self.logger.info(
            f"🎯 Library '{library_name}': "
            f"Found {len(assets)} untreated images "
            f"in albums: {', '.join(resolved_album_names)}"
        )

        return assets
    
    def download_asset(self, asset_id: str, use_thumbnail: bool = True) -> bytes:
        """Download an asset (thumbnail or original)."""
        endpoint = f"/api/assets/{asset_id}/{'thumbnail' if use_thumbnail else 'download'}"
        
        self.logger.debug("Downloading asset", asset_id=asset_id, use_thumbnail=use_thumbnail)
        
        url = f"{self.base_url}{endpoint}"
        
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.get(url)
                response.raise_for_status()
                content = response.content
                self.logger.debug("Asset downloaded", asset_id=asset_id, size=len(content))
                return content
                    
            except httpx.HTTPStatusError as e:
                if e.response.status_code >= 500 and attempt < self.max_retries:
                    self.logger.warning(
                        f"⚠️  Server error {e.response.status_code}, retrying "
                        f"(attempt {attempt + 1}/{self.max_retries})"
                    )
                    time.sleep(self.retry_delay * (2 ** attempt))
                    continue
                else:
                    self.logger.error(
                        "HTTP request failed",
                        method="GET",
                        url=url,
                        status_code=e.response.status_code,
                        response_text=e.response.text
                    )
                    raise ImmichAPIError(f"HTTP {e.response.status_code}: {e.response.text}")
                    
            except httpx.RequestError as e:
                if attempt < self.max_retries:
                    self.logger.warning(f"Request error, retrying (attempt {attempt + 1}/{self.max_retries}): {e}")
                    time.sleep(self.retry_delay * (2 ** attempt))
                    continue
                else:
                    self.logger.error(f"❌ Request failed: {str(e)}")
                    raise ImmichAPIError(f"Request failed: {e}")
    
    def get_all_tags(self, use_cache: bool = True) -> List[Tag]:
        """Get all tags from Immich with optional caching."""
        current_time = time.time()
        
        # Check if cache is valid
        cache_props = self._get_cache_properties()
        if (use_cache and cache_props['valid'] and 
            current_time - cache_props['timestamp'] < cache_props['ttl']):
            self.logger.debug("Using cached tags", count=len(self._tag_cache))
            return list(self._tag_cache.values())
        
        self.logger.debug("Fetching all tags from API")
        
        response = self._make_request(method="GET", endpoint="/api/tags")
        tags_data = response.json()
        tags = [Tag(**tag_data) for tag_data in tags_data]
        
        # Update cache
        if use_cache:
            self._tag_cache = {self._tag_cache_key(tag): tag for tag in tags}
            self._set_cache_properties(valid=True, timestamp=current_time)
            self.logger.debug("Updated tag cache", count=len(self._tag_cache))
        
        self.logger.debug("Fetched tags", count=len(tags))
        return tags
    
    def create_tag(self, tag_request: CreateTagRequest) -> Tag:
        """Create a new tag in Immich."""
        self.logger.debug("Creating tag", name=tag_request.name)
        
        response = self._make_request(
            method="POST",
            endpoint="/api/tags",
            json_data=tag_request.model_dump(exclude_none=True)
        )
        
        tag_data = response.json()
        tag = Tag(**tag_data)
        
        # Update cache immediately
        self._tag_cache[self._tag_cache_key(tag)] = tag
        
        self.logger.debug("Created tag", tag_id=tag.id, name=tag.path)
        return tag

    def get_tag(self, tag_name: str) -> Optional[Tag]:
        """Return an existing tag by its complete path, if present."""
        cache_props = self._get_cache_properties()
        if not cache_props['valid']:
            self.get_all_tags(use_cache=True)
        return self._tag_cache.get(tag_name.strip().casefold())
    
    def get_or_create_tag(self, tag_name: str) -> Tag:
        """Get an existing tag or create it if it doesn't exist."""
        # Validate tag name first
        if not self._is_valid_tag_name(tag_name):
            self.logger.debug(f"Skipping invalid tag name: '{tag_name}'")
            raise ValueError(f"Invalid tag name: '{tag_name}'")
        
        tag_name_clean = tag_name.strip()
        tag_name_lower = tag_name_clean.casefold()
        
        # Ensure cache is populated
        cache_props = self._get_cache_properties()
        if not cache_props['valid']:
            self.get_all_tags(use_cache=True)
        
        # Check cache first
        if tag_name_lower in self._tag_cache:
            performance_monitor.record_cache_hit()
            performance_monitor.record_tag_from_cache()
            return self._tag_cache[tag_name_lower]
        
        # Create new tag if not found
        performance_monitor.record_cache_miss()
        self.logger.debug("Creating new tag", name=tag_name_clean)
        try:
            tag_request = CreateTagRequest(name=tag_name_clean)
            new_tag = self.create_tag(tag_request)
            performance_monitor.record_tag_created()
            
            # Add to cache
            self._tag_cache[tag_name_lower] = new_tag
            return new_tag
            
        except Exception as e:
            # Handle "tag already exists" case
            if "already exists" in str(e).lower():
                # Refresh cache and try again
                self.invalidate_tag_cache()
                self.get_all_tags(use_cache=True)
                if tag_name_lower in self._tag_cache:
                    self.logger.debug(f"Found existing tag after cache refresh: {tag_name_clean}")
                    return self._tag_cache[tag_name_lower]
            
            # Re-raise the exception if we can't handle it
            raise

    def get_or_create_child_tag(self, parent: Tag, child_name: str) -> Tag:
        """Get or create a true Immich child tag beneath ``parent``."""
        if not self._is_valid_tag_name(child_name) or "/" in child_name:
            raise ValueError(f"Invalid child tag name: '{child_name}'")

        child_name_clean = child_name.strip()
        child_path = f"{parent.path}/{child_name_clean}"
        existing = self.get_tag(child_path)
        if existing:
            return existing

        try:
            return self.create_tag(
                CreateTagRequest(
                    name=child_name_clean,
                    parentId=parent.id,
                )
            )
        except Exception as e:
            if "already exists" not in str(e).lower():
                raise

            self.invalidate_tag_cache()
            existing = self.get_tag(child_path)
            if existing:
                return existing
            raise
    
    def _is_valid_tag_name(self, tag_name: str) -> bool:
        """Check if a tag name is valid for Immich."""
        if not tag_name or not tag_name.strip():
            return False
        
        # Only filter out characters that would actually break the API or filesystem
        # Be more permissive for anime tags which may have special characters
        invalid_chars = ['\n', '\r', '\t']  # Only control characters
        for char in invalid_chars:
            if char in tag_name:
                return False
        
        # Check length (reasonable limits)
        tag_cleaned = tag_name.strip()
        if len(tag_cleaned) < 1 or len(tag_cleaned) > 100:
            return False
            
        return True

    def get_or_create_tags_bulk(self, tag_names: List[str]) -> Dict[str, Tag]:
        """Get or create multiple tags efficiently. Returns a mapping of original tag names to Tag objects."""
        if not tag_names:
            return {}
        
        # Filter out invalid tag names
        valid_tag_names = [name for name in tag_names if self._is_valid_tag_name(name)]
        if len(valid_tag_names) < len(tag_names):
            invalid_tags = [name for name in tag_names if not self._is_valid_tag_name(name)]
            self.logger.debug(f"Filtered out {len(invalid_tags)} invalid tag names", invalid_tags=invalid_tags)
        
        if not valid_tag_names:
            return {}
        
        # Ensure cache is populated
        cache_props = self._get_cache_properties()
        if not cache_props['valid']:
            self.get_all_tags(use_cache=True)
        
        result = {}
        missing_tags = []
        
        # Check which tags exist in cache
        for tag_name in valid_tag_names:
            tag_name_lower = tag_name.casefold()
            if tag_name_lower in self._tag_cache:
                result[tag_name] = self._tag_cache[tag_name_lower]
                performance_monitor.record_cache_hit()
                performance_monitor.record_tag_from_cache()
            else:
                missing_tags.append(tag_name)
                performance_monitor.record_cache_miss()
        
        # Create missing tags
        if missing_tags:
            performance_monitor.record_bulk_operation()
            self.logger.debug("Creating missing tags", count=len(missing_tags))
            for tag_name in missing_tags:
                try:
                    new_tag = self.create_tag(CreateTagRequest(name=tag_name.strip()))
                    result[tag_name] = new_tag
                    performance_monitor.record_tag_created()
                except Exception as e:
                    # Check if tag already exists (common race condition)
                    if "already exists" in str(e).lower():
                        # Refresh cache and try to find the tag
                        self.invalidate_tag_cache()
                        self.get_all_tags(use_cache=True)
                        tag_name_lower = tag_name.casefold()
                        if tag_name_lower in self._tag_cache:
                            result[tag_name] = self._tag_cache[tag_name_lower]
                            self.logger.debug(f"Found existing tag after cache refresh: {tag_name}")
                        else:
                            self.logger.debug(f"Tag exists but not found in cache: {tag_name}")
                    else:
                        self.logger.debug(f"Failed to create tag '{tag_name}': {e}")
                    # Continue with other tags
                    continue
        
        self.logger.debug("Bulk tag lookup/creation completed", 
                         requested=len(tag_names), 
                         found=len(result))
        return result
    
    def bulk_tag_assets(self, asset_ids: List[str], tag_ids: List[str]) -> None:
        """Bulk tag assets with multiple tags."""
        if not asset_ids or not tag_ids:
            return
        
        self.logger.debug(
            "Bulk tagging assets",
            asset_count=len(asset_ids),
            tag_count=len(tag_ids)
        )
        
        # Use the correct bulk tagging endpoint with PUT method
        request_data = BulkTagRequest(assetIds=asset_ids, tagIds=tag_ids)
        
        self._make_request(
            method="PUT",
            endpoint="/api/tags/assets",
            json_data=request_data.dict()
        )
        
        self.logger.debug(
            "Bulk tagged assets",
            asset_count=len(asset_ids),
            tag_count=len(tag_ids)
        )

    def bulk_untag_assets(self, tag_id: str, asset_ids: List[str]) -> None:
        """Remove one tag from a batch of assets."""
        if not asset_ids:
            return

        self._make_request(
            method="DELETE",
            endpoint=f"/api/tags/{tag_id}/assets",
            json_data={"ids": asset_ids},
        )

        self.logger.debug(
            "Bulk untagged assets",
            tag_id=tag_id,
            asset_count=len(asset_ids),
        )
    
    def tag_single_asset(self, asset_id: str, tag_ids: List[str]) -> None:
        """Tag a single asset with multiple tags."""
        if not tag_ids:
            return
        
        self.logger.debug("Tagging single asset", asset_id=asset_id, tag_count=len(tag_ids))
        
        # Use the bulk endpoint for single asset tagging (simpler approach)
        request_data = BulkTagRequest(assetIds=[asset_id], tagIds=tag_ids)
        
        self._make_request(
            method="PUT",
            endpoint="/api/tags/assets",
            json_data=request_data.dict()
        )
        
        self.logger.debug("Tagged single asset", asset_id=asset_id, tag_count=len(tag_ids))
    
    def get_assets_with_tag(self, tag_id: str, limit: Optional[int] = None) -> List[Asset]:
        """Get all assets that have a specific tag.
        
        Args:
            tag_id: The tag ID to search for
            limit: Maximum number of assets to return (default: 1000)
        """
        if limit is None:
            limit = 1000  # Default reasonable limit for tagged asset queries
            
        self.logger.debug(f"📊 Getting assets with tag {tag_id}, limit={limit}")
        
        # Use metadata search with specific tag filter
        response = self._make_request(
            method="POST",
            endpoint="/api/search/metadata",
            json_data={"tagIds": [tag_id]}
        )
        
        response_data = response.json()
        assets_section = response_data.get("assets", {})
        assets_list = assets_section.get("items", [])
        
        # Parse assets
        assets = []
        for asset_data in assets_list[:limit]:  # Respect limit
            try:
                assets.append(Asset(**asset_data))
            except Exception as e:
                self.logger.warning(f"⚠️  Failed to parse tagged asset: {e}")
                continue
        
        if len(assets_list) >= limit:
            self.logger.warning(
                f"Retrieved {len(assets)} tagged assets (limit: {limit}). "
                "There may be more assets with this tag."
            )
        
        return assets

    def get_asset_ids_with_tag(self, tag_id: str) -> List[str]:
        """Return every asset ID associated with a tag."""
        asset_ids = []
        for asset_data in self._iter_metadata_asset_items(
            {"tagIds": [tag_id]},
        ):
            asset_id = asset_data.get("id")
            if not asset_id:
                raise ImmichAPIError(
                    "Tagged asset search returned an asset without an ID"
                )
            asset_ids.append(asset_id)
        return asset_ids

    def migrate_tag(self, source: Tag, destination: Tag, batch_size: int = 250) -> int:
        """Move all asset associations to another tag, then delete the source."""
        if batch_size < 1:
            raise ValueError("Tag migration batch size must be positive")
        if source.id == destination.id:
            return 0

        asset_ids = self.get_asset_ids_with_tag(source.id)
        for offset in range(0, len(asset_ids), batch_size):
            batch = asset_ids[offset:offset + batch_size]
            self.bulk_tag_assets(batch, [destination.id])
            self.bulk_untag_assets(source.id, batch)

        self.delete_tag(source.id)
        self.invalidate_tag_cache()
        return len(asset_ids)
    
    def get_asset(self, asset_id: str) -> Asset:
        """Get a specific asset by ID."""
        self.logger.debug("Getting asset", asset_id=asset_id)
        
        response = self._make_request(method="GET", endpoint=f"/api/assets/{asset_id}")
        asset_data = response.json()
        asset = Asset(**asset_data)
        
        self.logger.debug("Retrieved asset", asset_id=asset_id, name=asset.originalFileName)
        return asset
    
    def remove_tags_from_asset(self, asset_id: str, tag_ids: List[str]) -> None:
        """Remove specific tags from an asset."""
        if not tag_ids:
            return
        
        self.logger.debug("Removing tags from asset", asset_id=asset_id, tag_count=len(tag_ids))
        
        for tag_id in tag_ids:
            self.bulk_untag_assets(tag_id, [asset_id])
        
        self.logger.info("Removed tags from asset", asset_id=asset_id, tag_count=len(tag_ids))
    
    def delete_tag(self, tag_id: str) -> None:
        """Delete a tag from Immich."""
        self.logger.debug("Deleting tag", tag_id=tag_id)
        
        self._make_request(method="DELETE", endpoint=f"/api/tags/{tag_id}")
        
        self.logger.info("Deleted tag", tag_id=tag_id)
    
    def test_connection(self) -> bool:
        """Test the connection to Immich."""
        try:
            # Try to get tags as a simple test
            self.get_all_tags()
            self.logger.info("Connection test successful")
            return True
        except Exception as e:
            self.logger.error(f"Connection test failed: {e}")
            return False
    
    def invalidate_tag_cache(self):
        """Invalidate the tag cache to force refresh on next access."""
        self._set_cache_properties(valid=False)
        self._tag_cache = {}
        self.logger.debug("Tag cache invalidated")
    
    def close(self):
        """Close the HTTP client."""
        self.client.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
