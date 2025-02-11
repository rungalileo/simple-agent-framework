import aiohttp
import os
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dotenv import load_dotenv
from agent_framework.tools.base import BaseTool
from agent_framework.models import ToolMetadata

class EventFinderTool(BaseTool):
    """Tool for finding local events using Ticketmaster API"""

    @classmethod
    def get_metadata(cls) -> ToolMetadata:
        """Get tool metadata"""
        return ToolMetadata(
            name="event_finder",
            description="Searches for local events, concerts, sports, and more using Ticketmaster API",
            tags=["events", "entertainment", "local", "travel", "activities"],
            input_schema={
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "Location to search for events (city name or latitude,longitude)"
                    },
                    "start_date": {
                        "type": "string",
                        "format": "date",
                        "description": "Start date for event search (YYYY-MM-DD), defaults to today"
                    },
                    "end_date": {
                        "type": "string",
                        "format": "date",
                        "description": "End date for event search (YYYY-MM-DD), defaults to 7 days from start"
                    },
                    "category": {
                        "type": "string",
                        "enum": ["music", "sports", "arts", "family", "film", "misc"],
                        "description": "Type of events to search for"
                    },
                    "radius": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 100,
                        "default": 25,
                        "description": "Search radius in miles"
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 20,
                        "default": 10,
                        "description": "Maximum number of events to return"
                    }
                },
                "required": ["location"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "events": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "type": {"type": "string"},
                                "date": {"type": "string"},
                                "time": {"type": "string"},
                                "venue": {
                                    "type": "object",
                                    "properties": {
                                        "name": {"type": "string"},
                                        "address": {"type": "string"},
                                        "city": {"type": "string"},
                                        "state": {"type": "string"},
                                        "country": {"type": "string"}
                                    }
                                },
                                "price_range": {
                                    "type": "object",
                                    "properties": {
                                        "min": {"type": "number"},
                                        "max": {"type": "number"},
                                        "currency": {"type": "string"}
                                    }
                                },
                                "url": {"type": "string"},
                                "image_url": {"type": "string"}
                            }
                        }
                    },
                    "total_events": {"type": "integer"},
                    "search_location": {"type": "string"}
                }
            }
        )

    # Category mapping for Ticketmaster API
    CATEGORY_MAPPING = {
        "music": "KZFzniwnSyZfZ7v7nJ",
        "sports": "KZFzniwnSyZfZ7v7nE",
        "arts": "KZFzniwnSyZfZ7v7na",
        "family": "KZFzniwnSyZfZ7v7n1",
        "film": "KZFzniwnSyZfZ7v7nn",
        "misc": "KZFzniwnSyZfZ7v7n7"
    }

    async def execute(
        self,
        location: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        category: Optional[str] = None,
        radius: int = 25,
        limit: int = 10
    ) -> Dict[str, Any]:
        """Find events based on location and criteria"""
        # Load environment variables
        load_dotenv()
        api_key = os.getenv("TICKETMASTER_API_KEY")
        if not api_key:
            raise ValueError("TICKETMASTER_API_KEY environment variable is required")

        # Validate and process dates
        today = datetime.now().date()
        try:
            start = datetime.strptime(start_date, "%Y-%m-%d").date() if start_date else today
            end = datetime.strptime(end_date, "%Y-%m-%d").date() if end_date else start + timedelta(days=7)
        except ValueError as e:
            raise ValueError(f"Invalid date format. Use YYYY-MM-DD: {str(e)}")

        if start < today:
            start = today
        if end < start:
            end = start + timedelta(days=7)

        # Build API request
        base_url = "https://app.ticketmaster.com/discovery/v2/events"
        params = {
            "apikey": api_key,
            "locale": "en-US",
            "size": limit,
            "radius": radius,
            "unit": "miles",
            "startDateTime": f"{start}T00:00:00Z",
            "endDateTime": f"{end}T23:59:59Z"
        }

        # Add location parameter
        if "," in location:
            parts = [p.strip() for p in location.split(",")]
            # Try to parse as coordinates only if we have exactly 2 numeric parts
            if len(parts) == 2 and all(p.replace(".", "").replace("-", "").isdigit() for p in parts):
                try:
                    lat, lon = map(float, parts)
                    params["latlong"] = f"{lat},{lon}"
                except ValueError:
                    params["city"] = location
            else:
                # If parts don't look like coordinates, treat as city name
                params["city"] = location
        else:
            # No comma, treat as city name
            params["city"] = location

        # Add category if specified
        if category:
            if category not in self.CATEGORY_MAPPING:
                raise ValueError(f"Invalid category. Must be one of: {', '.join(self.CATEGORY_MAPPING.keys())}")
            params["segmentId"] = self.CATEGORY_MAPPING[category]

        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(base_url, params=params) as response:
                    if response.status == 401:
                        raise ValueError("Invalid API key")
                    elif response.status == 429:
                        raise Exception("Rate limit exceeded. Please try again later.")
                    elif response.status != 200:
                        raise Exception(f"Ticketmaster API error: {await response.text()}")
                    
                    data = await response.json()

            except aiohttp.ClientError as e:
                raise Exception(f"Network error while fetching events: {str(e)}")

        # Process results
        events = []
        if data.get("_embedded", {}).get("events"):
            for event in data["_embedded"]["events"]:
                # Extract venue information
                venue = event.get("_embedded", {}).get("venues", [{}])[0]
                
                # Extract price range
                price_range = None
                if "priceRanges" in event:
                    price = event["priceRanges"][0]
                    price_range = {
                        "min": price.get("min"),
                        "max": price.get("max"),
                        "currency": price.get("currency")
                    }

                # Format event data
                events.append({
                    "name": event.get("name"),
                    "type": event.get("type"),
                    "date": event.get("dates", {}).get("start", {}).get("localDate"),
                    "time": event.get("dates", {}).get("start", {}).get("localTime"),
                    "venue": {
                        "name": venue.get("name"),
                        "address": venue.get("address", {}).get("line1"),
                        "city": venue.get("city", {}).get("name"),
                        "state": venue.get("state", {}).get("stateCode"),
                        "country": venue.get("country", {}).get("countryCode")
                    },
                    "price_range": price_range,
                    "url": event.get("url"),
                    "image_url": next(
                        (img["url"] for img in event.get("images", [])
                         if img.get("ratio") == "16_9" and img.get("width") > 500),
                        next((img["url"] for img in event.get("images", [])), None)
                    )
                })

        return {
            "events": events,
            "total_events": data.get("page", {}).get("totalElements", 0),
            "search_location": location
        } 