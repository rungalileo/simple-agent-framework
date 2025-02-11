import aiohttp
import os
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
from agent_framework.tools.base import BaseTool
from agent_framework.models import ToolMetadata

class RestaurantRecommenderTool(BaseTool):
    """Tool for finding and recommending restaurants using Yelp Fusion API"""

    # Price level mapping for Yelp API
    PRICE_LEVELS = {
        "budget": "1,2",      # $ and $$
        "moderate": "2,3",    # $$ and $$$
        "expensive": "3,4",   # $$$ and $$$$
        "all": "1,2,3,4"     # All price ranges
    }

    @classmethod
    def get_metadata(cls) -> ToolMetadata:
        """Get tool metadata"""
        return ToolMetadata(
            name="restaurant_recommender",
            description="Finds and recommends restaurants based on location, cuisine, price range, and other criteria",
            tags=["restaurants", "food", "dining", "recommendations", "local"],
            input_schema={
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "Location to search for restaurants (city name or latitude,longitude)"
                    },
                    "cuisine": {
                        "type": "string",
                        "description": "Type of cuisine (e.g., 'italian', 'japanese', 'vegetarian')",
                        "default": None
                    },
                    "price_level": {
                        "type": "string",
                        "enum": ["budget", "moderate", "expensive", "all"],
                        "description": "Price range for restaurants",
                        "default": "all"
                    },
                    "min_rating": {
                        "type": "number",
                        "minimum": 1,
                        "maximum": 5,
                        "default": 3.5,
                        "description": "Minimum rating threshold (1-5)"
                    },
                    "open_now": {
                        "type": "boolean",
                        "default": True,
                        "description": "Only show currently open restaurants"
                    },
                    "radius": {
                        "type": "integer",
                        "minimum": 100,
                        "maximum": 40000,
                        "default": 5000,
                        "description": "Search radius in meters (max 40000)"
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 50,
                        "default": 10,
                        "description": "Number of results to return"
                    }
                },
                "required": ["location"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "restaurants": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "rating": {"type": "number"},
                                "review_count": {"type": "integer"},
                                "price_level": {"type": "string"},
                                "cuisine_types": {
                                    "type": "array",
                                    "items": {"type": "string"}
                                },
                                "location": {
                                    "type": "object",
                                    "properties": {
                                        "address": {"type": "string"},
                                        "city": {"type": "string"},
                                        "state": {"type": "string"},
                                        "zip_code": {"type": "string"},
                                        "country": {"type": "string"},
                                        "coordinates": {
                                            "type": "object",
                                            "properties": {
                                                "latitude": {"type": "number"},
                                                "longitude": {"type": "number"}
                                            }
                                        }
                                    }
                                },
                                "hours": {
                                    "type": "object",
                                    "properties": {
                                        "is_open_now": {"type": "boolean"},
                                        "hours_display": {"type": "string"}
                                    }
                                },
                                "contact": {
                                    "type": "object",
                                    "properties": {
                                        "phone": {"type": "string"},
                                        "website": {"type": "string"}
                                    }
                                },
                                "photos": {
                                    "type": "array",
                                    "items": {"type": "string"}
                                },
                                "url": {"type": "string"}
                            }
                        }
                    },
                    "total_found": {"type": "integer"},
                    "search_location": {"type": "string"},
                    "search_criteria": {
                        "type": "object",
                        "properties": {
                            "cuisine": {"type": "string"},
                            "price_level": {"type": "string"},
                            "min_rating": {"type": "number"},
                            "radius_meters": {"type": "integer"}
                        }
                    }
                }
            }
        )

    async def execute(
        self,
        location: str,
        cuisine: Optional[str] = None,
        price_level: str = "all",
        min_rating: float = 3.5,
        open_now: bool = True,
        radius: int = 5000,
        limit: int = 10
    ) -> Dict[str, Any]:
        """Find and recommend restaurants based on criteria"""
        # Load environment variables
        load_dotenv()
        api_key = os.getenv("YELP_API_KEY")
        if not api_key:
            raise ValueError("YELP_API_KEY environment variable is required")

        # Validate inputs
        if not location:
            raise ValueError("Location cannot be empty")
        if price_level not in self.PRICE_LEVELS:
            raise ValueError(f"Invalid price level. Must be one of: {', '.join(self.PRICE_LEVELS.keys())}")
        if not 1 <= min_rating <= 5:
            raise ValueError("Minimum rating must be between 1 and 5")
        if not 100 <= radius <= 40000:
            raise ValueError("Radius must be between 100 and 40000 meters")

        # Build API request
        base_url = "https://api.yelp.com/v3/businesses/search"
        headers = {
            "Authorization": f"Bearer {api_key}"
        }
        
        params = {
            "term": "restaurants",
            "limit": limit,
            "radius": radius,
            "open_now": str(open_now).lower(),
            "sort_by": "rating",
            "price": self.PRICE_LEVELS[price_level]
        }

        # Add location parameter
        if "," in location:
            parts = [p.strip() for p in location.split(",")]
            # Try to parse as coordinates only if we have exactly 2 numeric parts
            if len(parts) == 2 and all(p.replace(".", "").replace("-", "").isdigit() for p in parts):
                try:
                    lat, lon = map(float, parts)
                    params["latitude"] = lat
                    params["longitude"] = lon
                except ValueError:
                    params["location"] = location
            else:
                # If parts don't look like coordinates, treat as city name
                params["location"] = location
        else:
            # No comma, treat as city name
            params["location"] = location

        # Add cuisine type if specified
        if cuisine:
            params["categories"] = cuisine

        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(base_url, headers=headers, params=params) as response:
                    if response.status == 401:
                        raise ValueError("Invalid API key")
                    elif response.status == 429:
                        raise Exception("Rate limit exceeded. Please try again later.")
                    elif response.status != 200:
                        raise Exception(f"Yelp API error: {await response.text()}")
                    
                    data = await response.json()

            except aiohttp.ClientError as e:
                raise Exception(f"Network error while fetching restaurants: {str(e)}")

        # Filter results by minimum rating
        restaurants = []
        for business in data.get("businesses", []):
            if business.get("rating", 0) >= min_rating:
                # Format restaurant data
                restaurant = {
                    "name": business.get("name"),
                    "rating": business.get("rating"),
                    "review_count": business.get("review_count"),
                    "price_level": business.get("price", "N/A"),
                    "cuisine_types": [
                        category["title"] 
                        for category in business.get("categories", [])
                    ],
                    "location": {
                        "address": business.get("location", {}).get("address1"),
                        "city": business.get("location", {}).get("city"),
                        "state": business.get("location", {}).get("state"),
                        "zip_code": business.get("location", {}).get("zip_code"),
                        "country": business.get("location", {}).get("country"),
                        "coordinates": business.get("coordinates", {})
                    },
                    "hours": {
                        "is_open_now": business.get("is_closed", True) == False,
                        "hours_display": "Hours available on Yelp"  # Full hours require additional API call
                    },
                    "contact": {
                        "phone": business.get("phone"),
                        "website": business.get("url")  # Using Yelp URL as website
                    },
                    "photos": [business.get("image_url")] if business.get("image_url") else [],
                    "url": business.get("url")
                }
                restaurants.append(restaurant)

        return {
            "restaurants": restaurants,
            "total_found": data.get("total", 0),
            "search_location": location,
            "search_criteria": {
                "cuisine": cuisine,
                "price_level": price_level,
                "min_rating": min_rating,
                "radius_meters": radius
            }
        } 