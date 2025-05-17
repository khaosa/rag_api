from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import json
import google.generativeai as genai
from dotenv import load_dotenv
import os

app = FastAPI(
    title="Gemini Trip Planner API",
    description="API for generating travel itineraries using Google's Gemini AI",
    version="1.0.0"
)

class TripRequest(BaseModel):
    destination: str
    duration_days: int
    traveler_preferences: List[str]
    trip_style: str = "luxury"
    pace: str = "moderate"

class GeminiTripPlanner:
    def __init__(self, api_key: str):
        """Initialize the trip planner with API key."""
        genai.configure(api_key=api_key)
        self.api_key = api_key
        self.model = genai.GenerativeModel("gemini-2.0-flash-lite")  # Fixed model

    def generate_trip_plan(
        self,
        destination: str,
        duration_days: int,
        traveler_preferences: list,
        trip_style: str = "luxury",
        pace: str = "moderate"
    ) -> Dict[str, Any]:
        """
        Generate a structured trip plan in JSON format.

        Args:
            destination: The travel destination (e.g., "Egypt", "Japan")
            duration_days: Number of days for the trip
            traveler_preferences: List of interests (e.g., ["history", "culture"])
            trip_style: Type of trip (e.g., "luxury", "budget", "backpacking")
            pace: Trip pace ("relaxed", "moderate", "packed")

        Returns:
            Dictionary containing the structured trip plan
        """
        # Generate the prompt
        prompt = self._create_prompt(
            destination=destination,
            duration_days=duration_days,
            traveler_preferences=traveler_preferences,
            trip_style=trip_style,
            pace=pace
        )

        # Generate and validate the response
        try:
            response = self.model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.5,
                    "top_p": 0.9,
                    "max_output_tokens": 4096,
                }
            )

            # Extract the text response
            response_text = response.text

            # Clean markdown formatting if present
            if response_text.startswith('```json'):
                response_text = response_text[7:-3]  # Remove ```json and trailing ```
            elif response_text.startswith('```'):
                response_text = response_text[3:-3]  # Remove ``` and trailing ```

            return json.loads(response_text.strip())
        except json.JSONDecodeError as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to parse JSON response from Gemini: {str(e)}"
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to generate trip plan: {str(e)}"
            )

    def _create_prompt(
        self,
        destination: str,
        duration_days: int,
        traveler_preferences: list,
        trip_style: str,
        pace: str
    ) -> str:
        """Create the detailed prompt for the AI."""
        return f"""You are a professional travel planner specializing in generating machine-readable JSON itineraries.

Create a detailed {duration_days}-day {trip_style} trip itinerary for {destination} with a {pace} pace.
Traveler preferences: {', '.join(traveler_preferences)}.

IMPORTANT INSTRUCTIONS:
1. Format your response as a perfect JSON object
2. Do not include any text outside the JSON object
3. Escape all special characters
4. Follow this exact structure:

{{
  "trip_name": "string",
  "destination": "string",
  "duration_days": number,
  "trip_style": "string",
  "pace": "string",
  "traveler_preferences": ["string"],
  "itinerary": [
    {{
      "day": number,
      "activities": [
        {{
          "time": "string (e.g., 09:00-11:00)",
          "time_window": "string (morning/afternoon/evening)",
          "place": "string",
          "description": "string",
          "duration": "string",
          "transportation": "string",
          "notes": "string (optional)"
        }}
      ]
    }}
  ],
  "estimated_costs": {{
    "currency": "string",
    "accommodation": "string",
    "meals": "string",
    "transportation": "string",
    "activities": "string",
    "total_estimate": "string"
  }},
  "packing_recommendations": ["string"],
  "travel_tips": ["string"]
}}

Example of valid time formats:
- "09:00-11:00"
- "14:30-16:00"
- "19:00-22:00"

Example of duration formats:
- "2 hours"
- "30 minutes"
- "Full day"

DO NOT include any additional text or explanations outside the JSON object."""

# Get API key from environment
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file")

# init trip planner with api key
planner = GeminiTripPlanner(api_key=API_KEY)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Gemini Trip Planner API"}

@app.post("/generate-itinerary", response_model=Dict[str, Any])
async def generate_itinerary(request: TripRequest):
    """
    Generate a travel itinerary based on user preferences.
    
    Parameters:
    - destination: The travel destination (e.g., "Japan", "France")
    - duration_days: Number of days for the trip
    - traveler_preferences: List of interests (e.g., ["history", "food"])
    - trip_style: Type of trip ("luxury", "budget", "backpacking", etc.)
    - pace: Trip pace ("relaxed", "moderate", "packed")
    
    Returns:
    - JSON itinerary with detailed daily plans
    """
    try:
        itinerary = planner.generate_trip_plan(
            destination=request.destination,
            duration_days=request.duration_days,
            traveler_preferences=request.traveler_preferences,
            trip_style=request.trip_style,
            pace=request.pace
        )
        return itinerary
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))