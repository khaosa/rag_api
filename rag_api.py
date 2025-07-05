from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any
import json
from datetime import datetime, date, timedelta
from decimal import Decimal
import google.generativeai as genai
from dotenv import load_dotenv
import os
import mysql.connector

# Load environment variables
load_dotenv()

# Get Gemini API key and MySQL credentials from .env
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MYSQL_HOST = os.getenv("MYSQL_HOST")
MYSQL_USER = os.getenv("MYSQL_USER")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env")
if not (MYSQL_HOST and MYSQL_USER and MYSQL_PASSWORD and MYSQL_DATABASE):
    raise ValueError("MySQL credentials not fully found in .env")

# Connect to MySQL database
def get_db_connection():
    return mysql.connector.connect(
        host=MYSQL_HOST,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DATABASE,
        port=3306
    )

app = FastAPI(
    title="Gemini Trip Planner API",
    description="API for generating travel itineraries using Google's Gemini AI",
    version="1.0.0"
)

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        elif isinstance(obj, (datetime, date)):
            return obj.isoformat()
        elif isinstance(obj, timedelta):
            return str(obj)
        return super().default(obj)

class TripRequest(BaseModel):
    destination: str
    duration_days: int
    traveler_preferences: List[str]
    trip_style: str = "luxury"
    pace: str = "moderate"

class GeminiTripPlanner:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.api_key = api_key
        self.model = genai.GenerativeModel("gemini-2.0-flash-lite")

    def generate_trip_plan(
        self,
        destination: str,
        duration_days: int,
        traveler_preferences: list,
        trip_style: str = "luxury",
        pace: str = "moderate",
        db_places: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:

        prompt = self._create_prompt(
            destination=destination,
            duration_days=duration_days,
            traveler_preferences=traveler_preferences,
            trip_style=trip_style,
            pace=pace,
            db_places=db_places
        )

        try:
            response = self.model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.5,
                    "top_p": 0.9,
                    "max_output_tokens": 8192,
                }
            )
            response_text = response.text

            if response_text.startswith('```json'):
                response_text = response_text[7:-3]
            elif response_text.startswith('```'):
                response_text = response_text[3:-3]

            parsed = json.loads(response_text.strip())
            return parsed

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
        pace: str,
        db_places: List[Dict[str, Any]] = None
    ) -> str:
        places_section = ""
        if db_places:
            places_json = json.dumps(db_places, indent=2, ensure_ascii=False, cls=CustomJSONEncoder)
            places_section = f"\n\nAvailable Places (use these JSON objects directly in the itinerary `place` field):\n{places_json}\n"
        return f"""You are a professional travel planner specializing in generating machine-readable JSON itineraries.

Create a detailed {duration_days}-day {trip_style} trip itinerary for {destination} with a {pace} pace.
Traveler preferences: {', '.join(traveler_preferences)}.{places_section}

IMPORTANT INSTRUCTIONS:
1. Format your response as a perfect JSON object
2. Do not include any text outside the JSON object
3. Escape all special characters
4. Please adjust the number of activities and free time based on the selected pace:
- For fast-paced trips: include more activities with minimal but realistic breaks for transportation and rest. The number of places to visit must be at least 7.
- For moderate pace: balance activities and free time reasonably. The number of places to visit must be at least 5.
- For slow-paced: fewer activities with more room for rest and exploration. The number of places to visit must be at least 3.
5. Always consider reasonable transportation time between places, especially for fast-paced itineraries, to avoid unrealistic schedules.
6. Do not return the same place more than once.
7. Follow this exact structure:

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
          "id": int (should be the id of the place coming from places array sent to in the prompt),
          "time": "string (e.g., 09:00-11:00)",
          "time_window": "string (morning/afternoon/evening)",
          "place": "string",
          "description": "string",
          "duration": "string",
          "notes": "string (optional)",
          "place_label": "string (should be the place_label of the place coming from places array sent to in the prompt)",
          "parent_label": "string (should be the parent_label of the place coming from places array sent to in the prompt)",
          "latitude": decimal number (should be the latitude of the place coming from places array sent to in the prompt),
          "longitude": decimal number (should be the longitude of the place coming from places array sent to in the prompt),
          "image_url": string (should be the image_url of the place coming from places array sent to in the prompt),
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

# Initialize planner instance
planner = GeminiTripPlanner(api_key=GEMINI_API_KEY)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Gemini Trip Planner API"}

def serialize_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Convert non-serializable DB row values to JSON-serializable formats."""
    return {
        k: (
            v.isoformat() if isinstance(v, (datetime, date)) else
            float(v) if isinstance(v, Decimal) else
            str(v) if isinstance(v, timedelta) else
            v
        )
        for k, v in row.items()
    }

@app.post("/generate-itinerary", response_model=Dict[str, Any])
async def generate_itinerary(request: TripRequest):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        placeholders = ', '.join(['%s'] * len(request.traveler_preferences))
        query = f"""
            SELECT DISTINCT p.id, p.name, p.longitude, p.latitude, p.city, p.country, p.country_id, p.open_hours, p.rating, 
            p.number_of_ratings, p.created_at, p.updated_at, p.website, p.phone, p.price_range, pl.place_id, pl.label_id,
            l.label_name as place_label, l.parent_id, l2.label_name as parent_label, pli.img_url as image_url
            FROM places p
            JOIN places_labels pl ON p.id = pl.place_id
            JOIN places_images pli on p.id = pli.place_id 
            JOIN labels l ON pl.label_id = l.id
            JOIN labels l2 ON l.parent_id = l2.id
            WHERE p.city = %s
            AND (
                l.label_name IN ({placeholders})  -- Direct matches
                OR 
                l.parent_id IN (
                    SELECT l.id
                    FROM labels l 
                    WHERE l.label_name IN ({placeholders})  -- Parent matches
                )
                )
                """
        params = [request.destination] + request.traveler_preferences + request.traveler_preferences
        cursor.execute(query, params)

        columns = [col[0] for col in cursor.description]
        rows = cursor.fetchall()
        cursor.close()
        conn.close()

        places = [serialize_row(dict(zip(columns, row))) for row in rows]
        print(places)
        print("END OF PLACES")
        itinerary = planner.generate_trip_plan(
            destination=request.destination,
            duration_days=request.duration_days,
            traveler_preferences=request.traveler_preferences,
            trip_style=request.trip_style,
            pace=request.pace,
            db_places=places
        )
        print(itinerary)
        # Return using CustomJSONEncoder in case of datetime or others
        return JSONResponse(content=json.loads(json.dumps(itinerary, cls=CustomJSONEncoder)))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
