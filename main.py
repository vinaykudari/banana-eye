import os
import base64
import io
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
import requests
from PIL import Image
import google.generativeai as genai
import uvicorn
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = FastAPI(title="Banana Eye", description="Aerial view generator using Vertex AI and Mapbox")

# Initialize Google AI Studio
gemini_api_key = os.getenv("GEMINI_API_KEY")
if gemini_api_key:
    genai.configure(api_key=gemini_api_key)

class AerialViewRequest(BaseModel):
    latitude: float
    longitude: float
    text_prompt: str
    year: Optional[int] = 2024
    altitude: Optional[int] = 1000  # meters
    zoom: Optional[int] = 15
    width: Optional[int] = 512
    height: Optional[int] = 512

class AerialViewImageResponse(BaseModel):
    latitude: float
    longitude: float
    year: int
    altitude: int
    image_base64: str
    status: str

def get_mapbox_image(lat: float, lon: float, zoom: int = 15, width: int = 512, height: int = 512) -> bytes:
    """Get satellite image from Mapbox Static Images API"""
    mapbox_token = os.getenv("MAPBOX_ACCESS_TOKEN")
    if not mapbox_token:
        raise HTTPException(status_code=500, detail="Mapbox access token not configured")
    
    # Mapbox Static Images API URL for satellite imagery - include token in URL
    url = f"https://api.mapbox.com/styles/v1/mapbox/satellite-v9/static/{lon},{lat},{zoom}/{width}x{height}?access_token={mapbox_token}"
    
    response = requests.get(url)
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail=f"Failed to fetch map image: {response.text}")
    
    return response.content

def generate_enhanced_aerial_view(image_bytes: bytes, text_prompt: str, year: int, altitude: int) -> bytes:
    """Generate enhanced aerial view image using Google AI Studio Gemini 2.5 Flash Image Preview"""
    if not gemini_api_key:
        raise HTTPException(status_code=500, detail="Gemini API key not configured")
    
    try:
        # Create Gemini client
        model = genai.GenerativeModel("gemini-2.5-flash-image-preview")
        
        # Convert image bytes to PIL Image for Gemini
        image = Image.open(io.BytesIO(image_bytes))
        
        enhanced_prompt = f"""
        Based on this satellite image, generate a realistic aerial photograph taken from {altitude} meters altitude in the year {year}.
        
        User request: {text_prompt}
        
        Please create an enhanced aerial view image that shows:
        - The same geographical location but from the specified altitude perspective
        - Realistic lighting and atmospheric effects for the given altitude
        - Enhanced detail and clarity appropriate for the year {year}
        - Incorporate the user's specific request: {text_prompt}
        
        Generate a high-quality aerial photograph that looks like it was captured by a professional drone or aircraft camera.
        """
        
        # Generate content with image input
        response = model.generate_content(
            [enhanced_prompt, image]
        )
        
        # Extract the generated image from the response
        for part in response.parts:
            if hasattr(part, 'inline_data') and part.inline_data:
                if part.inline_data.mime_type.startswith('image/'):
                    return base64.b64decode(part.inline_data.data)
        
        # If no image found in response, raise an error
        raise HTTPException(status_code=500, detail="No image generated in response")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate enhanced aerial view: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Banana Eye - Aerial View Generator", "status": "running"}

@app.post("/get-aerial-view")
async def get_aerial_view_endpoint(request: AerialViewRequest):
    """
    Get basic aerial view image from Mapbox satellite imagery
    Returns image directly viewable in Postman
    """
    try:
        # Get satellite image from Mapbox
        image_bytes = get_mapbox_image(
            lat=request.latitude,
            lon=request.longitude,
            zoom=request.zoom,
            width=request.width,
            height=request.height
        )
        
        return Response(
            content=image_bytes,
            media_type="image/jpeg",
            headers={"Content-Disposition": "inline; filename=aerial_view.jpg"}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-enhanced-aerial-view")
async def generate_enhanced_aerial_view_endpoint(request: AerialViewRequest):
    """
    Generate enhanced aerial view image using Vertex AI Gemini 2.5 Flash Image Preview
    Returns image directly viewable in Postman
    """
    try:
        # Get satellite image from Mapbox
        image_bytes = get_mapbox_image(
            lat=request.latitude,
            lon=request.longitude,
            zoom=request.zoom,
            width=request.width,
            height=request.height
        )
        
        # Generate enhanced aerial view image using Vertex AI
        enhanced_image_bytes = generate_enhanced_aerial_view(
            image_bytes=image_bytes,
            text_prompt=request.text_prompt,
            year=request.year,
            altitude=request.altitude
        )
        
        return Response(
            content=enhanced_image_bytes,
            media_type="image/jpeg",
            headers={"Content-Disposition": "inline; filename=enhanced_aerial_view.jpg"}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    checks = {
        "mapbox_token": bool(os.getenv("MAPBOX_ACCESS_TOKEN")),
        "gemini_api_key": bool(os.getenv("GEMINI_API_KEY"))
    }
    
    return {
        "status": "healthy" if all(checks.values()) else "degraded",
        "checks": checks
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
