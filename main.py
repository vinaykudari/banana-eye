import os
import base64
import io
import logging
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
import requests
from PIL import Image
import google.generativeai as genai
import uvicorn
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        logger.error("Mapbox access token not configured")
        raise HTTPException(status_code=500, detail="Mapbox access token not configured")
    
    # Mapbox Static Images API URL for satellite imagery - include token in URL
    url = f"https://api.mapbox.com/styles/v1/mapbox/satellite-v9/static/{lon},{lat},{zoom}/{width}x{height}?access_token={mapbox_token}"
    logger.info(f"Requesting Mapbox image from: {url[:100]}...")
    
    response = requests.get(url)
    logger.info(f"Mapbox response status: {response.status_code}")
    
    if response.status_code != 200:
        logger.error(f"Mapbox API error: {response.text}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch map image: {response.text}")
    
    logger.info(f"Successfully fetched Mapbox image, content-type: {response.headers.get('content-type')}")
    return response.content

def generate_enhanced_aerial_view(image_bytes: bytes, text_prompt: str, year: int, altitude: int) -> bytes:
    """Generate enhanced aerial view image using Google AI Studio Gemini 2.5 Flash Image Preview"""
    if not gemini_api_key:
        logger.error("Gemini API key not configured")
        raise HTTPException(status_code=500, detail="Gemini API key not configured")
    
    try:
        logger.info("Creating Gemini model instance")
        model = genai.GenerativeModel("gemini-2.5-flash-image-preview")
        
        image = Image.open(io.BytesIO(image_bytes))
        logger.info(f"Converted image to PIL format: {image.size}, mode: {image.mode}")
        
        enhanced_prompt = f"""
        Based on the provided satellite image, generate a **realistic aerial landscape photograph** of the same geographical location.

        **Perspective and Altitude:**
        * Simulate a view as if captured from a professional drone or aircraft camera at an altitude of approximately {altitude} meters.
        * Focus on presenting a **sweeping landscape vista**, similar to a postcard or travel photograph, rather than a flat, top-down map view.
        * Include the horizon and a sense of depth, capturing the natural beauty and significant features of the area.

        **Realism and Detail:**
        * Depict the scene with **high-quality, photorealistic detail** appropriate for the year {year}.
        * Incorporate realistic lighting, atmospheric effects (e.g., subtle haze, clear skies), and natural colors.
        * Emphasize the geographical context, showing how different elements (land, water, structures) interact within the broader landscape.



        Generate a breathtaking aerial landscape photograph that could be used for tourism or scenic appreciation.
        """
        
        logger.info("Sending request to Gemini API")
        response = model.generate_content([enhanced_prompt, image])
        logger.info("Received response from Gemini API")

        # ================================================================
        # START: ADDED CODE TO CHECK FOR SAFETY BLOCK
        # ================================================================
        # The most reliable way to check for a failed generation is to
        # inspect the prompt_feedback for a block reason.
        if response.prompt_feedback.block_reason:
            error_message = f"Image generation blocked by API. Reason: {response.prompt_feedback.block_reason.name}"
            logger.error(error_message)
            # Also log the safety ratings for more detail
            logger.error(f"Safety Ratings: {response.prompt_feedback.safety_ratings}")
            # Raise an exception that will be caught by the endpoint
            raise ValueError(error_message)
        # ================================================================
        # END: ADDED CODE
        # ================================================================

        # Your original logic for extracting the image was good, but now it's
        # protected by the safety check above.
        if response.parts:
            for part in response.parts:
                if part.inline_data and part.inline_data.mime_type.startswith('image/'):
                    logger.info("Found generated image in response.parts")
                    # Decode the base64 data to get the image bytes
                    return part.inline_data.data # The data is already in bytes, no need for b64decode
        
        logger.warning("No generated image found in response, returning original image.")
        return image_bytes
        
    except ValueError as ve:
        # Catch the specific error from the safety block check
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Exception in generate_enhanced_aerial_view: {str(e)}")
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
        logger.info(f"Getting aerial view for coordinates: {request.latitude}, {request.longitude}")
        logger.info(f"Request params - zoom: {request.zoom}, size: {request.width}x{request.height}")
        
        # Get satellite image from Mapbox
        image_bytes = get_mapbox_image(
            lat=request.latitude,
            lon=request.longitude,
            zoom=request.zoom,
            width=request.width,
            height=request.height
        )
        
        logger.info(f"Successfully fetched Mapbox image, size: {len(image_bytes)} bytes")
        
        return Response(
            content=image_bytes,
            media_type="image/jpeg",
            headers={"Content-Disposition": "inline; filename=aerial_view.jpg"}
        )
        
    except Exception as e:
        logger.error(f"Error in get_aerial_view_endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-enhanced-aerial-view")
async def generate_enhanced_aerial_view_endpoint(request: AerialViewRequest):
    """
    Generate enhanced aerial view image using Vertex AI Gemini 2.5 Flash Image Preview
    Returns image directly viewable in Postman
    """
    try:
        logger.info(f"Generating enhanced aerial view for coordinates: {request.latitude}, {request.longitude}")
        logger.info(f"Request params - altitude: {request.altitude}m, year: {request.year}")
        logger.info(f"Text prompt: {request.text_prompt}")
        
        # Get satellite image from Mapbox
        image_bytes = get_mapbox_image(
            lat=request.latitude,
            lon=request.longitude,
            zoom=request.zoom,
            width=request.width,
            height=request.height
        )
        
        logger.info(f"Fetched Mapbox image, size: {len(image_bytes)} bytes")
        
        # Generate enhanced aerial view image using Vertex AI
        enhanced_image_bytes = generate_enhanced_aerial_view(
            image_bytes=image_bytes,
            text_prompt=request.text_prompt,
            year=request.year,
            altitude=request.altitude
        )
        
        logger.info(f"Generated enhanced image, size: {len(enhanced_image_bytes)} bytes")
        
        return Response(
            content=enhanced_image_bytes,
            media_type="image/jpeg",
            headers={"Content-Disposition": "inline; filename=enhanced_aerial_view.jpg"}
        )
        
    except Exception as e:
        logger.error(f"Error in generate_enhanced_aerial_view_endpoint: {str(e)}")
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
