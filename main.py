import os
import base64
import io
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
import requests
from PIL import Image
import vertexai
from vertexai.generative_models import GenerativeModel, Part
from vertexai.preview.vision_models import ImageGenerationModel
import uvicorn
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = FastAPI(title="Banana Eye", description="Aerial view generator using Vertex AI and Mapbox")

# Initialize Vertex AI
project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")

if project_id:
    vertexai.init(project=project_id, location=location)

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
    """Generate enhanced aerial view image using Vertex AI Imagen based on Mapbox satellite image"""
    if not project_id:
        raise HTTPException(status_code=500, detail="Google Cloud project not configured")
    
    try:
        # First, analyze the satellite image with Gemini to understand the location
        analysis_model = GenerativeModel("gemini-2.0-flash-exp")
        image_part = Part.from_data(image_bytes, mime_type="image/jpeg")
        
        analysis_prompt = f"""
        Analyze this satellite image and describe the key geographical features, landmarks, and urban/natural elements visible. 
        Focus on details that would be important for generating an aerial view from {altitude} meters altitude in the year {year}.
        Be specific about terrain, buildings, water bodies, vegetation, and any notable structures.
        Keep the description concise but detailed.
        """
        
        analysis_response = analysis_model.generate_content([analysis_prompt, image_part])
        location_description = analysis_response.text
        
        # Generate enhanced aerial view using Imagen
        imagen_model = ImageGenerationModel.from_pretrained("imagegeneration@006")
        
        enhanced_prompt = f"""
        Create a realistic aerial photograph taken from {altitude} meters altitude showing:
        {location_description}
        
        Additional context: {text_prompt}
        Time period: {year}
        
        Style: High-resolution aerial photography, clear visibility, natural lighting, realistic colors and shadows.
        The image should look like it was taken from a drone or aircraft at the specified altitude.
        Show the same geographical area with enhanced detail and perspective appropriate for the altitude.
        """
        
        # Generate the image
        images = imagen_model.generate_images(
            prompt=enhanced_prompt,
            number_of_images=1,
            aspect_ratio="1:1",
            safety_filter_level="allow_most",
            person_generation="dont_allow"
        )
        
        # Convert generated image to bytes
        generated_image = images[0]
        img_byte_arr = io.BytesIO()
        generated_image._pil_image.save(img_byte_arr, format='JPEG', quality=95)
        return img_byte_arr.getvalue()
        
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
        "google_cloud_project": bool(os.getenv("GOOGLE_CLOUD_PROJECT")),
        "vertex_ai": project_id is not None
    }
    
    return {
        "status": "healthy" if all(checks.values()) else "degraded",
        "checks": checks
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
