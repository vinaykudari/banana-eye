# Banana Eye 🍌👁️

A FastAPI server that generates aerial view descriptions using Vertex AI Gemini 2.5 Flash Image Preview and Mapbox satellite imagery.

## Features

- **Coordinate-based aerial views**: Input latitude/longitude to get satellite imagery
- **AI-powered descriptions**: Uses Vertex AI Gemini to analyze and describe aerial views
- **Customizable parameters**: Specify year, altitude, zoom level, and image dimensions
- **Mapbox integration**: High-quality satellite imagery from Mapbox
- **Health monitoring**: Built-in health check endpoint

## Setup

### Prerequisites

- Python 3.11+
- UV package manager (from astral.sh)
- Google Cloud Project with Vertex AI enabled
- Mapbox account with access token

### Installation

1. Clone and navigate to the project:
```bash
cd banana-eye
```

2. Install dependencies using UV:
```bash
uv sync
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your actual values
```

### Environment Variables

Required environment variables:

- `GOOGLE_CLOUD_PROJECT`: Your GCP project ID
- `GOOGLE_CLOUD_LOCATION`: GCP region (default: us-central1)
- `MAPBOX_ACCESS_TOKEN`: Your Mapbox access token

Optional:
- `GOOGLE_APPLICATION_CREDENTIALS`: Path to service account key (if not using default credentials)

### Google Cloud Setup

1. Enable Vertex AI API in your GCP project
2. Set up authentication (service account or gcloud auth)
3. Ensure your project has access to Gemini 2.5 Flash Image Preview

### Mapbox Setup

1. Create account at mapbox.com
2. Get your access token from the account dashboard
3. Add token to environment variables

## Usage

### Start the server

```bash
uv run python main.py
```

The server will start on `http://localhost:8000`

### API Endpoints

#### Generate Aerial View
```http
POST /generate-aerial-view
```

Request body:
```json
{
  "latitude": 37.7749,
  "longitude": -122.4194,
  "text_prompt": "Describe the urban landscape and notable landmarks",
  "year": 2024,
  "altitude": 1000,
  "zoom": 15,
  "width": 512,
  "height": 512
}
```

Response:
```json
{
  "latitude": 37.7749,
  "longitude": -122.4194,
  "year": 2024,
  "altitude": 1000,
  "description": "AI-generated description of the aerial view...",
  "status": "success"
}
```

#### Health Check
```http
GET /health
```

Returns service status and configuration checks.

### Example Usage

```bash
curl -X POST "http://localhost:8000/generate-aerial-view" \
  -H "Content-Type: application/json" \
  -d '{
    "latitude": 37.7749,
    "longitude": -122.4194,
    "text_prompt": "What landmarks can you see from this aerial view?",
    "year": 2024,
    "altitude": 500
  }'
```

## API Documentation

Once the server is running, visit:
- Interactive docs: `http://localhost:8000/docs`
- OpenAPI spec: `http://localhost:8000/openapi.json`

## Development

### Project Structure

```
banana-eye/
├── main.py              # FastAPI application
├── pyproject.toml       # UV project configuration
├── .env.example         # Environment variables template
└── README.md           # This file
```

### Key Components

- **FastAPI server**: Handles HTTP requests and responses
- **Mapbox integration**: Fetches satellite imagery based on coordinates
- **Vertex AI client**: Processes images with Gemini 2.5 Flash Image Preview
- **Pydantic models**: Request/response validation

## Troubleshooting

### Common Issues

1. **"Google Cloud project not configured"**
   - Ensure `GOOGLE_CLOUD_PROJECT` is set
   - Verify GCP authentication is working

2. **"Mapbox access token not configured"**
   - Check `MAPBOX_ACCESS_TOKEN` environment variable
   - Verify token is valid and has required permissions

3. **"Failed to generate aerial view"**
   - Check Vertex AI API is enabled
   - Verify your project has access to Gemini models
   - Check GCP quotas and billing

### Health Check

Use the health endpoint to verify configuration:
```bash
curl http://localhost:8000/health
```

## License

MIT License