import daft
import os
from pathlib import Path
from typing import Any, Dict
import logging
from daft import col
from main import get_mapbox_image, generate_enhanced_aerial_view

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_row_to_image(batch_id: str, latitude: float, longitude: float, altitude: float, year: int) -> str:
    """
    Function that processes a single row and generates an image.
    
    Args:
        batch_id: The batch identifier for organizing outputs
        latitude: Geographic latitude coordinate
        longitude: Geographic longitude coordinate  
        altitude: Altitude value
        year: Year value
        
    Returns:
        str: Path to the generated image file
    """
    # Create directory for this batch_id if it doesn't exist
    batch_dir = Path(f"output/{batch_id}")
    batch_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate a unique filename based on the coordinates and year
    filename = f"img_{latitude}_{longitude}_{altitude}_{year}.png"
    image_path = batch_dir / filename
    
    # TODO: Replace this with your actual image generation logic
    # For now, we'll create a placeholder file
    logger.info(f"Generating image for batch {batch_id} at ({latitude}, {longitude}, {altitude}) for year {year}")
    
    # Generate enhanced aerial view image
    create_enhanced_image(image_path, latitude, longitude, altitude, year)
    
    return str(image_path)

def create_enhanced_image(image_path: Path, lat: float, lon: float, alt: float, year: int):
    """
    Generate enhanced aerial view image using AI and save to file.
    """
    try:
        # Get satellite image from Mapbox
        satellite_image_bytes = get_mapbox_image(
            lat=lat,
            lon=lon,
            zoom=14,
            width=512,
            height=512
        )
        
        # Generate enhanced aerial view using AI
        enhanced_image_bytes = generate_enhanced_aerial_view(
            image_bytes=satellite_image_bytes,
            year=year,
            altitude=alt
        )

        
        # Save the enhanced image
        with open(image_path, 'wb') as f:
            f.write(enhanced_image_bytes)
            
        logger.info(f"Successfully generated enhanced aerial view for ({lat}, {lon}) at {alt}m altitude in year {year}")
        
    except Exception as e:
        logger.error(f"Failed to generate enhanced image: {e}")
        # Fallback to a simple text file if image generation fails
        with open(image_path, 'w') as f:
            f.write(f"Enhanced Aerial View Data:\n")
            f.write(f"Latitude: {lat}\n")
            f.write(f"Longitude: {lon}\n")
            f.write(f"Altitude: {alt}m\n")
            f.write(f"Year: {year}\n")
            f.write(f"Error: {str(e)}\n")

@daft.udf(return_dtype=daft.DataType.string())
def generate_image_path_udf(batch_id, latitude, longitude, altitude, year):
    """
    Daft UDF (User Defined Function) for generating image paths in parallel.
    This function will be executed in parallel across batches of rows.
    """
    import pyarrow as pa
    
    # Convert inputs to Python lists for processing
    batch_ids = batch_id.to_pylist()
    latitudes = latitude.to_pylist()
    longitudes = longitude.to_pylist()
    altitudes = altitude.to_pylist()
    years = year.to_pylist()
    
    result_paths = []
    
    # Process each row in the batch
    for bid, lat, lon, alt, yr in zip(batch_ids, latitudes, longitudes, altitudes, years):
        try:
            # Create directory for this batch_id if it doesn't exist
            batch_dir = Path(f"output/{bid}")
            batch_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate a unique filename based on the coordinates and year
            filename = f"img_{lat}_{lon}_{alt}_{yr}.png"
            image_path = batch_dir / filename
            
            # Generate enhanced aerial view image
            create_enhanced_image(image_path, lat, lon, alt, yr)
            
            result_paths.append(str(image_path))
        except Exception as e:
            result_paths.append(f"ERROR: {str(e)}")
    
    # Return as a pyarrow array
    return pa.array(result_paths)

def process_dataframe_pipeline(df):
    """
    Main pipeline function that processes each row using Daft's native parallel processing.
    
    Args:
        df: Daft DataFrame with columns: batchID, lattitude, longitude, altitude, year
        
    Returns:
        DataFrame with an additional 'image_path' column containing the generated image paths
    """
    logger.info("Starting image generation pipeline...")
    
    # Ensure required columns exist
    required_columns = ['batchID', 'lattitude', 'longitude', 'altitude', 'year']
    df_columns = df.column_names
    
    missing_columns = [col_name for col_name in required_columns if col_name not in df_columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Create the output directory
    Path("output").mkdir(exist_ok=True)
    
    # Use Daft's native parallel processing with UDF
    # This will automatically distribute the work across available cores
    result_df = df.with_column(
        "image_path",
        generate_image_path_udf(
            col("batchID"),
            col("lattitude"), 
            col("longitude"),
            col("altitude"),
            col("year")
        )
    )
    
    logger.info("Pipeline processing complete!")
    return result_df



def create_sample_dataframe():
    """
    Create a sample dataframe for testing the pipeline.
    """
    import pandas as pd
    
    sample_data = {
        'batchID': ['batch_001', 'batch_001', 'batch_002', 'batch_002', 'batch_003'],
        'lattitude': [37.7749, 40.7128, 34.0522, 41.8781, 29.7604],
        'longitude': [-122.4194, -74.0060, -118.2437, -87.6298, -95.3698],
        'altitude': [50, 100, 25, 200, 75],
        'year': [2023, 2023, 2022, 2024, 2023]
    }
    
    pdf = pd.DataFrame(sample_data)
    return daft.from_pandas(pdf)

def main():
    """
    Example usage of the parallel processing pipeline.
    """
    # Create sample data
    df = create_sample_dataframe()
    
    print("Input DataFrame:")
    df.show()
    
    # Process the dataframe
    result_df = process_dataframe_pipeline(df)
    
    print("\nResult DataFrame with image paths:")
    result_df.show()
    
    # Collect results to see the output
    results = result_df.collect()
    print(f"\nProcessed {len(results)} rows successfully!")

if __name__ == "__main__":
    main()