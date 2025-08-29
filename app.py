from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import daft
from pipeline import process_dataframe_pipeline
from pathlib import Path
import uuid
import os
from datetime import datetime

app = Flask(__name__)

@app.route('/')
def index():
    """Serve the main UI page"""
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_coordinates():
    """Process the coordinates through the Daft pipeline for multiple years"""
    try:
        # Get data from the form
        data = request.get_json()
        latitude = float(data['latitude'])
        longitude = float(data['longitude'])
        altitude = float(data['altitude'])
        
        # Generate a unique batch ID for this request
        batch_id = f"web_{uuid.uuid4().hex[:8]}"
        
        # Years to process
        years = [1850, 1950, 2000, 2020]
        
        # Create a DataFrame with multiple rows (one for each year)
        df_data = {
            'batchID': [batch_id] * len(years),
            'lattitude': [latitude] * len(years),
            'longitude': [longitude] * len(years),
            'altitude': [altitude] * len(years),
            'year': years
        }
        
        # Convert to Daft DataFrame
        pdf = pd.DataFrame(df_data)
        df = daft.from_pandas(pdf)
        
        # Process through the pipeline
        result_df = process_dataframe_pipeline(df)
        
        # Get the results - convert to pandas first for easier access
        results_pdf = result_df.to_pandas()
        
        if len(results_pdf) > 0:
            # Check if all files were created successfully
            successful_years = []
            failed_years = []
            
            for _, row in results_pdf.iterrows():
                year = row['year']
                image_path = row['image_path']
                
                if os.path.exists(image_path) and not str(image_path).startswith('ERROR:'):
                    successful_years.append(year)
                else:
                    failed_years.append(year)
            
            if successful_years:
                return jsonify({
                    'success': True,
                    'batch_id': batch_id,
                    'years': successful_years,
                    'failed_years': failed_years,
                    'coordinates': {
                        'latitude': latitude,
                        'longitude': longitude,
                        'altitude': altitude
                    },
                    'message': f'Successfully processed {len(successful_years)} years for coordinates ({latitude}, {longitude}, {altitude})'
                })
            else:
                return jsonify({
                    'success': False,
                    'error': 'Failed to create any files'
                })
        else:
            return jsonify({
                'success': False,
                'error': 'No results returned from pipeline'
            })
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/results/<batch_id>')
def results_page(batch_id):
    """Serve the results page with year slider"""
    return render_template('results.html', batch_id=batch_id)

@app.route('/api/get_image/<batch_id>/<int:year>')
def get_image_content(batch_id, year):
    """Get image content for a specific year"""
    try:
        # Find the image file for this batch and year
        batch_dir = Path('output') / batch_id
        if not batch_dir.exists():
            return jsonify({'error': 'Batch not found'}), 404
        
        # Look for files matching the pattern
        image_files = list(batch_dir.glob(f'*_{year}.png'))
        if not image_files:
            return jsonify({'error': f'No image found for year {year}'}), 404
        
        image_path = image_files[0]
        
        # Check if it's an image file or text file
        if image_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
            # For image files, encode as base64 for web display
            with open(image_path, 'rb') as f:
                image_data = f.read()
            import base64
            content_base64 = base64.b64encode(image_data).decode('utf-8')
            
            return jsonify({
                'success': True,
                'year': year,
                'filename': image_path.name,
                'content_type': 'image',
                'content': content_base64,
                'path': str(image_path)
            })
        else:
            # For text files, read as text
            with open(image_path, 'r') as f:
                content = f.read()
            
            return jsonify({
                'success': True,
                'year': year,
                'filename': image_path.name,
                'content_type': 'text',
                'content': content,
                'path': str(image_path)
            })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/batch_info/<batch_id>')
def get_batch_info(batch_id):
    """Get information about a batch (available years, coordinates)"""
    try:
        batch_dir = Path('output') / batch_id
        if not batch_dir.exists():
            return jsonify({'error': 'Batch not found'}), 404
        
        # Find all image files and extract years
        image_files = list(batch_dir.glob('*.png'))
        available_years = []
        coordinates = {}
        
        for image_file in image_files:
            # Parse filename to extract year and coordinates
            # Format: img_lat_lon_alt_year.png
            parts = image_file.stem.split('_')
            if len(parts) >= 5:
                try:
                    year = int(parts[-1])
                    lat = float(parts[1])
                    lon = float(parts[2])
                    alt = float(parts[3])
                    
                    available_years.append(year)
                    if not coordinates:  # Use first file to get coordinates
                        coordinates = {
                            'latitude': lat,
                            'longitude': lon,
                            'altitude': alt
                        }
                except (ValueError, IndexError):
                    continue
        
        available_years.sort()
        
        return jsonify({
            'success': True,
            'batch_id': batch_id,
            'available_years': available_years,
            'coordinates': coordinates
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download/<path:filename>')
def download_file(filename):
    """Allow downloading of generated files"""
    try:
        file_path = Path('output') / filename
        if file_path.exists():
            return send_file(file_path, as_attachment=True)
        else:
            return jsonify({'error': 'File not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/list_outputs')
def list_outputs():
    """List all generated output files"""
    try:
        output_dir = Path('output')
        if not output_dir.exists():
            return jsonify({'files': []})
        
        files = []
        for batch_dir in output_dir.iterdir():
            if batch_dir.is_dir():
                for file_path in batch_dir.glob('*.png'):
                    relative_path = file_path.relative_to(output_dir)
                    files.append({
                        'batch_id': batch_dir.name,
                        'filename': file_path.name,
                        'path': str(relative_path),
                        'size': file_path.stat().st_size,
                        'modified': file_path.stat().st_mtime
                    })
        
        return jsonify({'files': files})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Ensure templates directory exists
    Path('templates').mkdir(exist_ok=True)
    
    print("Starting Flask web server...")
    print("Open your browser to: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
