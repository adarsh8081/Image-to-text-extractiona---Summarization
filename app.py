import os
# Set PyDev debugger timeout to a higher value
os.environ['PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT'] = '2.0'

from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from text_extraction_and_summarization import TextExtractorAndSummarizer
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload folder exists with proper permissions
try:
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    logger.info(f"Upload folder created/verified at: {UPLOAD_FOLDER}")
except Exception as e:
    logger.error(f"Error creating upload folder: {str(e)}")
    raise

# Initialize the extractor and summarizer
extractor = TextExtractorAndSummarizer()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_image():
    try:
        if 'image' not in request.files:
            logger.error("No image file in request")
            return jsonify({'error': 'No image file uploaded'}), 400
        
        file = request.files['image']
        if file.filename == '':
            logger.error("No selected file")
            return jsonify({'error': 'No selected file'}), 400
        
        if not file:
            logger.error("Invalid file")
            return jsonify({'error': 'Invalid file'}), 400

        # Check file extension
        allowed_extensions = {'png', 'jpg', 'jpeg'}
        if '.' not in file.filename or file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
            logger.error(f"Invalid file type: {file.filename}")
            return jsonify({'error': 'Invalid file type. Allowed types: PNG, JPG, JPEG'}), 400
        
        # Save file securely
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        logger.info(f"File saved temporarily at: {filepath}")
        
        try:
            # Extract text from image
            extracted_text = extractor.extract_text_from_image(filepath)
            
            if not extracted_text:
                logger.error("Failed to extract text from image")
                return jsonify({'error': 'Failed to extract text from image'}), 400
            
            # Generate summary
            summary = extractor.generate_summary(extracted_text)
            
            # Clean up the uploaded file
            try:
                os.remove(filepath)
                logger.info("Temporary file removed")
            except Exception as e:
                logger.warning(f"Could not remove temporary file: {str(e)}")
            
            return jsonify({
                'extracted_text': extracted_text,
                'summary': summary if summary else 'Could not generate summary'
            })
            
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            return jsonify({'error': f'Error processing image: {str(e)}'}), 500
        finally:
            # Ensure temporary file is removed even if an error occurs
            try:
                if os.path.exists(filepath):
                    os.remove(filepath)
            except Exception as e:
                logger.warning(f"Could not remove temporary file in finally block: {str(e)}")
                
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return jsonify({'error': 'An unexpected error occurred'}), 500

if __name__ == '__main__':
    app.run(debug=True) 