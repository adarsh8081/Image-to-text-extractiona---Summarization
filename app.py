import os
# Set PyDev debugger timeout to a higher value
os.environ['PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT'] = '2.0'

from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from text_extraction_and_summarization import TextExtractorAndSummarizer

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize the extractor and summarizer
extractor = TextExtractorAndSummarizer()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file uploaded'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Extract text from image
            extracted_text = extractor.extract_text_from_image(filepath)
            
            if not extracted_text:
                return jsonify({'error': 'Failed to extract text from image'}), 400
            
            # Generate summary
            summary = extractor.generate_summary(extracted_text)
            
            # Clean up the uploaded file
            os.remove(filepath)
            
            return jsonify({
                'extracted_text': extracted_text,
                'summary': summary if summary else 'Could not generate summary'
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) 