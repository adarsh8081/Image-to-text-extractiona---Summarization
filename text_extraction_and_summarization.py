import os
import pytesseract
from PIL import Image
import cv2
import numpy as np
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import logging
import tempfile

# Download required NLTK data
try:
    nltk.download('punkt')
    nltk.download('stopwords')
except Exception as e:
    print(f"Warning: Could not download NLTK data: {e}")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set Tesseract path for Windows
tesseract_paths = [
    r'C:\Program Files\Tesseract-OCR\tesseract.exe',
    r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
    r'C:\Tesseract-OCR\tesseract.exe'
]

def find_tesseract():
    """Find Tesseract executable"""
    for path in tesseract_paths:
        if os.path.exists(path):
            return path
    return None

class TextExtractorAndSummarizer:
    def __init__(self):
        # Set Tesseract path for Windows
        tesseract_path = find_tesseract()
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
            logger.info(f"Using Tesseract from: {tesseract_path}")
        else:
            logger.warning("Tesseract not found in common locations.")
            logger.warning("Please install Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki")
            logger.warning("Expected locations:")
            for path in tesseract_paths:
                logger.warning(f"- {path}")

    def enhance_image(self, image):
        """
        Enhance image quality for better OCR
        """
        try:
            # Convert to grayscale if not already
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()

            # Apply bilateral filter to preserve edges while removing noise
            denoised = cv2.bilateralFilter(gray, 9, 75, 75)
            
            # Increase contrast using CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            contrast = clahe.apply(denoised)
            
            # Adaptive thresholding with smaller block size for better text detection
            binary = cv2.adaptiveThreshold(
                contrast,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                15,  # Block size
                8    # C constant
            )

            # Remove small noise
            kernel = np.ones((2,2), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

            return binary
        except Exception as e:
            logger.error(f"Error enhancing image: {str(e)}")
            return None

    def try_different_psm(self, image):
        """
        Try different page segmentation modes (PSM) and OCR configurations
        """
        configs = [
            '--oem 3 --psm 1',  # Automatic page segmentation with OSD
            '--oem 3 --psm 3',  # Fully automatic page segmentation, but no OSD
            '--oem 3 --psm 6',  # Assume a uniform block of text
            '--oem 3 --psm 4',  # Assume a single column of text
            '--oem 1 --psm 6',  # Legacy engine with uniform block of text
        ]
        
        best_result = ""
        max_confidence = 0
        
        for config in configs:
            try:
                # Get detailed results including confidence scores
                data = pytesseract.image_to_data(image, config=config, output_type=pytesseract.Output.DICT)
                
                # Calculate average confidence for this configuration
                confidences = [int(conf) for conf in data['conf'] if conf != '-1']
                if confidences:
                    avg_confidence = sum(confidences) / len(confidences)
                    text = ' '.join([word for i, word in enumerate(data['text']) if data['conf'][i] != '-1'])
                    
                    # Keep the result with highest confidence
                    if text.strip() and avg_confidence > max_confidence:
                        max_confidence = avg_confidence
                        best_result = text
                        logger.info(f"Found better result with config {config}, confidence: {avg_confidence:.2f}%")
            except Exception as e:
                logger.warning(f"Error with config {config}: {str(e)}")
                continue
        
        if best_result:
            return best_result.strip()
        return None

    def extract_text_from_image(self, image_path):
        """
        Extract text from an image using Tesseract OCR with improved configuration
        """
        try:
            if not os.path.exists(pytesseract.pytesseract.tesseract_cmd):
                raise Exception(f"Tesseract not found at: {pytesseract.pytesseract.tesseract_cmd}")

            # Read image
            image = cv2.imread(image_path)
            if image is None:
                raise Exception("Could not read image file")

            # Get image dimensions
            height, width = image.shape[:2]
            
            # Resize if image is too small
            min_size = 2000  # Increased minimum size for better OCR
            if width < min_size or height < min_size:
                scale = min_size / min(width, height)
                image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

            # Try OCR on original image first
            logger.info("Attempting OCR on original image...")
            original_text = self.try_different_psm(image)

            # Enhance image and try again
            enhanced_image = self.enhance_image(image)
            if enhanced_image is not None:
                logger.info("Attempting OCR on enhanced image...")
                enhanced_text = self.try_different_psm(enhanced_image)
                
                # Compare results and use the better one
                if original_text and enhanced_text:
                    # Use the longer result as it's likely more complete
                    if len(enhanced_text) > len(original_text):
                        final_text = enhanced_text
                    else:
                        final_text = original_text
                else:
                    final_text = enhanced_text or original_text
                
                if final_text:
                    # Clean up the text
                    cleaned_text = ' '.join(word.strip() for word in final_text.split())
                    return cleaned_text
            
            logger.warning("No text was extracted from the image")
            return None
                        
        except Exception as e:
            logger.error(f"Error extracting text from image: {str(e)}")
            logger.error(f"Image path: {image_path}")
            return None

    def calculate_sentence_scores(self, sentences):
        """
        Calculate importance scores for sentences using TF-IDF and cosine similarity
        """
        if len(sentences) < 2:
            return {0: 1.0} if sentences else {}

        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(stop_words='english')
        try:
            tfidf_matrix = vectorizer.fit_transform(sentences)
        except ValueError:
            # If vectorization fails, return equal scores
            return {i: 1.0 for i in range(len(sentences))}

        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

        # Calculate sentence scores
        scores = {}
        for i in range(len(sentences)):
            # Score is the sum of similarities with other sentences
            scores[i] = sum(similarity_matrix[i]) / len(sentences)

        return scores

    def generate_summary(self, text, max_length=130, min_length=30):
        """
        Generate an intelligent summary using NLP techniques
        """
        try:
            # Clean the text
            text = ' '.join(text.split())
            
            # Tokenize into sentences
            sentences = sent_tokenize(text)
            
            if not sentences:
                return "No text to summarize."

            if len(sentences) == 1:
                return text  # Return the original text if it's just one sentence

            # Calculate sentence scores
            sentence_scores = self.calculate_sentence_scores(sentences)

            # Sort sentences by score
            ranked_sentences = sorted(
                [(score, i, sentence) for i, (sentence, score) in enumerate(zip(sentences, sentence_scores.values()))],
                reverse=True
            )

            # Select top sentences (30% of total or at least 2 sentences)
            num_sentences = max(2, min(len(sentences), max(min_length // 30, len(sentences) // 3)))
            
            # Sort selected sentences by original position
            summary_sentences = sorted(
                ranked_sentences[:num_sentences],
                key=lambda x: x[1]
            )

            # Join sentences
            summary = ' '.join(sentence for _, _, sentence in summary_sentences)
            
            return summary.strip()
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            return None

    def calculate_accuracy(self, original_text, extracted_text):
        """
        Calculate accuracy score between original and extracted text
        """
        try:
            if not original_text or not extracted_text:
                return 0.0

            # Clean and normalize texts
            original_text = ' '.join(original_text.lower().split())
            extracted_text = ' '.join(extracted_text.lower().split())

            # Tokenize texts
            original_tokens = set(word_tokenize(original_text))
            extracted_tokens = set(word_tokenize(extracted_text))

            # Remove stopwords
            stop_words = set(stopwords.words('english'))
            original_tokens = {w for w in original_tokens if w not in stop_words}
            extracted_tokens = {w for w in extracted_tokens if w not in stop_words}

            # Calculate Jaccard similarity
            intersection = len(original_tokens.intersection(extracted_tokens))
            union = len(original_tokens.union(extracted_tokens))

            if union == 0:
                return 0.0

            return intersection / union * 100
        except Exception as e:
            logger.error(f"Error calculating accuracy: {str(e)}")
            return 0.0

def main():
    # Initialize the extractor and summarizer
    extractor = TextExtractorAndSummarizer()
    
    # Get image path from user
    image_path = input("Enter the path to your image: ")
    
    if not os.path.exists(image_path):
        logger.error("Error: Image file not found!")
        return
    
    # Extract text from image
    logger.info("\nExtracting text from image...")
    extracted_text = extractor.extract_text_from_image(image_path)
    
    if extracted_text:
        logger.info("\nExtracted Text:")
        logger.info("-" * 50)
        logger.info(extracted_text)
        logger.info("-" * 50)
        
        # Generate summary
        logger.info("\nGenerating summary...")
        summary = extractor.generate_summary(extracted_text)
        
        if summary:
            logger.info("\nSummary:")
            logger.info("-" * 50)
            logger.info(summary)
            logger.info("-" * 50)

            # If you have the original text, you can calculate accuracy
            original_text = input("\nEnter the original text (or press Enter to skip accuracy calculation): ")
            if original_text:
                accuracy = extractor.calculate_accuracy(original_text, extracted_text)
                logger.info(f"\nAccuracy Score: {accuracy:.2f}%")
        else:
            logger.error("Failed to generate summary.")
    else:
        logger.error("Failed to extract text from the image.")

if __name__ == "__main__":
    main() 