import logging
import os
import pandas as pd
import numpy as np
from datetime import datetime
from flask import Flask, request, jsonify, send_file, render_template, url_for
from werkzeug.utils import secure_filename
from flask_cors import CORS
from flasgger import Swagger
import pandasql as psql
import matplotlib.pyplot as plt
import io
import base64
from transformers import pipeline
import yake
from textblob import TextBlob
from PIL import Image
import cv2
from matplotlib.figure import Figure
from flask_sqlalchemy import SQLAlchemy
# Initialize the Flask application
app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///default.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
from flask_pymongo import PyMongo # type: ignore

db = SQLAlchemy(app)
app.config["MONGO_URI"] = os.getenv('MONGO_URI', "mongodb://mongo:27017/myDatabase")
mongo = PyMongo(app)
# Define a model
class ExampleModel(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80), nullable=False)

# Create tables
with app.app_context():
    db.create_all()

CORS(app)
Swagger(app)

# Configuration for text and tabular data
ALLOWED_EXTENSIONS = {'csv', 'xls', 'xlsx'}
UPLOAD_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), 'uploads'))
MAX_CONTENT_LENGTH = 500 * 1000 * 1000  # 500 MB limit

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH
app.config['CORS_HEADERS'] = 'Content-Type'
app.config['RESULT_FOLDER'] = 'results'

# Ensure the upload directory exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(app.config['RESULT_FOLDER']):
    os.makedirs(app.config['RESULT_FOLDER'])

data_store = {}

# Initialize the text summarization pipeline
summarizer = pipeline("summarization")

# Enable logging
logging.basicConfig(level=logging.DEBUG)

# Helper functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def create_timestamped_filename(filename):
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    basename, extension = os.path.splitext(filename)
    return f"{basename}_{timestamp}{extension}"

def preprocess_data(df):
    df = df.apply(pd.to_numeric, errors='coerce')  
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    return df

def calculate_statistics(df):
    numeric_df = df.select_dtypes(include=[np.number])
    categorical_df = df.select_dtypes(include=['object', 'category'])

    stats = {
        'numeric': {},
        'categorical': {}
    }

    if not numeric_df.empty:
        stats['numeric']['mean'] = numeric_df.mean().to_dict()
        stats['numeric']['std_dev'] = numeric_df.std().to_dict()
        stats['numeric']['variance'] = numeric_df.var().to_dict()
        stats['numeric']['skewness'] = numeric_df.skew().to_dict()
        stats['numeric']['median'] = numeric_df.median().to_dict()
        mode_df = numeric_df.mode()
        if not mode_df.empty:
            stats['numeric']['mode'] = mode_df.iloc[0].to_dict()
        stats['numeric']['quartiles'] = numeric_df.quantile([0.25, 0.5, 0.75]).to_dict()
        stats['numeric']['outliers'] = calculate_outliers(numeric_df)

    for column in categorical_df.columns:
        stats['categorical'][column] = {
            'mode': categorical_df[column].mode().iloc[0] if not categorical_df[column].mode().empty else None,
            'unique_values': categorical_df[column].nunique()
        }

    return stats

def calculate_outliers(df):
    outlier_results = {}
    for column in df.columns:
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1
        filter = (df[column] < (q1 - 1.5 * iqr)) | (df[column] > (q3 + 1.5 * iqr))
        outlier_results[column] = df[column][filter].tolist()
    return outlier_results

def convert_nan_to_none(obj):
    if isinstance(obj, dict):
        return {k: convert_nan_to_none(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_nan_to_none(i) for i in obj]
    elif isinstance(obj, float) and (pd.isna(obj) or obj != obj):  # Check for NaN
        return None
    else:
        return obj

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def file_upload():
    if 'files' not in request.files:
        return jsonify({'message': 'No files part in the request'}), 400
    files = request.files.getlist('files')
    if not files:
        return jsonify({'message': 'No files uploaded'}), 400

    filenames = []
    for f in files:
        if not allowed_file(f.filename):
            return jsonify({'message': 'File type not allowed'}), 400
        original_filename = secure_filename(f.filename)
        new_filename = create_timestamped_filename(original_filename)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], new_filename)
        f.save(save_path)
        df = pd.read_csv(save_path) if new_filename.endswith('.csv') else pd.read_excel(save_path)
        
        # Store both original and cleaned dataframes
        data_store[new_filename] = {
            "original": df,
            "cleaned": preprocess_data(df.copy())
        }
        filenames.append(new_filename)
    
    return jsonify({"filenames": filenames, "status": "Files successfully uploaded"}), 200

@app.route('/query/<filename>', methods=['POST'])
def query_data(filename):
    if filename not in data_store:
        logging.debug(f"Filename {filename} not found in data_store")
        return jsonify({'error': 'File not found'}), 404

    params = request.json
    if not params or 'query' not in params:
        logging.debug("No query provided in request")
        return jsonify({'error': 'No query provided'}), 400

    try:
        df = data_store[filename]["original"]
        query = params['query']
        logging.debug(f"Executing query: {query}")

        result_df = psql.sqldf(query, locals())
        logging.debug(f"Query result: {result_df}")

        result_json = result_df.to_dict(orient='records')
        return jsonify(result_json), 200
    except Exception as e:
        logging.exception("Error executing query")
        return jsonify({'error': 'Error executing query: ' + str(e)}), 500

@app.route('/process/<filename>', methods=['GET'])
def process_data(filename):
    if filename not in data_store:
        return jsonify({'error': 'File not found'}), 404

    try:
        df = data_store[filename]["cleaned"]
        stats_type = request.args.get('stats', 'all')
        stats = calculate_statistics(df)

        stats = convert_nan_to_none(stats)

        return jsonify({'filename': filename, 'statistics': stats}), 200
    except Exception as e:
        return jsonify({'error': 'Error processing file: ' + str(e)}), 500

@app.route('/data/<filename>', methods=['GET', 'POST', 'PUT', 'DELETE'])
def manage_data(filename):
    if filename not in data_store:
        return jsonify({'error': 'File not found'}), 404

    df = data_store[filename]["original"]

    if request.method == 'GET':
        return jsonify(convert_nan_to_none(df.to_dict(orient='records'))), 200

    elif request.method == 'POST':
        new_data = request.json
        try:
            new_df = pd.DataFrame([new_data])
            df = pd.concat([df, new_df], ignore_index=True)
            data_store[filename]["original"] = df
            return jsonify(convert_nan_to_none(df.to_dict(orient='records'))), 200
        except Exception as e:
            return jsonify({'error': 'Error adding data: ' + str(e)}), 500

    elif request.method == 'PUT':
        update_data = request.json
        try:
            for index, row in update_data.items():
                df.loc[int(index)] = row
            data_store[filename]["original"] = df
            return jsonify(convert_nan_to_none(df.to_dict(orient='records'))), 200
        except Exception as e:
            return jsonify({'error': 'Error updating data: ' + str(e)}), 500

    elif request.method == 'DELETE':
        delete_indices = request.json.get('indices')
        try:
            df.drop(index=delete_indices, inplace=True)
            data_store[filename]["original"] = df
            return jsonify(convert_nan_to_none(df.to_dict(orient='records'))), 200
        except Exception as e:
            return jsonify({'error': 'Error deleting data: ' + str(e)}), 500

@app.route('/visualize/<filename>', methods=['GET'])
def visualize_data(filename):
    if filename not in data_store:
        return jsonify({'error': 'File not found'}), 404

    try:
        df = data_store[filename]["original"]
        plot_type = request.args.get('plot_type', 'histogram')
        column = request.args.get('column')

        if column not in df.columns:
            return jsonify({'error': 'Column not found'}), 404

        plt.figure(figsize=(10, 6))

        if plot_type == 'histogram':
            df[column].hist()
            plt.title(f'Histogram of {column}')
        elif plot_type == 'bar':
            df[column].value_counts().plot(kind='bar')
            plt.title(f'Bar Chart of {column}')
        elif plot_type == 'box':
            df[column].plot(kind='box')
            plt.title(f'Box Plot of {column}')
        else:
            return jsonify({'error': 'Invalid plot type'}), 400

        plt.tight_layout()

        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()

        return jsonify({'plot_url': f'data:image/png;base64,{plot_url}'}), 200
    except Exception as e:
        logging.exception("Error visualizing data")
        return jsonify({'error': 'Error visualizing data: ' + str(e)}), 500

@app.route('/summarize', methods=['POST'])
def summarize():
    text = request.form['text']
    summary = summarizer(text, max_length=50, min_length=25, do_sample=False)
    return render_template('index.html', summary=summary[0]['summary_text'], input_text_summary=text)

@app.route('/keywords', methods=['POST'])
def extract_keywords():
    text = request.form['text']
    kw_extractor = yake.KeywordExtractor()
    keywords = kw_extractor.extract_keywords(text)
    keywords_list = [kw[0] for kw in keywords]
    return render_template('index.html', keywords=keywords_list, input_text_keywords=text)

@app.route('/search', methods=['POST'])
def search_text():
    text = request.form.get('text')
    search_term = request.form.get('search_term')
    if text and search_term:
        if search_term in text:
            result = "Found"
        else:
            result = "Not Found"
        return jsonify({'result': result, 'input_text_search': text, 'search_term': search_term})
    return jsonify({'error': 'Both text and search_term are required'}), 400

@app.route('/custom_query', methods=['POST'])
def custom_query():
    text = request.form.get('text')
    query = request.form.get('query')
    if text and query:
        # For simplicity, let's just count the number of occurrences of the query in the text
        occurrences = text.count(query)
        return jsonify({'occurrences': occurrences, 'input_text_query': text, 'query': query})
    return jsonify({'error': 'Both text and query are required'}), 400

@app.route('/sentiment', methods=['POST'])
def sentiment_analysis():
    text = request.form['text']
    blob = TextBlob(text)
    sentiment_score = blob.sentiment.polarity
    if sentiment_score > 0.1:
        sentiment = "Positive"
    elif sentiment_score < -0.1:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
    return render_template('index.html', sentiment=f"{sentiment} ({sentiment_score})", input_text_sentiment=text)

# Image processing routes
@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'images' not in request.files:
        return jsonify({"error": "No images provided"}), 400

    files = request.files.getlist('images')
    file_info = []

    for file in files:
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            with Image.open(file_path) as img:
                width, height = img.size
            file_info.append({
                "filename": filename,
                "width": width,
                "height": height,
                "path": url_for('uploaded_file', filename=filename),
                "mode": img.mode,
                "format": img.format
            })

    return jsonify({"message": "Images successfully uploaded", "files": file_info}), 200

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))

def generate_histogram(image_path):
    image = cv2.imread(image_path)
    color = ('b', 'g', 'r')
    figure = Figure()
    axis = figure.add_subplot(1, 1, 1)

    for i, col in enumerate(color):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        axis.plot(hist, color=col)

    output_path = os.path.join(app.config['RESULT_FOLDER'], os.path.basename(image_path) + '_histogram.png')
    figure.savefig(output_path)

    return output_path

@app.route('/histogram/<filename>', methods=['GET'])
def get_histogram(filename):
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    histogram_path = generate_histogram(image_path)
    return send_file(histogram_path, mimetype='image/png')

def generate_segmentation_mask(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    output_path = os.path.join(app.config['RESULT_FOLDER'], os.path.basename(image_path) + '_segmentation.png')
    cv2.imwrite(output_path, mask)
    
    return output_path

@app.route('/segmentation/<filename>', methods=['GET'])
def get_segmentation_mask(filename):
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    mask_path = generate_segmentation_mask(image_path)
    return send_file(mask_path, mimetype='image/png')

@app.route('/resize/<filename>', methods=['POST'])
def resize_image(filename):
    width = request.json.get('width')
    height = request.json.get('height')
    image_path = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(image_path):
        return jsonify({"error": "File not found"}), 404

    try:
        image = Image.open(image_path)
        resized_image = image.resize((width, height))
        img_io = io.BytesIO()
        resized_image.save(img_io, 'PNG')
        img_io.seek(0)
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        return jsonify({"error": str(e)}), 500
@app.route('/crop/<filename>', methods=['POST'])
def crop_image(filename):
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image = Image.open(image_path)
    left = request.json.get('left')
    top = request.json.get('top')
    right = request.json.get('right')
    bottom = request.json.get('bottom')

    if left < 0 or top < 0 or right > image.width or bottom > image.height or left >= right or top >= bottom:
        logging.error("Invalid crop coordinates")
        return jsonify({"error": "Invalid crop coordinates"}), 400

    cropped_image = image.crop((left, top, right, bottom))
    
    output = io.BytesIO()
    cropped_image.save(output, format='PNG')
    output.seek(0)
    
    return send_file(output, mimetype='image/png')

@app.route('/convert/<filename>', methods=['POST'])
def convert_image(filename):
    logging.debug(f"Received request to convert {filename}")
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    logging.debug(f"Image path: {image_path}")
    
    image = Image.open(image_path)
    format = request.json.get('format').lower()
    logging.debug(f"Requested format: {format}")

    valid_formats = ["jpeg", "png", "bmp", "gif", "tiff"]
    if format not in valid_formats:
        logging.error(f"Unsupported format: {format}")
        return jsonify({"error": f"Unsupported format: {format}"}), 400

    try:
        output_path = os.path.join(app.config['RESULT_FOLDER'], os.path.splitext(os.path.basename(image_path))[0] + f'.{format}')
        image.save(output_path, format=format.upper())
        return send_file(output_path, mimetype=f'image/{format}')
    except KeyError as e:
        logging.error(f"Error saving image in format {format}: {e}")
        return jsonify({"error": f"Error saving image in format: {format}"}), 500

@app.errorhandler(413)
def too_large(e):
    return jsonify({'message': 'The file is too large'}), 413

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'message': 'Internal server error'}), 500

if __name__ == "__main__":
    app.run(port=5003, debug=True)
