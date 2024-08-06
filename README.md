# Corporatica Backend Task

## Project Overview

This project is an advanced software application focused on containerization and deployment technologies. It provides in-depth analysis and manipulation of various data types, with a robust back-end developed in Python using Flask. The application processes tabular data, RGB images, and textual data.

## Features

- **Tabular Data**: Upload, process, and perform complex queries on tabular data. Compute advanced statistics and generate dynamic visualizations.
- **RGB Images**: Upload and store images, generate color histograms and segmentation masks, and perform image manipulations (resizing, cropping, format conversion).
- **Textual Data**: Perform text summarization, keyword extraction, sentiment analysis, and other text processing tasks.

## Technologies Used

- **Back-End**: Python 3.11, Flask, SQLAlchemy, PyMongo, pandas, numpy, pandasql, Flask-CORS, Flask-Swagger, textblob, yake, PIL, cv2
- **Database**: PostgreSQL for tabular data, MongoDB for document storage
- **Containerization and Deployment**: Docker, Docker Compose

## Installation Instructions

### Prerequisites

- Docker
- Docker Compose

### Setup

1. **Clone the repository**:
    ```sh
    git clone https://github.com/Samspei01/Corporatica-Backend-Task)
    cd Corporatica-Backend-Task
    ```

2. **Build and start the Docker containers**:
    ```sh
    docker-compose build
    docker-compose up -d
    ```

3. **Access the application**:
    - Flask application: [http://localhost:5003](http://localhost:5003)
    - Adminer (PostgreSQL management): [http://localhost:8080](http://localhost:8080)


### API Endpoints

- **Tabular Data**
    - `POST /upload`: Upload tabular data files (CSV, XLS, XLSX).
    - `POST /query/<filename>`: Execute SQL queries on uploaded data.
    - `GET /process/<filename>`: Compute statistics on uploaded data.

- **RGB Images**
    - `POST /upload_image`: Upload image files.
    - `GET /histogram/<filename>`: Get color histogram of an image.
    - `GET /segmentation/<filename>`: Get segmentation mask of an image.
    - `POST /resize/<filename>`: Resize an image.
    - `POST /crop/<filename>`: Crop an image.
    - `POST /convert/<filename>`: Convert image format.

- **Textual Data**
    - `POST /summarize`: Summarize text.
    - `POST /keywords`: Extract keywords from text.
    - `POST /sentiment`: Perform sentiment analysis on text.

### Running Tests

To verify the database connection, use the provided `test_db.py` script:
```sh
docker exec -it flask_app /bin/bash
python3.11 test_db.py
```

## Development Methodology

### Technology Selection

- **Backend**: Python, Flask for API development.
- **Database**: PostgreSQL for tabular data, MongoDB for document storage.
- **Containerization**: Docker for containerizing the application, Docker Compose for managing multi-container setups.

### Time Allocation

- **Research**: 8 hours
- **Development**: 24 hours
- **Testing**: 8 hours
- **Deployment**: 4 hours

## Requirements

All the Python dependencies are listed in the `requirements.txt` file. The project uses Python 3.11.

To install the dependencies locally, run:
```sh
pip install -r requirements.txt
```

## Docker

The project includes a `Dockerfile` for building the application container. The Dockerfile is set up to use Python 3.11 and includes all necessary dependencies.

### Dockerfile
```dockerfile
# Use the official Python 3.11 image
FROM python:3.11.0-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libopencv-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip to ensure we are using the latest version for Python 3.11
RUN python3.11 -m pip install --upgrade pip

# Install Python dependencies
COPY requirements.txt ./
RUN python3.11 -m pip install --no-cache-dir -r requirements.txt

# Check pip version
RUN python3.11 -m pip --version

# Make port 5003 available to the world outside this container
EXPOSE 5003

# Define environment variables
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0

# Run the command to start the application
CMD ["python3.11", "-m", "flask", "run", "--port=5003"]

```

## Contribution Guidelines

Contributions are welcome! Please fork the repository and create a pull request with your changes.

## License

This project is licensed under the MIT License.

## Scripts Documentation
app.py
The app.py script is the main application file for the Corporatica Backend project. It uses Flask to create a web application that can handle various data types (tabular data, RGB images, and text). Here's a breakdown of its key components:

Initialization: The script initializes the Flask application and sets up the database connections using SQLAlchemy for PostgreSQL and PyMongo for MongoDB.
Configuration: It configures the application settings, including upload folder paths and maximum content length for file uploads.
Models: Defines database models, such as ExampleModel, which is used to create tables in the database.
Helper Functions: Contains helper functions for tasks like checking allowed file types, creating timestamped filenames, preprocessing data, and calculating statistics and outliers.
Endpoints:
/upload: Endpoint for uploading tabular data files (CSV, XLS, XLSX).
/query/<filename>: Endpoint for executing SQL queries on uploaded data.
/process/<filename>: Endpoint for computing statistics on uploaded data.
/data/<filename>: Endpoint for managing data (CRUD operations).
/visualize/<filename>: Endpoint for visualizing data with different types of plots.
/summarize: Endpoint for summarizing text.
/keywords: Endpoint for extracting keywords from text.
/sentiment: Endpoint for performing sentiment analysis on text.
/upload_image: Endpoint for uploading image files.
/histogram/<filename>: Endpoint for generating color histograms of images.
/segmentation/<filename>: Endpoint for generating segmentation masks of images.
/resize/<filename>: Endpoint for resizing images.
/crop/<filename>: Endpoint for cropping images.
/convert/<filename>: Endpoint for converting image formats.
These endpoints allow users to interact with the application, upload and process data, and perform various analytical and visualization tasks. The application uses several libraries and tools, including pandas, numpy, matplotlib, and text processing libraries like textblob and yake.

Example for `app.py`:

```python
app.py

This is the main application script for the Advanced Data Processing and Analysis Application.
It initializes the Flask application, sets up database connections, and defines various API endpoints for handling tabular data, RGB images, and textual data.

Usage:
    - Run the script: `python3.11 app.py`
    - Access the application at http://localhost:5003
```

## Contact
For any questions or support, please contact me at [MY-EMAL](abdosaaed749@gmail.com).
