# DeepForest Tree Detection Project

A complete pipeline for tree detection using [DeepForest](https://deepforest.readthedocs.io/), including data preparation, model training, evaluation, prediction, and a web application for visualizing detected trees on a map.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [1. ML Pipeline: Training, Prediction & Tile Download](#1-ml-pipeline-training-prediction--tile-download)
  - [Downloading Tiles from WMS](#downloading-tiles-from-wms)
  - [Preparing Data](#preparing-data)
  - [Training the Model](#training-the-model)
  - [Predicting Trees](#predicting-trees)
  - [Evaluating the Model](#evaluating-the-model)
  - [Scraping Trees at Scale](#scraping-trees-at-scale)
- [2. DevOps: Running the Web Application](#2-devops-running-the-web-application)
  - [Local Development](#local-development)
  - [Production Deployment](#production-deployment)
- [Configuration](#configuration)

---

## Overview

This project provides:

1. **ML Pipeline**: Scripts for downloading satellite imagery tiles, preparing training data, fine-tuning the DeepForest model, and running predictions.
2. **Web Application**: A FastAPI backend and Vue.js frontend for storing and visualizing detected trees on an interactive map.

---

## Project Structure

```
.
├── src/                          # ML pipeline scripts
│   ├── config.py                 # Configuration loader
│   ├── download_florence_tiles.py # Download tiles from WMS
│   ├── prepare_data.py           # Split data into train/val/test
│   ├── convert_labels_to_deepforest_format.py  # Convert XML annotations to CSV
│   ├── train_model.py            # Fine-tune DeepForest model
│   ├── predict.py                # Run predictions on images
│   ├── evaluate_model.py         # Evaluate model performance
│   ├── scrape_trees.py           # Batch process tiles and post to API
│   ├── wms_utils.py              # WMS utilities for tile fetching
│   └── generate_graphs.py        # Generate evaluation visualizations
├── backend/                      # FastAPI backend application
│   ├── app/
│   │   ├── main.py               # API endpoints
│   │   ├── models.py             # Database models
│   │   └── database.py           # Database configuration
│   ├── alembic/                  # Database migrations
│   └── requirements.txt          # Backend dependencies
├── frontend/                     # Vue.js frontend application
│   └── src/
│       ├── components/Map.vue    # Leaflet map component
│       └── views/MapView.vue     # Map view
├── devops/                       # Docker deployment configurations
│   ├── local/                    # Local development setup
│   └── production/               # Production deployment setup
├── config.yml                    # Main configuration file
└── requirements.txt              # ML pipeline dependencies
```

---

## Installation

### ML Pipeline Dependencies

```bash
# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 1. ML Pipeline: Training, Prediction & Tile Download

### Downloading Tiles from WMS

Download satellite imagery tiles from the Regione Toscana WMS service:

```bash
python src/download_florence_tiles.py
```

This script downloads GeoTIFF tiles covering a specified area. To customize the area, edit the start and end points in the script:

```python
start_point = Point(674048.64, 4852250.78)  # Bottom-left (EPSG:25832)
end_point = Point(675960.26, 4853751.03)    # Top-right (EPSG:25832)
```

Tiles are saved to `data/01_raw/florence/` by default.

**Custom Tile Download:**

You can also use the `wms_utils.py` module directly:

```python
from src.wms_utils import Point, Tile, download_tiles

# Download tiles for a specific area
start = Point(674000, 4852000)
end = Point(676000, 4854000)
download_tiles(start=start, end=end, step_in_m=80, max_workers=10)
```

---

### Preparing Data

#### Step 1: Organize Label Studio Export

Place your Label Studio export (with Pascal VOC XML annotations) in:
- Images: `data/02_processed/label_studio_export/images/`
- Annotations: `data/02_processed/label_studio_export/Annotations/`

#### Step 2: Split Data into Train/Validation/Test Sets

```bash
python src/prepare_data.py
```

This splits annotations into:
- `data/02_processed/train/` (70%)
- `data/02_processed/evaluate/` (15%)
- `data/02_processed/test/` (15%)

#### Step 3: Convert Annotations to DeepForest Format

```bash
python src/convert_labels_to_deepforest_format.py
```

This converts Pascal VOC XML files to DeepForest CSV format with columns:
`image_path, xmin, ymin, xmax, ymax, label`

---

### Training the Model

Fine-tune the DeepForest model on your custom dataset:

```bash
python src/train_model.py [OPTIONS]
```

**Options:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--use-pretrained` | False | Use official DeepForest pretrained weights |
| `--base-model-path` | None | Path to a base model to continue training from |
| `--batch-size` | 1 | Training batch size |
| `--epochs` | 20 | Number of training epochs |
| `--learning-rate` | 0.0001 | Learning rate |
| `--score-thresh` | 0.4 | Confidence score threshold |
| `--nms-thresh` | 0.15 | Non-maximum suppression threshold |
| `--num-workers` | 1 | Number of data loading workers |
| `--output-dir` | models | Directory to save trained models |
| `--fast-dev-run` | False | Quick test run for debugging |

**Examples:**

```bash
# Train from pretrained DeepForest weights
python src/train_model.py --use-pretrained --epochs 30 --batch-size 4

# Continue training from an existing model
python src/train_model.py --base-model-path models/deepforest_finetuned_2.pt --epochs 10

# Quick debug run
python src/train_model.py --use-pretrained --fast-dev-run
```

Trained models are saved to `models/` as `deepforest_finetuned_N.pt`.

---

### Predicting Trees

Run tree detection on images:

```bash
python src/predict.py [OPTIONS]
```

**Options:**

| Argument | Description |
|----------|-------------|
| `--image_path` | Path to a local image file |
| `--image_url` | URL of an image to download and process |
| `--model_path` | Path to a fine-tuned model (optional, uses pretrained if not specified) |

**Examples:**

```bash
# Predict using the pretrained model
python src/predict.py --image_path data/test_image.png

# Predict using a fine-tuned model
python src/predict.py --image_path data/test_image.tif --model_path models/deepforest_finetuned_3.pt

# Predict from a URL
python src/predict.py --image_url "https://example.com/satellite_image.png"
```

**Programmatic Usage:**

```python
from src.predict import predict, load_model
from PIL import Image
import numpy as np

# Load image
image = np.array(Image.open("image.png").convert("RGB")).astype("float32")

# Run prediction
results_gdf = predict(image, model_path="models/deepforest_finetuned_3.pt", score_thresh=0.3)

# Results is a GeoDataFrame with columns: xmin, ymin, xmax, ymax, label, score
print(results_gdf)
```

---

### Evaluating the Model

Evaluate model performance on the test set:

```bash
python src/evaluate_model.py [OPTIONS]
```

**Options:**

| Argument | Description |
|----------|-------------|
| `--model_path` | Path to the model to evaluate (optional, uses pretrained if not specified) |

**Examples:**

```bash
# Evaluate pretrained model
python src/evaluate_model.py

# Evaluate fine-tuned model
python src/evaluate_model.py --model_path models/deepforest_finetuned_3.pt
```

Results include:
- **Average IoU** (Intersection over Union)
- **Box Recall** 
- **Box Precision**

Results are saved to `data/03_results/` as JSON files.

---

### Scraping Trees at Scale

Process tiles over a large area and post detected trees to the API:

```bash
python src/scrape_trees.py
```

This script:
1. Generates tiles covering the specified area
2. Downloads each tile from the WMS service
3. Runs tree detection on each tile
4. Extracts geographic coordinates (WGS 84)
5. Posts detected trees to the backend API

Edit the script to customize the area:

```python
start_point = Point(678481.65, 4844195.91)
end_point = Point(684964.18, 4851843.27)
```

**Note:** The API must be running locally at `http://localhost:8000` for this to work.

---

## 2. DevOps: Running the Web Application

The web application consists of:
- **PostgreSQL**: Database for storing detected trees
- **FastAPI Backend**: REST API for tree data
- **Vue.js Frontend**: Interactive map for visualization
- **Caddy** (production only): Reverse proxy with automatic HTTPS

### Local Development

Navigate to the local devops directory and start the services:

```bash
cd devops/local

# Start all services
docker-compose up -d

# Or start with build
docker-compose up -d --build
```

**Services:**

| Service | URL | Description |
|---------|-----|-------------|
| Backend API | http://localhost:8000 | FastAPI REST API |
| API Docs | http://localhost:8000/docs | Swagger UI documentation |
| Frontend | http://localhost:51730 | Vue.js development server |
| PostgreSQL | localhost:5432 | Database (user: postgres, password: postgres) |

**Development Features:**
- Hot-reload enabled for both backend and frontend
- Source code is mounted as volumes for live editing

**Useful Commands:**

```bash
# View logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f app
docker-compose logs -f frontend

# Stop services
docker-compose down

# Stop and remove data
docker-compose down -v

# Rebuild after dependency changes
docker-compose up -d --build
```

---

### Production Deployment

For production deployment on a VPS with automatic HTTPS.

#### Prerequisites

- Docker and Docker Compose installed
- Domain name pointing to your server
- Ports 80 and 443 open

#### Deployment Steps

1. **Clone and Navigate:**
   ```bash
   git clone <your-repo-url>
   cd test-deepforest/devops/production
   ```

2. **Configure Environment:**
   ```bash
   cp .env.example .env
   nano .env
   ```
   
   Update these values in `.env`:
   ```env
   POSTGRES_USER=your_postgres_user
   POSTGRES_PASSWORD=your_secure_password
   POSTGRES_DB=trees_db
   DATABASE_URL=postgresql://your_postgres_user:your_secure_password@db:5432/trees_db
   DOMAIN=your-domain.com
   CADDY_EMAIL=your-email@example.com
   ```

3. **Configure Caddyfile:**
   ```bash
   cp Caddyfile.example Caddyfile
   nano Caddyfile
   ```
   Update the email and domain in `Caddyfile`.

4. **Build Frontend:**
   ```bash
   ./build-frontend.sh
   ```

5. **Start Services:**
   ```bash
   docker-compose up -d
   ```

6. **Verify Deployment:**
   ```bash
   docker-compose ps
   docker-compose logs -f
   ```

**Production URLs:**

| Endpoint | URL |
|----------|-----|
| Frontend | https://your-domain.com |
| API | https://your-domain.com/api/ |
| API Docs | https://your-domain.com/docs |

**Updating the Application:**

```bash
# Update backend
git pull
docker-compose build backend
docker-compose up -d backend

# Update frontend
git pull
./build-frontend.sh
docker-compose restart caddy
```

**Database Backup:**

```bash
# Backup
docker-compose exec db pg_dump -U $POSTGRES_USER $POSTGRES_DB > backup.sql

# Restore
docker-compose exec -T db psql -U $POSTGRES_USER $POSTGRES_DB < backup.sql
```

---

## Configuration

The main configuration is in `config.yml`:

```yaml
data:
  raw_data_dir: "data/01_raw/label-studio-export/images/"
  processed_data_dir: "data/02_processed/label-studio-export/images"
  annotations_file: "data/02_processed/deepforest_annotations.csv"

model:
  pretrained_model_name: "weecology/deepforest-tree"
  pretrained_model_revision: "main"
  checkpoint_dir: "models/checkpoints"
  final_model_path: "models/deepforest_finetuned_3.pt"

training:
  accelerator: "mps"  # Options: "mps" (Apple Silicon), "gpu" (CUDA), "cpu"
  devices: 1
  max_epochs: 20
  training_data: data/02_processed/train/images
  training_annotations: data/02_processed/train/annotations.csv
  validation_data: data/02_processed/evaluate/images
  validation_annotations: data/02_processed/evaluate/annotations.csv
  score_thresh: 0.3

evaluation:
  model: "models/deepforest_finetuned_3.pt"
  data: data/02_processed/test/images
  annotations: data/02_processed/test/annotations.csv
  score_thresh: 0.3

prediction:
  score_thresh: 0.3
```

---

## API Reference

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| GET | `/trees` | Get detected trees (with optional bounding box filter) |
| POST | `/trees` | Create a new tree (development only) |

### GET /trees Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `min_lat` | float | Minimum latitude filter |
| `max_lat` | float | Maximum latitude filter |
| `min_lon` | float | Minimum longitude filter |
| `max_lon` | float | Maximum longitude filter |
| `limit` | int | Maximum number of results (default: 100) |

---

## License

MIT License
