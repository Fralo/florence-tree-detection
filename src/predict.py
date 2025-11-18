from functools import lru_cache
import geopandas

import torch
from pathlib import Path
from deepforest import main
from deepforest.visualize import plot_results
from config import load_config
import rasterio
from rasterio.warp import transform as rio_transform
from typing import List, Tuple
import numpy as np
from PIL import ImageFile, Image
import io

# Load configuration
config = load_config()
model_config = config["model"]
pred_config = config["prediction"]


def extract_tree_coordinates_from_prediction(
    image_path: Path, predictions: geopandas.GeoDataFrame
) -> List[Tuple[float, float]]:
    """
    Extract geographic coordinates for detected trees from predictions in WGS 84 (EPSG:4326).

    Args:
        image_path: Path to the GeoTIFF image file
        predictions: GeoDataFrame containing bounding box predictions with columns:
                     xmin, ymin, xmax, ymax (in pixel coordinates)

    Returns:
        List of tuples (longitude, latitude) representing WGS 84 coordinates of tree centers
    """
    coordinates = []

    with rasterio.open(image_path) as src:
        # Get the affine transform (converts pixel coordinates to geographic coordinates)
        transform = src.transform
        source_crs = src.crs

        print(f"Debug - Transform: {transform}")
        print(f"Debug - Source CRS: {source_crs}")
        print(f"Debug - Bounds: {src.bounds}")

        # Process each prediction
        for idx, pred in predictions.iterrows():
            # Calculate center point of bounding box in pixel coordinates
            center_col = pred["xmin"] + (pred["xmax"] - pred["xmin"]) / 2
            center_row = pred["ymin"] + (pred["ymax"] - pred["ymin"]) / 2

            # Convert pixel coordinates to source CRS geographic coordinates
            # The * operator applies the affine transform: (geo_x, geo_y) = transform * (col, row)
            geo_x, geo_y = transform * (center_col, center_row)

            # Transform from source CRS to WGS 84 (EPSG:4326)
            lon, lat = rio_transform(source_crs, "EPSG:4326", [geo_x], [geo_y])

            if idx == 0:  # Debug first prediction
                print(
                    f"Debug - First prediction pixel coords: col={center_col}, row={center_row}"
                )
                print(
                    f"Debug - First prediction source CRS coords: x={geo_x}, y={geo_y}"
                )
                print(
                    f"Debug - First prediction WGS 84 coords: lon={lon[0]}, lat={lat[0]}"
                )

            coordinates.append((lon[0], lat[0]))

    return coordinates

@lru_cache()
def load_model(model_path: str | None) -> main.deepforest:
    """
    Load the fine-tuned DeepForest model from the specified path.
    If no path is provided, load the default pre-trained model.
    Args:
        model_path: Path to the fine-tuned model file. If None, load the default model.
    Returns:
        An instance of the DeepForest model.
    """
    model = main.deepforest()

    if model_path is None:
        model.load_model(model_name="weecology/deepforest-tree", revision="main")

    model.model = torch.load(model_path, weights_only=False)
    
    return model

def predict(image: np.ndarray, model_path: str | None = None, score_thresh: float = 0.3) -> geopandas.GeoDataFrame | None:
    """Load the fine-tuned model and predict on a single image."""
    if image is None:
        raise ValueError("None is not allowed for argument `image`")

    
    model = load_model(model_path), 
    model.model.score_thresh = score_thresh
    
    img_prediction = model.predict_image(image)
    
    return img_prediction


if __name__ == "__main__":
    import argparse
    import requests
    import numpy as np

    parser = argparse.ArgumentParser(
        description="Predict trees in an image using a fine-tuned model."
    )
    parser.add_argument(
        "--image_path",
        required=False,
        type=Path,
        help="Path to the image for prediction.",
    )
    parser.add_argument(
        "--model_path",
        required=False,
        type=str,
        help="Path to the model.pt file"
    )
    parser.add_argument("--image_url", required=False, type=str)

    args = parser.parse_args()
    if args.image_path:
        img_file = Image.open(args.image_path)
    elif args.image_url:
        response = requests.get(args.image_url, timeout=30)
        response.raise_for_status()
        img_file = Image.open(io.BytesIO(response.content))
    else:
        raise ValueError("Either --image_path or --image_url must be provided.")

    image = np.array(img_file.convert("RGB")).astype("float32")

    results_gdf = predict(image, args.model_path)

    if results_gdf is None:
        
        print("No tees found.")
    elif not results_gdf.empty:
        # print only the columns name of the  dataframe
        print(results_gdf)
        
        if args.image_path:
            results_gdf["image_path"] = args.image_path.name
            results_gdf.root_dir = str(args.image_path.parent)
            plot_results(results_gdf)

            # Example to extract trees coordintes from TIF image
            # 
            # tree_coordinates = extract_tree_coordinates_from_prediction(
            #     args.image_path,
            #     results_gdf,
            # )

            # print("\nTree coordinates (WGS 84 - EPSG:4326):")
            # for i, (lon, lat) in enumerate(tree_coordinates, 1):
            #     print(f"  Tree {i}: ({lat:.10f},{lon:.10f})")
