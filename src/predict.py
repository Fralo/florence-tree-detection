import geopandas
import torch
from pathlib import Path
from deepforest import main
from deepforest.visualize import plot_results
from config import load_config
import numpy as np
from PIL import Image
import io

from wms_utils import extract_tree_coordinates_from_geotiff

# Load configuration
config = load_config()
model_config = config["model"]
pred_config = config["prediction"]


_model_cache = {}


def load_model(model_path: str | None = None) -> main.deepforest:
    """
    Load the fine-tuned DeepForest model from the specified path.
    If no path is provided, load the default pre-trained model.
    Args:
        model_path: Path to the fine-tuned model file. If None, load the default model.
    Returns:
        An instance of the DeepForest model.
    """
    
    cache_key = model_path or "default"
    if cache_key not in _model_cache:
        print(f"Loading model for: {cache_key}")
        model = main.deepforest()
        if model_path is None:
            model.load_model(model_name="weecology/deepforest-tree", revision="main")
        else:
            print(f"Loading model from: {model_path}")
            model.model = torch.load(model_path, weights_only=False)
        _model_cache[cache_key] = model
        
    return _model_cache[cache_key]


def predict(image: np.ndarray, model_path: str | None = None, score_thresh: float = 0.3) -> geopandas.GeoDataFrame | None:
    """Load the fine-tuned model and predict on a single image."""
    if image is None:
        raise ValueError("None is not allowed for argument `image`")

    
    model = load_model(model_path)
    if model.model is not None:
        model.model.score_thresh = score_thresh
    
    img_prediction = model.predict_image(image)
    
    return img_prediction


if __name__ == "__main__":
    import argparse
    import requests

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

            # Example to extract trees coordinates from TIF image
            # 
            # tree_coordinates = extract_tree_coordinates_from_geotiff(
            #     str(args.image_path),
            #     results_gdf,
            # )

            # print("\nTree coordinates (WGS 84 - EPSG:4326):")
            # for i, (lon, lat) in enumerate(tree_coordinates, 1):
            #     print(f"  Tree {i}: ({lat:.10f},{lon:.10f})")
