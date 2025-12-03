"""
Scrape trees from WMS tiles and send predictions to API.
"""

import io
import requests
import numpy as np
from PIL import Image

from predict import load_model, predict
from wms_utils import (
    Point,
    Tile,
    fetch_wms_geotiff,
    process_tiles,
    extract_tree_coordinates_from_geotiff,
)


def process_tile_and_post_trees(tile: Tile) -> None:
    """
    Fetch a tile, run tree detection, and post results to API.
    
    Args:
        tile: Tile object to process.
    """
    bbox_str = ",".join(map(str, tile.bbox))
    image_data = fetch_wms_geotiff(tile.bbox)
    
    if image_data is None:
        return
    
    with io.BytesIO(image_data) as img_buffer:
        img_file = Image.open(img_buffer)
        image = np.array(img_file.convert("RGB")).astype("float32")

        results_gdf = predict(image, model_path="models/deepforest_finetuned_4.pt")
        
        if results_gdf is None or results_gdf.empty:
            return
            
        # Filter out all rows where score < 0.5
        results_gdf = results_gdf[results_gdf['score'] >= 0.5]
        
        if results_gdf.empty:
            return
            
        tree_coords = extract_tree_coordinates_from_geotiff(
            image_data, results_gdf
        )
        
        for i, (lon, lat) in enumerate(tree_coords, 1):
            prediction_row = results_gdf.iloc[i - 1]
            tree_data = {
                "latitude": lat,
                "longitude": lon,
                "source_file": f"bbox_{bbox_str}.tif",
                "bbox_xmin": int(prediction_row["xmin"]),
                "bbox_ymin": int(prediction_row["ymin"]),
                "bbox_xmax": int(prediction_row["xmax"]),
                "bbox_ymax": int(prediction_row["ymax"]),
            }
            try:
                post_response = requests.post(
                    "http://localhost:8000/trees", json=tree_data
                )
                post_response.raise_for_status()
                print(
                    f"Successfully posted tree {i} to API: {post_response.json()}"
                )
            except requests.exceptions.RequestException as e:
                print(f"Error posting tree {i} to API: {e}")


if __name__ == "__main__":
    # Warm up the model before downloading tiles
    model = load_model("models/deepforest_finetuned_4.pt")

    start_point = Point(678481.65, 4844195.91)
    end_point = Point(684964.18, 4851843.27)

    process_tiles(
        start=start_point,
        end=end_point,
        process_fn=process_tile_and_post_trees,
    )
