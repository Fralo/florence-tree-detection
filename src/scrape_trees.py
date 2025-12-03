import io
import geopandas
import numpy as np
import requests
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from PIL import Image

from predict import load_model, predict

import rasterio
from rasterio.warp import transform as rio_transform


def extract_tree_coordinates_from_prediction(
    image_data: bytes, predictions: geopandas.GeoDataFrame
) -> List[Tuple[float, float]]:
    """
    Extract geographic coordinates for detected trees from predictions in WGS 84 (EPSG:4326).

    Args:
        image_data: In-memory GeoTIFF image data in bytes.
        predictions: GeoDataFrame containing bounding box predictions with columns:
                     xmin, ymin, xmax, ymax (in pixel coordinates)

    Returns:
        List of tuples (longitude, latitude) representing WGS 84 coordinates of tree centers
    """
    coordinates = []

    with rasterio.open(io.BytesIO(image_data)) as src:
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


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):
        return f"point_{self.x}_{self.y}"


class Tile:
    def __init__(self, point: Point, bbox_step: int = 80, prefix: str = "data/01_raw/florence"):
        self.point = point
        self.bbox_step = bbox_step
        self.prefix = prefix

    def __str__(self):
        return f"{self.point}"

    @property
    def bbox(self):
        return self.coordinates_to_bbox(self.point, self.bbox_step)

    # def download(self, output_file: str | None = None):
    #     if output_file is None:
    #         output_file = f"zz_23_{self.point.x}_{self.point.y}.tif"

    #     get_wms_geotiff(self.bbox, f"{self.prefix}/{output_file}")

    @classmethod
    def coordinates_to_bbox(cls, point: Point, step=80) -> list:
        # min_x = 680012.63 - 250  # 250m to the west
        # min_y = 4849412.92 - 250  # 250m to the south
        # max_x = 680012.63 + 250  # 250m to the east
        # max_y = 4849412.92 + 250  # 250m to the north

        step = step / 2
        return [
            point.x - step,
            point.y - step,
            point.x + step,
            point.y + step,
        ]


def get_wms_geotiff(
    bbox: List[float],
    layer: str = "rt_ofc.5k23.32bit", #for 2024/25 photos use -> "rt_ofc.5k24.32bit",
    width: int = 800,
    height: int = 800,
    crs: str = "EPSG:25832",
    base_url: str = "https://www502.regione.toscana.it/ows_ofc/com.rt.wms.RTmap/wms",
):
    """
    Downloads a GeoTIFF tile from a WMS service given a bounding box.

    Args:
        bbox (List[float]): The bounding box for the tile in the format
                            [minX, minY, maxX, maxY]. The coordinates must
                            be in the same CRS specified by the 'crs' param.
        output_filepath (str): The path and filename to save the downloaded
                               .tif file (e.g., 'my_tile.tif').
        layer (str, optional): The WMS layer to request.
                               Defaults to 'rt_ofc.5k24.32bit'.
        width (int, optional): The width of the output image in pixels.
                               Defaults to 800.
        height (int, optional): The height of the output image in pixels.
                                Defaults to 800.
        crs (str, optional): The Coordinate Reference System for the BBOX.
                             Defaults to 'EPSG:25832'.
        base_url (str, optional): The base URL of the WMS service.
                                  Defaults to the Regione Toscana service.
    """

    # Convert the bounding box list to the comma-separated string
    # required by the WMS BBOX parameter
    bbox_str = ",".join(map(str, bbox))

    # Define all the URL parameters for the GetMap request
    params = {
        "map": "owsofc_rt",  # Required MapServer map identifier
        "SERVICE": "WMS",
        "VERSION": "1.3.0",
        "REQUEST": "GetMap",
        "LAYERS": layer,
        "STYLES": "",
        "CRS": crs,
        "BBOX": bbox_str,
        "WIDTH": width,
        "HEIGHT": height,
        "FORMAT": "image/tiff",
        "EXCEPTIONS": "INIMAGE",  # How to report errors
        "TRANSPARENT": "true",
    }

    print(f"Requesting 800x800 GeoTIFF for BBOX: {bbox_str}...")

    try:
        # Make the HTTP GET request
        response = requests.get(base_url, params=params)

        # This will raise an exception if the server returns an HTTP error
        # (e.g., 404, 500)
        response.raise_for_status()

        # Check if the server returned a GeoTIFF or an error message
        # WMS errors are often returned as XML or text
        content_type = response.headers.get("Content-Type")

        if "image/tiff" in content_type:
            # Save image file in memory without writing to disk
            image_data = response.content
            
            with io.BytesIO(image_data) as img_buffer:
                img_file = Image.open(img_buffer)
                image = np.array(img_file.convert("RGB")).astype("float32")

                results_gdf = predict(image, model_path="models/deepforest_finetuned_4.pt")
                
                # filter out all row where score < 0.5
                results_gdf = results_gdf[results_gdf['score'] >= 0.5]
                
                if not results_gdf.empty:
                    tree_coords = extract_tree_coordinates_from_prediction(
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
                        

        else:
            # The server returned something other than a GeoTIFF
            # (likely an error message)
            print("Error: Server did not return a GeoTIFF.")
            print(f"Response Content-Type: {content_type}")
            print(f"Server Response (first 500 chars):\n{response.text[:500]}...")

    except requests.exceptions.HTTPError as e:
        # Handle HTTP errors (e.g., 404 Not Found, 500 Internal Server Error)
        print(f"HTTP Error: {e}")
        print(f"Response text: {response.text}")
    except requests.exceptions.RequestException as e:
        # Handle other network-related errors (e.g., connection error)
        print(f"An error occurred: {e}")


def process_tile(tile: Tile):
    get_wms_geotiff(tile.bbox)

def download_florence_tiles(start: Point, end: Point):
    """
    Params:
    start_point: tuple of (x, y) in EPSG:25832, bottom-left corner
    end_point: tuple of (x, y) in EPSG:25832, top-right corner
    """
    step_in_m = 80

    current = deepcopy(start)

    tiles_to_download: list[Tile] = []

    while current.y < end.y:
        while current.x < end.x:
            tiles_to_download.append(Tile(Point(current.x, current.y)))
            current.x = current.x + step_in_m
        current.y = current.y + step_in_m
        current.x = start.x

    with ThreadPoolExecutor(max_workers=10) as executor:
        executor.map(process_tile, tiles_to_download)

        



if __name__ == "__main__":
    
    model = load_model("models/deepforest_finetuned_4.pt") #for warming up the model before downloading tiles
    
    

    start_point = Point(678481.65,4844195.91)
    end_point = Point(684964.18, 4851843.27)

    download_florence_tiles(start=start_point, end=end_point)
