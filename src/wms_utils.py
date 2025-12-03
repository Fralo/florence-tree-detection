"""
Shared utilities for WMS tile operations.
Contains Point, Tile classes and functions for downloading GeoTIFF tiles from WMS services.
"""

import io
import requests
from typing import List, Tuple, Callable, Optional
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy

import geopandas
import numpy as np
import rasterio
from rasterio.warp import transform as rio_transform


class Point:
    """Represents a point in a coordinate system."""
    
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def __str__(self) -> str:
        return f"point_{self.x}_{self.y}"


class Tile:
    """Represents a tile with a center point and bounding box."""
    
    def __init__(self, point: Point, bbox_step: int = 80, prefix: str = "data/01_raw/florence"):
        self.point = point
        self.bbox_step = bbox_step
        self.prefix = prefix

    def __str__(self) -> str:
        return f"{self.point}"

    @property
    def bbox(self) -> List[float]:
        return self.coordinates_to_bbox(self.point, self.bbox_step)

    def download(self, output_file: str | None = None) -> None:
        """Download the tile to a file."""
        if output_file is None:
            output_file = f"zz_23_{self.point.x}_{self.point.y}.tif"

        download_wms_geotiff_to_file(self.bbox, f"{self.prefix}/{output_file}")

    @classmethod
    def coordinates_to_bbox(cls, point: Point, step: int = 80) -> List[float]:
        """Convert a center point and step size to a bounding box."""
        half_step = step / 2
        return [
            point.x - half_step,
            point.y - half_step,
            point.x + half_step,
            point.y + half_step,
        ]


def _build_wms_params(
    bbox: List[float],
    layer: str = "rt_ofc.5k23.32bit",
    width: int = 800,
    height: int = 800,
    crs: str = "EPSG:25832",
) -> dict:
    """Build WMS request parameters."""
    bbox_str = ",".join(map(str, bbox))
    return {
        "map": "owsofc_rt",
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
        "EXCEPTIONS": "INIMAGE",
        "TRANSPARENT": "true",
    }


def fetch_wms_geotiff(
    bbox: List[float],
    layer: str = "rt_ofc.5k23.32bit",
    width: int = 800,
    height: int = 800,
    crs: str = "EPSG:25832",
    base_url: str = "https://www502.regione.toscana.it/ows_ofc/com.rt.wms.RTmap/wms",
) -> bytes | None:
    """
    Fetch a GeoTIFF tile from a WMS service and return the raw bytes.

    Args:
        bbox: Bounding box [minX, minY, maxX, maxY] in the specified CRS.
        layer: WMS layer to request.
        width: Output image width in pixels.
        height: Output image height in pixels.
        crs: Coordinate Reference System for the BBOX.
        base_url: Base URL of the WMS service.

    Returns:
        Raw bytes of the GeoTIFF image, or None if the request failed.
    """
    params = _build_wms_params(bbox, layer, width, height, crs)
    bbox_str = params["BBOX"]

    print(f"Requesting 800x800 GeoTIFF for BBOX: {bbox_str}...")

    response: Optional[requests.Response] = None
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()

        content_type = response.headers.get("Content-Type", "")

        if "image/tiff" in content_type:
            print(f"Successfully fetched tile for BBOX: {bbox_str}")
            return response.content
        else:
            print("Error: Server did not return a GeoTIFF.")
            print(f"Response Content-Type: {content_type}")
            print(f"Server Response (first 500 chars):\n{response.text[:500]}...")
            return None

    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error: {e}")
        if response is not None:
            print(f"Response text: {response.text}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return None


def download_wms_geotiff_to_file(
    bbox: List[float],
    output_filepath: str,
    layer: str = "rt_ofc.5k23.32bit",
    width: int = 800,
    height: int = 800,
    crs: str = "EPSG:25832",
    base_url: str = "https://www502.regione.toscana.it/ows_ofc/com.rt.wms.RTmap/wms",
) -> bool:
    """
    Download a GeoTIFF tile from a WMS service and save to file.

    Args:
        bbox: Bounding box [minX, minY, maxX, maxY] in the specified CRS.
        output_filepath: Path to save the downloaded .tif file.
        layer: WMS layer to request.
        width: Output image width in pixels.
        height: Output image height in pixels.
        crs: Coordinate Reference System for the BBOX.
        base_url: Base URL of the WMS service.

    Returns:
        True if successful, False otherwise.
    """
    image_data = fetch_wms_geotiff(bbox, layer, width, height, crs, base_url)
    
    if image_data:
        with open(output_filepath, "wb") as f:
            f.write(image_data)
        print(f"Successfully downloaded tile to: {output_filepath}")
        return True
    
    return False


def generate_tiles(start: Point, end: Point, step_in_m: int = 80) -> List[Tile]:
    """
    Generate a list of tiles covering an area.

    Args:
        start: Bottom-left corner point (EPSG:25832).
        end: Top-right corner point (EPSG:25832).
        step_in_m: Step size in meters between tile centers.

    Returns:
        List of Tile objects covering the area.
    """
    current = deepcopy(start)
    tiles: List[Tile] = []

    while current.y < end.y:
        while current.x < end.x:
            tiles.append(Tile(Point(current.x, current.y)))
            current.x = current.x + step_in_m
        current.y = current.y + step_in_m
        current.x = start.x

    return tiles


def download_tiles(
    start: Point, 
    end: Point, 
    max_workers: int = 10,
    step_in_m: int = 80,
) -> None:
    """
    Download tiles covering an area to files.

    Args:
        start: Bottom-left corner point (EPSG:25832).
        end: Top-right corner point (EPSG:25832).
        max_workers: Maximum number of concurrent downloads.
        step_in_m: Step size in meters between tile centers.
    """
    tiles = generate_tiles(start, end, step_in_m)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        executor.map(lambda t: t.download(), tiles)


def process_tiles(
    start: Point, 
    end: Point, 
    process_fn: Callable[[Tile], None],
    max_workers: int = 10,
    step_in_m: int = 80,
) -> None:
    """
    Process tiles covering an area with a custom function.

    Args:
        start: Bottom-left corner point (EPSG:25832).
        end: Top-right corner point (EPSG:25832).
        process_fn: Function to call for each tile.
        max_workers: Maximum number of concurrent workers.
        step_in_m: Step size in meters between tile centers.
    """
    tiles = generate_tiles(start, end, step_in_m)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        executor.map(process_fn, tiles)


def extract_tree_coordinates_from_geotiff(
    image_source: str | bytes,
    predictions: geopandas.GeoDataFrame,
) -> List[Tuple[float, float]]:
    """
    Extract geographic coordinates for detected trees from predictions in WGS 84 (EPSG:4326).

    Args:
        image_source: Either a file path (str) or in-memory GeoTIFF data (bytes).
        predictions: GeoDataFrame containing bounding box predictions with columns:
                     xmin, ymin, xmax, ymax (in pixel coordinates).

    Returns:
        List of tuples (longitude, latitude) representing WGS 84 coordinates of tree centers.
    """
    coordinates = []

    # Handle both file path and bytes input
    if isinstance(image_source, bytes):
        src_input = io.BytesIO(image_source)
    else:
        src_input = image_source

    with rasterio.open(src_input) as src:
        transform = src.transform
        source_crs = src.crs

        print(f"Debug - Transform: {transform}")
        print(f"Debug - Source CRS: {source_crs}")
        print(f"Debug - Bounds: {src.bounds}")

        for idx, pred in predictions.iterrows():
            # Calculate center point of bounding box in pixel coordinates
            center_col = pred["xmin"] + (pred["xmax"] - pred["xmin"]) / 2
            center_row = pred["ymin"] + (pred["ymax"] - pred["ymin"]) / 2

            # Convert pixel coordinates to source CRS geographic coordinates
            geo_x, geo_y = transform * (center_col, center_row)

            # Transform from source CRS to WGS 84 (EPSG:4326)
            transformed = rio_transform(source_crs, "EPSG:4326", [geo_x], [geo_y])
            lon, lat = transformed[0], transformed[1]

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
