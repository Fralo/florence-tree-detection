"""
Download Florence tiles from WMS service.
"""

from wms_utils import Point, download_tiles


if __name__ == "__main__":
    start_point = Point(674048.64, 4852250.78)
    end_point = Point(675960.26, 4853751.03)

    download_tiles(start=start_point, end=end_point)
