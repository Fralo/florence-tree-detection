"""
Download Florence tiles from WMS service.
"""

from wms_utils import Point, download_tiles


if __name__ == "__main__":
    start_point = Point(680271.64,4848409.26)
    end_point = Point(682964.18,4849702.91)

    download_tiles(start=start_point, end=end_point)
