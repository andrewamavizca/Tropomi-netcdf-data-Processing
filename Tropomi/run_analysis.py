# run_analysis.py

import argparse
import matplotlib.pyplot as plt
import numpy as np
from my_module import MethaneDataProcessor

def main(file_name, bbox=None):
    processor = MethaneDataProcessor(file_name)
    scenes = processor.scenes
    
    np.save(f'test.npy', scenes, allow_pickle=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and visualize methane data from netCDF files.")
    parser.add_argument('file_name', type=str, help="The path to the netCDF file.")
    parser.add_argument('--bbox', nargs=4, type=float, help="Bounding box for filtering scenes: min_lat max_lat min_lon max_lon")

    args = parser.parse_args()
    bbox = args.bbox if args.bbox else None
    main(args.file_name, bbox)