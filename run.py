import os
import numpy as np
import geemap
import argparse
from src.LandsatProcessor import LandsatProcessor
from src.MODISProcessor import MODISProcessor
from src.GetPairs import GetPairs
from src.DataProcessor import DataProcessor

# Process Landsat 8 images: download, filter, interpolate, get LST, and export
def process_landsat(start_date='2013-03-18', end_date='2024-10-15', roi=[1.767713, 47.760413, 2.109171, 48.013396]):
    landsat_processor = LandsatProcessor(start_date=start_date, end_date=end_date, bounds=roi)
    l8_data = landsat_processor.get_Landsat_collection()
    print('Images before filtering:', landsat_processor.countImages(l8_data))
    l8_data_filtered = landsat_processor.filter_disponible_images(l8_data, 80)
    print('Images after filtering:', landsat_processor.countImages(l8_data_filtered))
    l8_data_interpolated = landsat_processor.temporalInterpolation(l8_data_filtered, 32)
    print('Images after interpolation:', landsat_processor.countImages(l8_data_interpolated))
    l8_lst_data_interpolated = landsat_processor.get_LST(l8_data_interpolated)
    l8_times_interpolated = landsat_processor.get_times(l8_data_interpolated)
    out_dir = "data/Landsat"
    os.makedirs(out_dir, exist_ok=True)
    geemap.ee_export_image_collection(
        l8_lst_data_interpolated,
        out_dir=out_dir,
        scale=30,
        region=landsat_processor.aoi
    )
    return l8_times_interpolated

# Process Modis Terra images: download, filter, get LST, and export
def process_modis_with_common_dates(l8_times_interpolated, start_date='2013-03-18', end_date='2024-10-15', roi=[1.767713, 47.760413, 2.109171, 48.013396]):
    print("Processing MODIS data...")
    modis_processor = MODISProcessor(start_date=start_date, end_date=end_date, bounds=roi)
    modis_data = modis_processor.get_MODIS_collection()
    print('MODIS images before filtering:', modis_processor.countImages(modis_data))
    modis_data_filtered = modis_processor.filter_disponible_images(modis_data, 90)  # Apply 90% availability threshold
    print('MODIS images after filtering:', modis_processor.countImages(modis_data_filtered))
    modis_lst_data = modis_processor.get_LST(modis_data_filtered)
    modis_times = modis_processor.get_formatted_times(modis_data_filtered)
    common_dates_array = find_common_dates(l8_times_interpolated, modis_times)
    modis_lst_data_common = modis_processor.filter_by_common_dates(modis_lst_data, common_dates_array)
    out_dir = 'data/MODIS'
    os.makedirs(out_dir, exist_ok=True)
    geemap.ee_export_image_collection(
        modis_lst_data_common,
        out_dir=out_dir,
        scale=1000,
        region=modis_processor.aoi
    )
    
# Identify common dates between Landsat and MODIS and save them as a .npy file
def find_common_dates(l8_times_interpolated, modis_times):
    landsat_dates_only = [date.split('T')[0] for date in l8_times_interpolated]
    modis_dates_only = [date.split(' ')[0] for date in modis_times]
    common_dates_set = set(landsat_dates_only).intersection(set(modis_dates_only))
    common_dates_sorted = sorted(common_dates_set)
    common_dates_array = np.array(common_dates_sorted)
    np.save('data/commun_dates.npy', common_dates_array)
    return common_dates_array

# Generate paired dataset of MODIS-Landsat images, apply spatial interpolation, and save results
def generate_pairs_and_interpolate():
    get_pair = GetPairs()
    dates = np.load('data/commun_dates.npy')
    landsat_images = get_pair.load_landsat('data/Landsat', dates)
    modis_images = get_pair.load_modis('data/MODIS', dates)
    out_dir = "data/Pairs_MODIS_Landsat"
    os.makedirs(out_dir, exist_ok=True)
    get_pair.save_landsat_formatted(landsat_images, dates, out_dir)
    get_pair.save_modis_formatted(modis_images, dates, out_dir)
    data_processor = DataProcessor()
    landsat_images_filled = data_processor.progressive_focal_mean(landsat_images)
    modis_images_filled = data_processor.progressive_focal_mean(modis_images)
    height, width = landsat_images_filled[0][0].shape
    modis_images_interpolated = data_processor.resize_modis_images(modis_images_filled, height, width)
    out_dir = "data/Pairs_MODIS_Landsat_filled"
    os.makedirs(out_dir, exist_ok=True)
    get_pair.save_landsat_formatted(landsat_images_filled, dates, out_dir)
    get_pair.save_modis_formatted(modis_images_interpolated, dates, out_dir)

# Main script entry point: parse arguments, process Landsat and MODIS, then generate pairs
def main():
    # Argument parser to handle optional arguments
    parser = argparse.ArgumentParser(description="Process Landsat and MODIS data.")
    parser.add_argument('--start_date', type=str, default='2013-03-18', help="Start date (default: '2013-03-18')")
    parser.add_argument('--end_date', type=str, default='2024-10-15', help="End date (default: '2024-10-15')")
    parser.add_argument('--roi', type=float, nargs=4, default=[1.7505414127167631, 47.76054614894737, 2.126342587283237, 48.01326285105263],
                        help="Region of interest (default: [1.7505414127167631, 47.76054614894737, 2.126342587283237, 48.01326285105263])")

    # Parse arguments
    args = parser.parse_args()

    print("*** Download Landsat LST data ***")
    l8_times_interpolated = process_landsat(args.start_date, args.end_date, args.roi)

    print("*** Download and process MODIS LST data ***")
    process_modis_with_common_dates(l8_times_interpolated, args.start_date, args.end_date, args.roi)

    print("*** Generate dataset: Pairs MODIS-Landsat ***")
    generate_pairs_and_interpolate()
    print("Processing complete. Results saved.")

if __name__ == "__main__":
    main()