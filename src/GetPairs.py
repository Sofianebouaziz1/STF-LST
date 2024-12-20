import os
import rasterio
import pandas as pd

class GetPairs:
    def load_landsat(self, path, dates):
        """Load Landsat images along with their CRS and transform."""
        landsat_images = []

        common_dates = pd.Series(dates)
        formatted_dates = common_dates.apply(lambda x: pd.to_datetime(x).strftime('%Y%m%d')).tolist()

        for date in formatted_dates:
            filename_pattern = f"LC08_199027_{date}.tif"
            file_path = os.path.join(path, filename_pattern)
              
            if os.path.exists(file_path):
                print(f"Reading Landsat file: {file_path}")
                with rasterio.open(file_path) as src:
                    array = src.read(1)  # Read the first band (image data)
                    crs = src.crs  # Get CRS
                    transform = src.transform  # Get transform (affine matrix)
                    
                    landsat_images.append((array, crs, transform))  # Append as a tuple (image, crs, transform)

        return landsat_images

    def load_modis(self, path, dates):
        """Load MODIS images along with their CRS and transform."""
        modis_images = []

        common_dates = pd.Series(dates)
        formatted_dates = common_dates.apply(lambda x: pd.to_datetime(x).strftime('%Y_%m_%d')).tolist()

        for date in formatted_dates:
            filename_pattern = f"{date}.tif"  # Assuming MODIS files are saved with .tif extension
            file_path = os.path.join(path, filename_pattern)

            if os.path.exists(file_path):
                print(f"Reading MODIS file: {file_path}")
                with rasterio.open(file_path) as src:
                    array = src.read(1)  # Read the first band (image data)
                    crs = src.crs  # Get CRS
                    transform = src.transform  # Get transform (affine matrix)
                    
                    modis_images.append((array, crs, transform))  # Append as a tuple (image, crs, transform)

        return modis_images

    def save_landsat_formatted(self, landsat_images, dates, output_folder):
        """Save formatted Landsat images to the specified output folder."""
        dates = pd.Series(dates)
        
        for (image, crs, transform), date in zip(landsat_images, dates):
            filename = f"L_{pd.to_datetime(date).strftime('%Y%m%d')}.tif"
            file_path = os.path.join(output_folder, filename)

            with rasterio.open(
                file_path,
                'w',
                driver='GTiff',
                height=image.shape[0],
                width=image.shape[1],
                count=1,  # Assuming 1 band
                dtype=image.dtype,
                crs=crs,  # Use the CRS from the input file
                transform=transform  # Use the transform from the input file
            ) as dst:
                dst.write(image, 1)

    def save_modis_formatted(self, modis_images, dates, output_folder):
        """Save formatted MODIS images to the specified output folder."""
        dates = pd.Series(dates)

        for (image, crs, transform), date in zip(modis_images, dates):
            filename = f"M_{pd.to_datetime(date).strftime('%Y%m%d')}.tif"
            file_path = os.path.join(output_folder, filename)

            with rasterio.open(
                file_path,
                'w',
                driver='GTiff',
                height=image.shape[0],
                width=image.shape[1],
                count=1,  # Assuming 1 band
                dtype=image.dtype,
                crs=crs,  # Use the CRS from the input file
                transform=transform  # Use the transform from the input file
            ) as dst:
                dst.write(image, 1)
