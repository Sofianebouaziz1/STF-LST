# STF-LST Dataset : Spatio Temporal Fusion Dataset for Land Surface Temperature Estimation

The <strong>STF-LST Dataset</strong> is a robust foundation for developing and evaluating innovative spatio-temporal fusion techniques, specifically designed to address the challenges of land surface temperature estimation. This dataset includes 51 paired <em>MODIS/Landsat</em> images, each with a resolution of <em>950 x 950</em> pixels. hey were collected between March 18, 2013, and October 15, 2024, and they cover the Orleans Métropole in the Centre-Val de Loire region of France.

Below is a video showcasing diverse samples from the STF-LST Dataset (click on the picture and download the video) : 

[![Watch the video](example_sample.png)](.mp4)


## Features

The STF-LST Dataset offers the following features:

* 51 paired MODIS-Landsat images covering a wide range of time periods.
* Various preprocessing techniques were applied, including linear, spatial, and bicubic interpolation methods.
* A fully reproducible codebase that can be adapted for different regions and time periods by simply adjusting the parameters.

## Guide of use
To generate the STF-LST dataset, run the following command in your terminal:  

```bash
python3 run.py
```
The code utilizes the Google Earth Engine platform, so you will need a valid account for authentication before downloading the data.

Please note that this process may take some time.


## Requirements
STF-LST dataset has been generated using the following versions: 
- Python (v3.9.19).
- Torch (v2.4.1+cu121).
- Scipy (v1.13.1).
- Earth Engine (v1.1.2).
- Geemap (v0.34.5).
- Rasterio (v1.3.10).
- NumPy (v1.26.4).
- Pandas (V2.2.2).
