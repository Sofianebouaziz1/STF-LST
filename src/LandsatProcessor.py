import ee
import geemap
import pandas as pd

class LandsatProcessor:
    def __init__(self, start_date, end_date, bounds):
        """
        Initialize the LandsatProcessor with study area and date range.
        
        Parameters:
        - start_date (str): The start date of the image collection (format: 'YYYY-MM-DD').
        - end_date (str): The end date of the image collection (format: 'YYYY-MM-DD').
        - bounds (list): List of coordinates defining the area of interest (AOI) in [xmin, ymin, xmax, ymax] format.
        """
        # Initialize the Earth Engine module
        ee.Authenticate()  # Uncomment if authentication is needed
        ee.Initialize()

        # Define the study area and date range
        self.aoi = ee.Geometry.Rectangle(bounds)
        self.start_date = start_date
        self.end_date = end_date

        
    def get_Landsat_collection(self):

        return ee.ImageCollection('LANDSAT/LC08/C02/T1_L2') \
            .filterBounds(self.aoi) \
            .filterDate(self.start_date, self.end_date) \
            .map(self.applyScaleFactors) \
            .map(self.cloudMask) \
            .map(self.addTimeBand)
        

    def cloudMask(self, image):
        """
        Apply a cloud mask to the image using the pixel quality band.
        """
        # Bits 3 and 5 are cloud shadow and cloud, respectively.
        cloudShadowBitMask = (1 << 3)
        cloudsBitMask = (1 << 5)

        # Get the pixel QA band.
        qa = image.select('QA_PIXEL')

        # Both flags should be set to zero, indicating clear conditions.
        mask = qa.bitwiseAnd(cloudShadowBitMask).eq(0).And(
               qa.bitwiseAnd(cloudsBitMask).eq(0))

        return image.updateMask(mask)

    def applyScaleFactors(self, image):
        """
        Apply scale factors to optical and thermal bands.
        """
        opticalBands = image.select('SR_B.*').multiply(0.0000275).add(-0.2)
        thermalBands = image.select('ST_B.*').multiply(0.00341802).add(149.0)
        return image.addBands(opticalBands, None, True).addBands(thermalBands, None, True)

    def addTimeBand(self, image):
        """"""  """
        Add a time band to each image for interpolation.
        """
        timeImage = image.metadata('system:time_start').rename('timestamp')
        timeImageMasked = timeImage.updateMask(image.mask().select(0))
        return image.addBands(timeImageMasked)

    def calculatePixelAvailability(self, image):
        """
        Calculate the percentage of valid pixels (non-masked) in the image.
        """
        totalPixels = image.select('QA_PIXEL').mask().reduceRegion(
            reducer=ee.Reducer.count(),
            geometry=self.aoi,
            scale=30,
            maxPixels=1e9
        ).values().get(0)

        validPixels = image.select('QA_PIXEL').reduceRegion(
            reducer=ee.Reducer.count(),
            geometry=self.aoi,
            scale=30,
            maxPixels=1e9
        ).values().get(0)

        pixelAvailability = ee.Number(validPixels).divide(totalPixels).multiply(100)
        return image.set('pixelAvailability', pixelAvailability)

    def joinImages(self, collection, millis):
        """
        Apply temporal joins to match images within a specific time window.
        Returns an image collection with 'before' and 'after' image lists for each image.
        """
        # Define filters for the join
        maxDiffFilter = ee.Filter.maxDifference(
            difference= millis,
            leftField='system:time_start',
            rightField='system:time_start'
        )

        lessEqFilter = ee.Filter.lessThanOrEquals(
            leftField='system:time_start',
            rightField='system:time_start'
        )

        greaterEqFilter = ee.Filter.greaterThanOrEquals(
            leftField='system:time_start',
            rightField='system:time_start'
        )

        # Join 1: Match all images after the given image within the time-window
        filter1 = ee.Filter.And(maxDiffFilter, lessEqFilter)
        join1 = ee.Join.saveAll(
            matchesKey='after',
            ordering='system:time_start',
            ascending=False
        )
        join1Result = join1.apply(primary= collection, secondary= collection, condition=filter1)

        # Join 2: Match all images before the given image within the time-window
        filter2 = ee.Filter.And(maxDiffFilter, greaterEqFilter)
        join2 = ee.Join.saveAll(
            matchesKey='before',
            ordering='system:time_start',
            ascending=True
        )
        join2Result = join2.apply(primary=join1Result, secondary=join1Result, condition=filter2)

        return join2Result

    def interpolateImages(self, image):
        """
        Interpolate between the 'before' and 'after' images.
        """
        image = ee.Image(image)

        beforeImages = ee.List(image.get('before'))
        beforeMosaic = ee.ImageCollection.fromImages(beforeImages).mosaic()
        afterImages = ee.List(image.get('after'))
        afterMosaic = ee.ImageCollection.fromImages(afterImages).mosaic()


        t1 = beforeMosaic.select('timestamp').rename('t1')
        t2 = afterMosaic.select('timestamp').rename('t2')
        t = image.metadata('system:time_start').rename('t')

        timeImage = ee.Image.cat([t1, t2, t])
        timeRatio = timeImage.expression('(t - t1) / (t2 - t1)', {
            't': timeImage.select('t'),
            't1': timeImage.select('t1'),
            't2': timeImage.select('t2'),
        })


        interpolated = beforeMosaic.add((afterMosaic.subtract(beforeMosaic).multiply(timeRatio)))
        result = image.unmask(interpolated)

        return result.copyProperties(image, ['system:time_start'])

    def temporalInterpolation(self, collection, days):
        """
        Apply interpolation to the image collection and return the result.
        """
        millis = ee.Number(days).multiply(1000 * 60 * 60 * 24)

        joinedImages = self.joinImages(collection, millis)
        interpolatedCollection = ee.ImageCollection(joinedImages.map(self.interpolateImages))

        return interpolatedCollection

    def calculateLST(self, image):
        """
        Calculate the Land Surface Temperature (LST) from the thermal band.
        """
        thermalLST = image.select('ST_B10').subtract(273.15).rename('LST_thermal')
        return image.addBands(thermalLST)

    def countImages(self, collection):
        """
        Count the number of images in the filtered collection.
        """
        return collection.size().getInfo()


    def filter_disponible_images(self, collection, pourcentage):
        """
        Filter the image collection based on the percentage of valid (non-masked) pixels.

        Parameters:
        - collection (ee.ImageCollection): The image collection to filter.
        - pourcentage (float): The minimum percentage of valid pixels required.

        Returns:
        - ee.ImageCollection: The filtered image collection.
        """
        # Apply the calculatePixelAvailability function to each image in the collection
        L8_data_filtered = collection.map(self.calculatePixelAvailability)

        # Filter the images based on the pixelAvailability property (at least the specified percentage)
        L8_data_filtered = L8_data_filtered.filter(ee.Filter.gte('pixelAvailability', pourcentage))
        
        return L8_data_filtered

    def get_image(self, collection, image_index):
        count = self.countImages(collection)

        return ee.Image(collection.toList(count).get(image_index))
    
    def get_LST(self, collection):
        collection = collection.map(self.calculateLST)
        return collection.select('LST_thermal')
    

    def get_times(self, collection):
        dates = collection.aggregate_array("system:time_start")

        # Format the dates using map with a lambda function
        dates = dates.map(lambda ele: ee.Date(ele).format())

        # Print the formatted dates
        return dates.getInfo()
    
    def filter_by_common_dates(self, collection, common_dates_array):
        filters = ee.Filter.Or(*[self.date_filter(date) for date in common_dates_array])
        return collection.filter(filters)
