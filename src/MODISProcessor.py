import ee
import geemap

class MODISProcessor:
    def __init__(self, start_date, end_date, bounds):
        """
        Initialize the MODISProcessor with study area and date range.
        
        Parameters:
        - start_date (str): The start date of the image collection (format: 'YYYY-MM-DD').
        - end_date (str): The end date of the image collection (format: 'YYYY-MM-DD').
        - bounds (list): List of coordinates defining the area of interest (AOI) in [xmin, ymin, xmax, ymax] format.
        """
        # Initialize the Earth Engine module
        # ee.Authenticate()  # Uncomment if authentication is needed
        ee.Initialize()

        # Define the study area and date range
        self.aoi = ee.Geometry.Rectangle(bounds)
        self.start_date = start_date
        self.end_date = end_date

    def get_MODIS_collection(self):
        """
        Get the MODIS LST collection filtered by date and bounds.
        """
        return ee.ImageCollection('MODIS/061/MOD11A1') \
            .filterBounds(self.aoi) \
            .filterDate(self.start_date, self.end_date) \
            .map(self.applyQDMask) 

    def toCelsiusDay(self, image):
        """
        Convert LST from Kelvin to Celsius for daytime images.
        """
        lst = image.select('LST_Day_1km').multiply(0.02).subtract(273.15)
        return image.addBands(lst.rename('LST_Day_1km'), None, True)

    def bitwiseExtract(self, value, fromBit, toBit):
        """
        Extract bit range from a given value.
        """
        maskSize = ee.Number(1).add(toBit).subtract(fromBit)
        mask = ee.Number(1).leftShift(maskSize).subtract(1)
        return value.rightShift(fromBit).bitwiseAnd(mask)

    def applyQDMask(self, image):
        """
        Apply a quality mask to MODIS LST daytime images based on the QC_Day band.
        """
        qcDay = image.select('QC_Day')
        qaMask = self.bitwiseExtract(qcDay, 0, 1).lte(1)
        dataQualityMask = self.bitwiseExtract(qcDay, 2, 3).eq(0)
        lstErrorMask = self.bitwiseExtract(qcDay, 6, 7).lte(2)
        mask = qaMask.And(dataQualityMask).And(lstErrorMask)
        return image.updateMask(mask)

    def calculatePixelAvailability_MODIS(self, image):
        """
        Calculate the percentage of valid (non-masked) pixels for MODIS LST images.
        """
        totalPixels = image.select('QC_Day').mask().reduceRegion(
            reducer=ee.Reducer.count(),
            geometry=self.aoi,
            scale=1000,
            maxPixels=1e9
        ).values().get(0)

        validPixels = image.select('QC_Day').reduceRegion(
            reducer=ee.Reducer.count(),
            geometry=self.aoi,
            scale=1000,
            maxPixels=1e9
        ).values().get(0)

        pixelAvailability = ee.Number(validPixels).divide(totalPixels).multiply(100)
        return image.set('pixelAvailability', pixelAvailability)

    def filter_disponible_images(self, collection, pourcentage):
        """
        Filter the MODIS image collection based on the percentage of valid (non-masked) pixels.
        
        Parameters:
        - collection (ee.ImageCollection): The image collection to filter.
        - pourcentage (float): The minimum percentage of valid pixels required.

        Returns:
        - ee.ImageCollection: The filtered image collection.
        """
        modis_filtered = collection.map(self.calculatePixelAvailability_MODIS)
        modis_filtered = modis_filtered.filter(ee.Filter.gte('pixelAvailability', pourcentage))
        return modis_filtered

    def apply_scale_factors_time(self, image):
        """
        Apply scale factors to the Day_view_time band for time calculations.
        """
        optical_bands = image.select('Day_view_time').multiply(0.1)
        return image.addBands(optical_bands, None, True)

    def format_time(self, image):
        """
        Compute and format the Day_view_time for each MODIS image.
        """
        day_view_time_band = image.select('Day_view_time')
        
        stats = day_view_time_band.reduceRegion(
            reducer=ee.Reducer.mean()
              .combine(reducer2=ee.Reducer.max(), sharedInputs=True)
              .combine(reducer2=ee.Reducer.min(), sharedInputs=True),
            geometry=self.aoi,
            scale=1000,  # Adjust the scale based on the resolution of MODIS
            maxPixels=1e4
        )
        
        time = stats.get('Day_view_time_mean')

        hours = ee.Number(time).floor().divide(1).int()
        minutes = ee.Number(time).subtract(hours).multiply(60).round().int()
        result = ee.String(hours.format('%02d')).cat(':').cat(minutes.format('%02d')).cat(':00')

        system_time_start = ee.Date(image.get('system:time_start')).format('YYYY-MM-dd ')
        sts = system_time_start.cat(result)

        return ee.Feature(None, {'formattedTime': sts})

    def get_formatted_times(self, collection):
        """
        Retrieve and format the times for each image in the MODIS collection.
        """
        modis_lst_day = collection.map(self.apply_scale_factors_time)
        formatted_times = modis_lst_day.map(self.format_time)
        times_list = formatted_times.aggregate_array('formattedTime')
        return times_list.getInfo()
    
    def countImages(self, collection):
        """
        Count the number of images in the filtered collection.
        """
        return collection.size().getInfo()

    def get_LST(self, collection):
        collection  = collection.map(self.toCelsiusDay)
        return collection.select('LST_Day_1km')
    

    # Convert the date strings to ee.Date objects
    def date_filter(self, date_str):
        # Convert to ee.Date
        return ee.Filter.date(ee.Date(date_str), ee.Date(date_str).advance(1, 'day'))

    # Filter the MODIS_LST_data collection using the common dates
    def filter_by_common_dates(self, collection, common_dates_array):
        filters = ee.Filter.Or(*[self.date_filter(date) for date in common_dates_array])
        return collection.filter(filters)



    def addTimeBand(self, image):
        """
        Add a time band to each image for interpolation.
        """
        timeImage = image.metadata('system:time_start').rename('timestamp')
        timeImageMasked = timeImage.updateMask(image.mask().select(0))
        return image.addBands(timeImageMasked)

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
        collection = collection.map(self.addTimeBand)
        millis = ee.Number(days).multiply(1000 * 60 * 60 * 24)

        joinedImages = self.joinImages(collection, millis)
        interpolatedCollection = ee.ImageCollection(joinedImages.map(self.interpolateImages))

        return interpolatedCollection
    
    # Filter the MODIS_LST_data collection using the common dates
    def filter_by_common_dates(self, collection, common_dates_array):
        filters = ee.Filter.Or(*[self.date_filter(date) for date in common_dates_array])
        return collection.filter(filters)

