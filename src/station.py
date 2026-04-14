import datetime
from enum import Enum

class MeasurementField(Enum):
    """
    Enum of supported measurement fields.

    Each value stores:
        - the CSV column name
        - whether the field should be graphed
    """
    PRCP = ("PRCP", True)
    TAVG = ("TAVG", True)
    TMAX = ("TMAX", True)
    TMIN = ("TMIN", True)
    AWND = ("AWND", True)
    SNOW = ("SNOW", True)
    WDF2 = ("WDF2", True)
    WDF5 = ("WDF5", True)
    WSF2 = ("WSF2", True)
    WSF5 = ("WSF5", True)

    def __init__(self, column_name, graphable):
        self.column_name = column_name
        self.graphable = graphable


class StationInfo:
    def __init__(self, name=None, year=None, month=None, longitude=None, latitude=None, elevation=None):
        """
        Create and store general information about a weather station
        for one specific year and month.

        Parameters:
            name (str): Name of the weather station.
            year (int): Year of the measurement, like 1995.
            month (int): Month of the measurement, from 1 to 12.
            longitude (float): Station longitude in decimal degrees.
            latitude (float): Station latitude in decimal degrees.
            elevation (float): Station elevation, usually in meters.
        """
        self.name = name
        self.year = year
        self.month = month

        if self.year is not None and self.month is not None:
            self.date = datetime.datetime(self.year, self.month, 1)
        else:
            self.date = None

        self.longitude = longitude
        self.latitude = latitude
        self.elevation = elevation
        self.coords = (self.longitude, self.latitude, self.elevation)


class StationMeasurements:
    def __init__(self, station_info, values=None):
        """
        Create and store weather measurements for one station and month.

        Parameters:
            station_info (StationInfo): The station metadata.
            values (dict): Dictionary mapping field names to values.
        """
        self.station_info = station_info
        self.values = values if values is not None else {}

        for field_name, value in self.values.items():
            setattr(self, field_name, value)

    def get(self, field: MeasurementField):
        """
        Return the value for a MeasurementField enum member.
        """
        return self.values.get(field.column_name, None)
