import datetime
from enum import Enum
from typing import Iterable, Sequence

import numpy as np


class MeasurementField(Enum):
    """
    Weather fields used in this project.

    Each enum value stores:
    1. CSV column name
    2. whether we want to graph it
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
    def __init__(
        self,
        name=None,
        year=None,
        month=None,
        day=None,
        longitude=None,
        latitude=None,
        elevation=None,
    ):
        """
        Store station metadata for one row in the dataset.

        Parameters:
            name (str): Station name.
            year (int): Year of this record.
            month (int): Month of this record.
            longitude (float): Longitude.
            latitude (float): Latitude.
            elevation (float): Elevation.
        """
        self.name = name
        self.year = year
        self.month = month
        self.day = day

        if self.year is not None and self.month is not None:
            safe_day = self.day if self.day is not None else 1
            self.date = datetime.datetime(self.year, self.month, safe_day)
        else:
            self.date = None

        self.longitude = longitude
        self.latitude = latitude
        self.elevation = elevation
        self.coords = (self.longitude, self.latitude, self.elevation)

    def __str__(self):
        return f"StationInfo(name={self.name}, date={self.date}, coords={self.coords})"


class StationMeasurements:
    def __init__(self, station_info: StationInfo, values=None):
        """
        Store measured values for one station/date record.

        Parameters:
            station_info (StationInfo): Metadata (name/date/location).
            values (dict): Map from field name to numeric value.
        """
        self.station_info = station_info
        self.values = values if values is not None else {}

        for field_name, value in self.values.items():
            setattr(self, field_name, value)

    def __str__(self):
        return f"StationMeasurements(station_info={self.station_info}, values={self.values})"

    def get(self, field: MeasurementField):
        """
        Get the value for one field enum.
        """
        return self.values.get(field.column_name, None)


def graphable_fields() -> list[MeasurementField]:
    """
    Return only fields that are marked for plotting.
    """
    return [field for field in MeasurementField if field.graphable]


def measurements_to_matrix(
    measurements: Iterable[StationMeasurements],
    fields: Sequence[MeasurementField] | None = None,
    drop_incomplete_rows: bool = False,
    sort_by_date: bool = True,
) -> tuple[np.ndarray, list[datetime.datetime], list[MeasurementField]]:
    """
    Convert records into a numeric matrix for linear algebra work.

    Returns:
        matrix: shape (rows, fields), missing entries are np.nan.
        dates: date for each matrix row.
        selected_fields: column order used in the matrix.
    """
    selected_fields = list(fields) if fields is not None else list(MeasurementField)

    measurements_list = list(measurements)
    if sort_by_date:
        measurements_list.sort(
            key=lambda m: m.station_info.date or datetime.datetime.min
        )

    rows: list[list[float]] = []
    dates: list[datetime.datetime] = []

    for measurement in measurements_list:
        date = measurement.station_info.date
        if date is None:
            continue

        row: list[float] = []
        has_missing_value = False

        for field in selected_fields:
            value = measurement.get(field)
            if value is None:
                row.append(np.nan)
                has_missing_value = True
            else:
                row.append(float(value))

        if drop_incomplete_rows and has_missing_value:
            continue

        rows.append(row)
        dates.append(date)

    if not rows:
        return np.empty((0, len(selected_fields))), [], selected_fields

    return np.array(rows, dtype=float), dates, selected_fields
