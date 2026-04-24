import datetime
from enum import Enum
from typing import Iterable, Sequence

import numpy as np


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
        Create and store weather measurements for one station and month.

        Parameters:
            station_info (StationInfo): The station metadata.
            values (dict): Dictionary mapping field names to values.
        """
        self.station_info = station_info
        self.values = values if values is not None else {}

        for field_name, value in self.values.items():
            setattr(self, field_name, value)

    def __str__(self):
        return f"StationMeasurements(station_info={self.station_info}, values={self.values})"

    def get(self, field: MeasurementField):
        """
        Return the value for a MeasurementField enum member.
        """
        return self.values.get(field.column_name, None)


def graphable_fields() -> list[MeasurementField]:
    """
    Return only fields marked as graphable.
    """
    return [field for field in MeasurementField if field.graphable]


def measurements_to_matrix(
    measurements: Iterable[StationMeasurements],
    fields: Sequence[MeasurementField] | None = None,
    drop_incomplete_rows: bool = False,
    sort_by_date: bool = True,
) -> tuple[np.ndarray, list[datetime.datetime], list[MeasurementField]]:
    """
    Convert station measurements into a numeric matrix for linear algebra.

    Returns:
        matrix: shape (n_rows, n_fields), with missing values represented as np.nan.
        dates: list of datetime values aligned with matrix rows.
        selected_fields: the field order used for matrix columns.
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
