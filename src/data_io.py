import pandas as pd
import numpy as np

from station import (
    StationInfo,
    StationMeasurements,
    MeasurementField,
    graphable_fields,
    measurements_to_matrix,
)


def clean_value(value):
    """
    Turn pandas missing values into None.
    """
    if pd.isna(value):
        return None
    return value


def parse_date_parts(date_value) -> tuple[int | None, int | None, int | None]:
    """
    Read year, month, and day from a CSV date value.
    """
    if date_value is None:
        return None, None, None

    parsed_date = pd.to_datetime(str(date_value), errors="coerce")
    if pd.isna(parsed_date):
        return None, None, None

    return int(parsed_date.year), int(parsed_date.month), int(parsed_date.day)


def read_csv_rows(filepath: str) -> list:
    """
    Read weather rows from CSV into StationMeasurements objects.
    """
    csv = pd.read_csv(filepath)
    csv.columns = csv.columns.str.strip().str.upper()

    measurements_list = []

    for _, row in csv.iterrows():
        date_value = clean_value(row.get("DATE"))
        year, month, day = parse_date_parts(date_value)

        station_info = StationInfo(
            name=clean_value(row.get("NAME")),
            year=year,
            month=month,
            day=day,
            longitude=clean_value(row.get("LONGITUDE")),
            latitude=clean_value(row.get("LATITUDE")),
            elevation=clean_value(row.get("ELEVATION")),
        )

        values = {}
        for field in MeasurementField:
            values[field.column_name] = clean_value(row.get(field.column_name))

        measurement = StationMeasurements(station_info=station_info, values=values)
        measurements_list.append(measurement)

    return measurements_list


def field_series_from_matrix(measurements: list, field: MeasurementField):
    """
    Get (dates, values) for one field, dropping rows with missing values.
    """
    matrix, dates, _ = measurements_to_matrix(
        measurements,
        fields=[field],
        drop_incomplete_rows=True,
    )

    if matrix.size == 0:
        return [], []

    return dates, matrix[:, 0]


def print_matrix_preview(measurements: list):
    """
    Print a quick preview of the data matrix.
    """
    matrix, dates, fields = measurements_to_matrix(
        measurements,
        fields=graphable_fields(),
        drop_incomplete_rows=False,
    )

    print(f"Matrix shape: {matrix.shape}")
    print("Columns:", [field.column_name for field in fields])

    if matrix.size > 0:
        print("First date:", dates[0])
        print("First row:", matrix[0])
