from station import MeasurementField, graphable_fields
from data_io import read_csv_rows, print_matrix_preview
from plotting import (
    graph_all_fields,
    graph_field_by_time_bucket,
    graph_temperature_trend,
)


DEFAULT_DATA_FILE = "weather.csv"

TREND_MODE_ALIASES = {
    "": "auto",
    "a": "auto",
    "auto": "auto",
    "l": "linear",
    "lin": "linear",
    "linear": "linear",
    "p": "polynomial",
    "poly": "polynomial",
    "polynomial": "polynomial",
    "q": "polynomial",
    "quad": "polynomial",
    "quadratic": "polynomial",
    "b": "both",
    "both": "both",
}

AGGREGATED_TREND_MODE_ALIASES = {
    **TREND_MODE_ALIASES,
    "n": "none",
    "none": "none",
}


def normalize_choice(
    raw_value: str,
    aliases: dict[str, str],
    default_value: str,
    unknown_message: str,
) -> str:
    """
    Map user input aliases to one value, or use a default.
    """
    selected_value = aliases.get(raw_value)
    if selected_value is None:
        print(unknown_message)
        return default_value
    return selected_value


def get_trend_mode_override() -> str:
    """
    Ask the user which trend mode to use.

    Returns: auto, linear, polynomial, or both.
    """
    raw = (
        input("Trend model mode [auto/linear/polynomial/both] (default auto): ")
        .strip()
        .lower()
    )

    return normalize_choice(
        raw_value=raw,
        aliases=TREND_MODE_ALIASES,
        default_value="auto",
        unknown_message=f"Unknown mode '{raw}', using auto.",
    )


def get_aggregated_trend_mode_override() -> str:
    """
    Ask which trend mode to use on aggregated data.

    Returns: none, auto, linear, polynomial, or both.
    """
    raw = (
        input(
            "Aggregated trend mode [none/auto/linear/polynomial/both] "
            "(default auto): "
        )
        .strip()
        .lower()
    )

    return normalize_choice(
        raw_value=raw,
        aliases=AGGREGATED_TREND_MODE_ALIASES,
        default_value="auto",
        unknown_message=f"Unknown mode '{raw}', using auto.",
    )


def get_field_override(
    default_field: MeasurementField = MeasurementField.TAVG,
) -> MeasurementField:
    """
    Ask which weather field to analyze.
    """
    fields = graphable_fields()
    options = [field.column_name for field in fields]
    print("Available fields:", ", ".join(options))
    raw = (
        input(f"Field to analyze (default {default_field.column_name}): ")
        .strip()
        .upper()
    )

    if raw == "":
        return default_field

    for field in fields:
        if raw == field.column_name:
            return field

    print(f"Unknown field '{raw}', using {default_field.column_name}.")
    return default_field


def get_time_bucket_override() -> str:
    """
    Ask how to group data for comparison.

    Returns: year, month, day, or skip.
    """
    print(
        "Note: if your data is monthly, day-of-month is often fixed (usually 1), "
        "so day comparison may not be very informative."
    )
    raw = (
        input(
            "Compare data by [year/month/day/skip] (default skip). "
            "Tip: choose year or month for monthly datasets: "
        )
        .strip()
        .lower()
    )

    aliases = {
        "": "skip",
        "s": "skip",
        "skip": "skip",
        "y": "year",
        "year": "year",
        "m": "month",
        "month": "month",
        "d": "day",
        "day": "day",
    }

    return normalize_choice(
        raw_value=raw,
        aliases=aliases,
        default_value="skip",
        unknown_message=f"Unknown option '{raw}', skipping comparison plot.",
    )


def run_weather_analysis(filename: str = DEFAULT_DATA_FILE):
    """
    Run the full analysis flow: load data, graph, and fit trends.
    """
    measurements = read_csv_rows(filename)
    print_matrix_preview(measurements)
    graph_all_fields(measurements)

    selected_field = get_field_override(default_field=MeasurementField.TAVG)
    trend_mode = get_trend_mode_override()
    graph_temperature_trend(
        measurements,
        field=selected_field,
        future_months=120,
        mode=trend_mode,
    )

    bucket_mode = get_time_bucket_override()
    if bucket_mode == "skip":
        return

    aggregated_trend_mode = get_aggregated_trend_mode_override()
    graph_field_by_time_bucket(
        measurements,
        field=selected_field,
        bucket=bucket_mode,
        trend_mode=aggregated_trend_mode,
    )


if __name__ == "__main__":
    run_weather_analysis()
