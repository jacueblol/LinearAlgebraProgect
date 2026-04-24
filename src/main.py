import os
import datetime
import pandas as pd
import matplotlib.pyplot as plt
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
    Convert pandas missing values (NaN) into None.
    """
    if pd.isna(value):
        return None
    return value


def parse_date_parts(date_value) -> tuple[int | None, int | None, int | None]:
    """
    Parse year, month, and day from CSV date values like YYYY-MM or YYYY-MM-DD.
    """
    if date_value is None:
        return None, None, None

    parsed_date = pd.to_datetime(str(date_value), errors="coerce")
    if pd.isna(parsed_date):
        return None, None, None

    return int(parsed_date.year), int(parsed_date.month), int(parsed_date.day)


def read_csv_rows(filepath: str) -> list:
    # read CSV with pandas, clean column names
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
    Extract date/value series for a single field using the matrix conversion path.
    """
    matrix, dates, _ = measurements_to_matrix(
        measurements,
        fields=[field],
        drop_incomplete_rows=True,
    )

    if matrix.size == 0:
        return [], []

    return dates, matrix[:, 0]


def datetime_to_decimal_year(date) -> float:
    """
    Convert a datetime value to decimal year (e.g., 2024.5).
    """
    return date.year + (date.month - 1) / 12.0


def decimal_year_to_datetime(decimal_year: float) -> datetime.datetime:
    """
    Convert decimal-year values back to datetime for plotting.
    """
    year = int(np.floor(decimal_year))
    month_float = (decimal_year - year) * 12.0
    month = int(np.round(month_float)) + 1

    if month < 1:
        month = 1
    if month > 12:
        month = 12

    return datetime.datetime(year, month, 1)


def solve_least_squares_normal_equations(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Solve x = (A^T A)^-1 A^T b via a linear solve on normal equations.
    """
    ata = A.T @ A
    atb = A.T @ b
    return np.linalg.solve(ata, atb)


def fit_polynomial_temperature_trend(
    measurements: list, field: MeasurementField, degree: int
):
    """
    Build matrix A and vector b, then fit polynomial trend b ~= A x.
    degree=1 gives linear, degree=2 gives quadratic.
    """
    dates, values = field_series_from_matrix(measurements, field)
    if len(dates) < degree + 1:
        return None

    decimal_years = np.array([datetime_to_decimal_year(d) for d in dates], dtype=float)
    year_mean = float(decimal_years.mean())
    centered_years = decimal_years - year_mean

    columns = [np.ones(len(centered_years))]
    for p in range(1, degree + 1):
        columns.append(centered_years**p)
    A = np.column_stack(columns)

    b = np.array(values, dtype=float)

    x = solve_least_squares_normal_equations(A, b)

    residuals = b - (A @ x)
    rmse = float(np.sqrt(np.mean(residuals**2)))

    return {
        "A": A,
        "b": b,
        "x": x,
        "dates": dates,
        "decimal_years": decimal_years,
        "year_mean": year_mean,
        "field": field,
        "degree": degree,
        "rmse": rmse,
    }


def fit_linear_temperature_trend(measurements: list, field: MeasurementField):
    """
    Convenience wrapper for degree-1 (linear) trend fitting.
    """
    return fit_polynomial_temperature_trend(measurements, field, degree=1)


def predict_from_polynomial_model(model: dict, decimal_years: np.ndarray) -> np.ndarray:
    """
    Predict values from a fitted polynomial model for decimal-year inputs.
    """
    centered = decimal_years - model["year_mean"]
    columns = [np.ones(len(centered))]
    for p in range(1, model["degree"] + 1):
        columns.append(centered**p)
    A_pred = np.column_stack(columns)

    return A_pred @ model["x"]


def choose_best_model(linear_model: dict, polynomial_model: dict) -> str:
    """
    Choose model with lower RMSE.
    """
    if linear_model["rmse"] <= polynomial_model["rmse"]:
        return "linear"
    return "polynomial"


def fit_polynomial_xy(x_values: np.ndarray, y_values: np.ndarray, degree: int):
    """
    Fit polynomial least-squares model y ~= A x for generic numeric x/y data.
    """
    if len(x_values) < degree + 1:
        return None

    x_values = np.array(x_values, dtype=float)
    y_values = np.array(y_values, dtype=float)

    x_mean = float(np.mean(x_values))
    centered_x = x_values - x_mean

    columns = [np.ones(len(centered_x))]
    for p in range(1, degree + 1):
        columns.append(centered_x**p)
    A = np.column_stack(columns)

    coeffs = solve_least_squares_normal_equations(A, y_values)
    residuals = y_values - (A @ coeffs)
    rmse = float(np.sqrt(np.mean(residuals**2)))

    return {
        "A": A,
        "b": y_values,
        "x": coeffs,
        "degree": degree,
        "x_mean": x_mean,
        "rmse": rmse,
    }


def predict_from_xy_model(model: dict, x_values: np.ndarray) -> np.ndarray:
    """
    Predict values from a fitted generic polynomial model.
    """
    x_values = np.array(x_values, dtype=float)
    centered_x = x_values - model["x_mean"]

    columns = [np.ones(len(centered_x))]
    for p in range(1, model["degree"] + 1):
        columns.append(centered_x**p)
    A_pred = np.column_stack(columns)
    return A_pred @ model["x"]


def get_trend_mode_override() -> str:
    """
    Ask the user how to run the trend model.

    Returns one of: auto, linear, polynomial, both
    """
    raw = (
        input("Trend model mode [auto/linear/polynomial/both] " "(default auto): ")
        .strip()
        .lower()
    )

    aliases = {
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

    mode = aliases.get(raw)
    if mode is None:
        print(f"Unknown mode '{raw}', using auto.")
        return "auto"

    return mode


def get_aggregated_trend_mode_override() -> str:
    """
    Ask how to fit trend lines on aggregated bucket data.

    Returns one of: none, auto, linear, polynomial, both
    """
    raw = (
        input(
            "Aggregated trend mode [none/auto/linear/polynomial/both] "
            "(default auto): "
        )
        .strip()
        .lower()
    )

    aliases = {
        "": "auto",
        "n": "none",
        "none": "none",
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

    mode = aliases.get(raw)
    if mode is None:
        print(f"Unknown mode '{raw}', using auto.")
        return "auto"

    return mode


def get_field_override(
    default_field: MeasurementField = MeasurementField.TAVG,
) -> MeasurementField:
    """
    Ask which measurement field to analyze.
    """
    options = [field.column_name for field in graphable_fields()]
    print("Available fields:", ", ".join(options))
    raw = (
        input(f"Field to analyze (default {default_field.column_name}): ")
        .strip()
        .upper()
    )

    if raw == "":
        return default_field

    for field in graphable_fields():
        if raw == field.column_name:
            return field

    print(f"Unknown field '{raw}', using {default_field.column_name}.")
    return default_field


def get_time_bucket_override() -> str:
    """
    Ask how to compare values across time buckets.

    Returns one of: year, month, day, skip
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

    bucket = aliases.get(raw)
    if bucket is None:
        print(f"Unknown option '{raw}', skipping comparison plot.")
        return "skip"

    return bucket


def aggregate_field_by_time_bucket(
    measurements: list,
    field: MeasurementField,
    bucket: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Aggregate one field by year, month-of-year, or day-of-month.
    """
    dates, values = field_series_from_matrix(measurements, field)
    if len(dates) == 0:
        return np.array([]), np.array([])

    df = pd.DataFrame(
        {"date": pd.to_datetime(dates), "value": np.array(values, dtype=float)}
    )

    if bucket == "year":
        df["bucket"] = df["date"].dt.year
    elif bucket == "month":
        df["bucket"] = df["date"].dt.month
    elif bucket == "day":
        df["bucket"] = df["date"].dt.day
    else:
        return np.array([]), np.array([])

    grouped = df.groupby("bucket")["value"].mean().reset_index().sort_values("bucket")
    return grouped["bucket"].to_numpy(), grouped["value"].to_numpy()


def graph_field_by_time_bucket(
    measurements: list,
    field: MeasurementField,
    bucket: str,
    trend_mode: str = "none",
    output_dir: str = "graphs",
):
    """
    Create a comparison graph for one field aggregated by selected time bucket.
    """
    os.makedirs(output_dir, exist_ok=True)

    x_values, y_values = aggregate_field_by_time_bucket(measurements, field, bucket)
    if len(x_values) == 0:
        print(f"No valid data found for {field.column_name} by {bucket}.")
        return

    if len(x_values) == 1:
        print(
            f"Only one {bucket} value exists in the dataset for {field.column_name}; "
            "comparison may not be meaningful."
        )

    plt.figure(figsize=(12, 6))
    plt.plot(x_values, y_values, marker="o", label=f"Average {field.column_name}")

    trend_mode = trend_mode.lower().strip()
    valid_modes = {"none", "auto", "linear", "polynomial", "both"}
    if trend_mode not in valid_modes:
        print(f"Unknown aggregated trend mode '{trend_mode}', using none.")
        trend_mode = "none"

    if trend_mode != "none":
        linear_model = fit_polynomial_xy(x_values, y_values, degree=1)
        polynomial_model = fit_polynomial_xy(x_values, y_values, degree=2)

        if linear_model is None or polynomial_model is None:
            print(f"Not enough bucket points to fit polynomial trends for {bucket}.")
            selected_mode = "none"
        else:
            if trend_mode == "auto":
                selected_mode = choose_best_model(linear_model, polynomial_model)
            else:
                selected_mode = trend_mode

            x_grid = np.linspace(float(np.min(x_values)), float(np.max(x_values)), 200)

            if selected_mode in {"linear", "both"}:
                y_linear = predict_from_xy_model(linear_model, x_grid)
                plt.plot(
                    x_grid,
                    y_linear,
                    "-",
                    linewidth=2.0,
                    label=f"Linear fit (RMSE={linear_model['rmse']:.3f})",
                )

            if selected_mode in {"polynomial", "both"}:
                y_poly = predict_from_xy_model(polynomial_model, x_grid)
                plt.plot(
                    x_grid,
                    y_poly,
                    "-",
                    linewidth=2.0,
                    label=f"Polynomial fit (RMSE={polynomial_model['rmse']:.3f})",
                )

            print(
                f"Aggregated RMSE comparison for {field.column_name} by {bucket} -> "
                f"linear: {linear_model['rmse']:.4f}, "
                f"polynomial: {polynomial_model['rmse']:.4f}"
            )
            if trend_mode == "auto":
                print(f"Auto-selected aggregated mode: {selected_mode}")
            else:
                print(f"Aggregated trend mode applied: {selected_mode}")

    plt.xlabel(bucket.capitalize())
    plt.ylabel(f"Average {field.column_name}")
    plt.title(f"{field.column_name} aggregated by {bucket}")
    plt.grid(True)
    plt.xticks(x_values)
    plt.legend()
    plt.tight_layout()

    output_path = os.path.join(output_dir, f"{field.column_name}_by_{bucket}.png")
    plt.savefig(output_path)
    plt.close()

    print(f"Saved {bucket} comparison graph: {output_path}")


def graph_one_field(
    measurements: list, field: MeasurementField, output_dir: str = "graphs"
):
    """
    Graph one measurement field over time and save it as a PNG.

    Parameters:
        measurements (list): List of StationMeasurements objects.
        field (MeasurementField): The enum field to graph.
        output_dir (str): Folder where the graph image will be saved.
    """
    if not field.graphable:
        print(f"Skipping {field.column_name}: not graphable")
        return

    os.makedirs(output_dir, exist_ok=True)

    dates, values = field_series_from_matrix(measurements, field)

    if len(dates) == 0:
        print(f"No valid data found for {field.column_name}")
        return

    plot_dates = pd.to_datetime(dates)

    plt.figure(figsize=(12, 6))
    plt.plot(plot_dates, values, marker="o")
    plt.xlabel("Date")
    plt.ylabel(field.column_name)
    plt.title(f"{field.column_name} Over Time")
    plt.xticks(rotation=45)
    plt.tight_layout()

    output_path = os.path.join(output_dir, f"{field.column_name}.png")
    plt.savefig(output_path)
    plt.close()

    print(f"Saved graph: {output_path}")


def graph_temperature_trend(
    measurements: list,
    field: MeasurementField = MeasurementField.TAVG,
    future_months: int = 60,
    mode: str = "auto",
    output_dir: str = "graphs",
):
    """
    Fit least-squares trends and plot according to selected mode.

    Modes:
        auto: choose lower RMSE model
        linear: only linear model
        polynomial: only quadratic model
        both: overlay linear and quadratic
    """
    os.makedirs(output_dir, exist_ok=True)

    linear_model = fit_polynomial_temperature_trend(measurements, field, degree=1)
    polynomial_model = fit_polynomial_temperature_trend(measurements, field, degree=2)

    if linear_model is None or polynomial_model is None:
        print(f"Not enough valid data to model {field.column_name}")
        return

    mode = mode.lower().strip()
    valid_modes = {"auto", "linear", "polynomial", "both"}
    if mode not in valid_modes:
        print(f"Unknown mode '{mode}', using auto.")
        mode = "auto"

    recommended_mode = choose_best_model(linear_model, polynomial_model)
    if mode == "auto":
        selected_mode = recommended_mode
    else:
        selected_mode = mode

    decimal_years = linear_model["decimal_years"]
    linear_fit_values = predict_from_polynomial_model(linear_model, decimal_years)
    polynomial_fit_values = predict_from_polynomial_model(
        polynomial_model, decimal_years
    )

    future_years = future_months / 12.0
    x_min = float(decimal_years.min())
    x_max = float(decimal_years.max() + future_years)
    x_grid = np.linspace(x_min, x_max, 300)
    y_grid_linear = predict_from_polynomial_model(linear_model, x_grid)
    y_grid_polynomial = predict_from_polynomial_model(polynomial_model, x_grid)

    c0, c1 = linear_model["x"]
    warming_per_decade = c1 * 10.0
    p0, p1, p2 = polynomial_model["x"]

    def save_trend_plot(plot_mode: str, output_path: str):
        plt.figure(figsize=(12, 6))
        plt.plot(
            linear_model["dates"],
            linear_model["b"],
            "o",
            label=f"Observed {field.column_name}",
        )

        x_grid_dates = pd.to_datetime(
            [decimal_year_to_datetime(year) for year in x_grid]
        )

        if plot_mode in {"linear", "both"}:
            plt.plot(
                x_grid_dates,
                y_grid_linear,
                "-",
                linewidth=2.0,
                label=f"Linear fit (RMSE={linear_model['rmse']:.3f})",
            )
            plt.plot(
                linear_model["dates"],
                linear_fit_values,
                "--",
                linewidth=1.2,
                label="Linear on data range",
            )

        if plot_mode in {"polynomial", "both"}:
            plt.plot(
                x_grid_dates,
                y_grid_polynomial,
                "-",
                linewidth=2.0,
                label=f"Polynomial fit (RMSE={polynomial_model['rmse']:.3f})",
            )
            plt.plot(
                polynomial_model["dates"],
                polynomial_fit_values,
                ":",
                linewidth=1.6,
                label="Polynomial on data range",
            )

        plt.xlabel("Date")
        plt.ylabel(field.column_name)
        plt.title(
            f"{field.column_name} Trend via Least Squares | "
            f"linear slope={warming_per_decade:+.3f}/decade | mode={plot_mode}"
        )
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    output_path = os.path.join(
        output_dir, f"{field.column_name}_least_squares_trend.png"
    )

    if mode == "auto":
        save_trend_plot(selected_mode, output_path)
        both_output_path = os.path.join(
            output_dir, f"{field.column_name}_least_squares_trend_both.png"
        )
        save_trend_plot("both", both_output_path)
    else:
        save_trend_plot(selected_mode, output_path)

    print(
        f"Linear model for {field.column_name}: "
        f"y = {c0:.4f} + {c1:.4f}*(year - {linear_model['year_mean']:.2f})"
    )
    print(
        f"Polynomial model for {field.column_name}: "
        f"y = {p0:.4f} + {p1:.4f}*t + {p2:.6f}*t^2, "
        f"where t=(year - {polynomial_model['year_mean']:.2f})"
    )
    print(
        f"RMSE comparison -> linear: {linear_model['rmse']:.4f}, "
        f"polynomial: {polynomial_model['rmse']:.4f}"
    )
    print(f"Recommended by RMSE: {recommended_mode}")
    if mode != "auto":
        print(f"User override applied: {selected_mode}")
    else:
        print(f"Auto-selected mode: {selected_mode}")
    print(f"Saved trend graph: {output_path}")
    if mode == "auto":
        print(f"Saved comparison graph: {both_output_path}")


def is_dir_empty(dir: str) -> bool:
    if not os.path.exists(dir):
        return True

    return len(os.listdir(dir)) == 0


def graph_all_fields(measurements: list, output_dir: str = "graphs"):
    os.makedirs(output_dir, exist_ok=True)

    if not is_dir_empty(output_dir):
        answer = (
            input(f"'{output_dir}' is not empty. Delete all files in it? (y/n): ")
            .strip()
            .lower()
        )

        if answer != "y":
            print("Graph creation canceled.")
            return

        for filename in os.listdir(output_dir):
            file_path = os.path.join(output_dir, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)

        print(f"Deleted all files in '{output_dir}'.")

    for field in graphable_fields():
        graph_one_field(measurements, field, output_dir)


def print_matrix_preview(measurements: list):
    """
    Show a compact preview of matrix data for downstream math operations.
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


if __name__ == "__main__":
    filename = "weather.csv"

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
    if bucket_mode != "skip":
        aggregated_trend_mode = get_aggregated_trend_mode_override()
        graph_field_by_time_bucket(
            measurements,
            field=selected_field,
            bucket=bucket_mode,
            trend_mode=aggregated_trend_mode,
        )


# --- (~FUNCTIONAL~) THIS VISUAL CHART is just to very quickly look at some data ---
# from visual_chart import VisualChartApp
# if __name__ == "__main__":
#     filename = "weather.csv"
#     measurements = read_csv_rows(filename)
#     app = VisualChartApp(measurements)
#     app.run()
