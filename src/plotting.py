import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from station import MeasurementField, graphable_fields
from data_io import field_series_from_matrix
from modeling import (
    LINEAR_MODEL_DEGREE,
    POLYNOMIAL_MODEL_DEGREE,
    choose_best_model,
    decimal_year_to_datetime,
    fit_polynomial_temperature_trend,
    fit_polynomial_xy,
    predict_from_polynomial_model,
    predict_from_xy_model,
)


DEFAULT_OUTPUT_DIR = "graphs"
DEFAULT_FIGURE_SIZE = (12, 6)
AGGREGATED_GRID_POINTS = 200
TREND_GRID_POINTS = 300
TREND_MODES = {"auto", "linear", "polynomial", "both"}
AGGREGATED_TREND_MODES = {"none", "auto", "linear", "polynomial", "both"}


def ensure_output_dir(output_dir: str) -> None:
    """
    Make sure the output folder exists.
    """
    os.makedirs(output_dir, exist_ok=True)


def normalize_mode(mode: str, valid_modes: set[str], default_mode: str) -> str:
    """
    Clean a mode string and keep only allowed options.
    """
    normalized = mode.lower().strip()
    if normalized not in valid_modes:
        print(f"Unknown mode '{normalized}', using {default_mode}.")
        return default_mode
    return normalized


def aggregate_field_by_time_bucket(
    measurements: list,
    field: MeasurementField,
    bucket: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Group one field by year/month/day and take the average in each group.
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
    output_dir: str = DEFAULT_OUTPUT_DIR,
):
    """
    Plot one field after grouping it by year, month, or day.
    """
    ensure_output_dir(output_dir)

    x_values, y_values = aggregate_field_by_time_bucket(measurements, field, bucket)
    if len(x_values) == 0:
        print(f"No valid data found for {field.column_name} by {bucket}.")
        return

    if len(x_values) == 1:
        print(
            f"Only one {bucket} value exists in the dataset for {field.column_name}; "
            "comparison may not be meaningful."
        )

    plt.figure(figsize=DEFAULT_FIGURE_SIZE)
    plt.plot(x_values, y_values, marker="o", label=f"Average {field.column_name}")

    trend_mode = normalize_mode(
        mode=trend_mode,
        valid_modes=AGGREGATED_TREND_MODES,
        default_mode="none",
    )

    if trend_mode != "none":
        linear_model = fit_polynomial_xy(x_values, y_values, degree=LINEAR_MODEL_DEGREE)
        polynomial_model = fit_polynomial_xy(
            x_values, y_values, degree=POLYNOMIAL_MODEL_DEGREE
        )

        if linear_model is None or polynomial_model is None:
            print(f"Not enough bucket points to fit polynomial trends for {bucket}.")
            selected_mode = "none"
        else:
            if trend_mode == "auto":
                selected_mode = choose_best_model(linear_model, polynomial_model)
            else:
                selected_mode = trend_mode

            x_grid = np.linspace(
                float(np.min(x_values)),
                float(np.max(x_values)),
                AGGREGATED_GRID_POINTS,
            )

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
    measurements: list,
    field: MeasurementField,
    output_dir: str = DEFAULT_OUTPUT_DIR,
):
    """
    Plot one field over time and save as a PNG.
    """
    if not field.graphable:
        print(f"Skipping {field.column_name}: not graphable")
        return

    ensure_output_dir(output_dir)

    dates, values = field_series_from_matrix(measurements, field)
    if len(dates) == 0:
        print(f"No valid data found for {field.column_name}")
        return

    plot_dates = pd.to_datetime(dates)

    plt.figure(figsize=DEFAULT_FIGURE_SIZE)
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
    output_dir: str = DEFAULT_OUTPUT_DIR,
):
    """
    Fit least-squares trend lines and plot based on selected mode.
    """
    ensure_output_dir(output_dir)

    linear_model = fit_polynomial_temperature_trend(
        measurements,
        field,
        degree=LINEAR_MODEL_DEGREE,
    )
    polynomial_model = fit_polynomial_temperature_trend(
        measurements,
        field,
        degree=POLYNOMIAL_MODEL_DEGREE,
    )

    if linear_model is None or polynomial_model is None:
        print(f"Not enough valid data to model {field.column_name}")
        return

    mode = normalize_mode(mode=mode, valid_modes=TREND_MODES, default_mode="auto")

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
    x_grid = np.linspace(x_min, x_max, TREND_GRID_POINTS)
    y_grid_linear = predict_from_polynomial_model(linear_model, x_grid)
    y_grid_polynomial = predict_from_polynomial_model(polynomial_model, x_grid)

    c0, c1 = linear_model["x"]
    warming_per_decade = c1 * 10.0
    p0, p1, p2 = polynomial_model["x"]

    def save_trend_plot(plot_mode: str, output_path: str):
        plt.figure(figsize=DEFAULT_FIGURE_SIZE)
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


def is_dir_empty(directory: str) -> bool:
    """
    Return True if a directory does not exist or has no files.
    """
    if not os.path.exists(directory):
        return True

    return len(os.listdir(directory)) == 0


def clear_output_dir_with_confirmation(output_dir: str) -> bool:
    """
    Ask before deleting old graph files.
    """
    if is_dir_empty(output_dir):
        return True

    answer = (
        input(f"'{output_dir}' is not empty. Delete all files in it? (y/n): ")
        .strip()
        .lower()
    )
    if answer != "y":
        print("Graph creation canceled.")
        return False

    for filename in os.listdir(output_dir):
        file_path = os.path.join(output_dir, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

    print(f"Deleted all files in '{output_dir}'.")
    return True


def graph_all_fields(measurements: list, output_dir: str = DEFAULT_OUTPUT_DIR):
    """
    Save a time-series graph for each graphable field.
    """
    ensure_output_dir(output_dir)

    if not clear_output_dir_with_confirmation(output_dir):
        return

    for field in graphable_fields():
        graph_one_field(measurements, field, output_dir)
