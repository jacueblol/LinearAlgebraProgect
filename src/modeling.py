import datetime

import numpy as np

from station import MeasurementField
from data_io import field_series_from_matrix


LINEAR_MODEL_DEGREE = 1
POLYNOMIAL_MODEL_DEGREE = 2


def build_polynomial_design_matrix(
    centered_values: np.ndarray, degree: int
) -> np.ndarray:
    """
    Build the matrix [1, x, x^2, ...] from centered x-values.
    """
    columns = [np.ones(len(centered_values))]
    for power in range(1, degree + 1):
        columns.append(centered_values**power)
    return np.column_stack(columns)


def datetime_to_decimal_year(date) -> float:
    """
    Convert a date to decimal year so it can be used as x in a model.
    """
    return date.year + (date.month - 1) / 12.0


def decimal_year_to_datetime(decimal_year: float) -> datetime.datetime:
    """
    Convert decimal year back to a date for plotting.
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
    Solve least squares with normal equations: (A^T A)x = A^T b.
    """
    # @ means matrix multiply.
    ata = A.T @ A
    # Multiply A^T by b to build the right-hand side.
    atb = A.T @ b
    return np.linalg.solve(ata, atb)


def fit_polynomial_temperature_trend(
    measurements: list, field: MeasurementField, degree: int
):
    """
    Fit a polynomial trend to one field using least squares.
    Degree 1 is linear, degree 2 is quadratic.
    """
    dates, values = field_series_from_matrix(measurements, field)
    if len(dates) < degree + 1:
        return None

    decimal_years = np.array([datetime_to_decimal_year(d) for d in dates], dtype=float)
    year_mean = float(decimal_years.mean())
    centered_years = decimal_years - year_mean
    A = build_polynomial_design_matrix(centered_years, degree)

    b = np.array(values, dtype=float)
    x = solve_least_squares_normal_equations(A, b)

    # A @ x is the model prediction.
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
    Shortcut for linear (degree-1) trend fitting.
    """
    return fit_polynomial_temperature_trend(measurements, field, degree=1)


def predict_from_polynomial_model(model: dict, decimal_years: np.ndarray) -> np.ndarray:
    """
    Compute predicted y-values from a fitted polynomial model.
    """
    centered = decimal_years - model["year_mean"]
    A_pred = build_polynomial_design_matrix(centered, model["degree"])
    # Multiply design matrix by coefficients to get predictions.
    return A_pred @ model["x"]


def choose_best_model(linear_model: dict, polynomial_model: dict) -> str:
    """
    Choose the model with smaller RMSE.
    """
    if linear_model["rmse"] <= polynomial_model["rmse"]:
        return "linear"
    return "polynomial"


def fit_polynomial_xy(x_values: np.ndarray, y_values: np.ndarray, degree: int):
    """
    Fit a polynomial least-squares model for generic x/y data.
    """
    if len(x_values) < degree + 1:
        return None

    x_values = np.array(x_values, dtype=float)
    y_values = np.array(y_values, dtype=float)

    x_mean = float(np.mean(x_values))
    centered_x = x_values - x_mean
    A = build_polynomial_design_matrix(centered_x, degree)

    coeffs = solve_least_squares_normal_equations(A, y_values)
    # A @ coeffs is the fitted curve values.
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
    Compute predicted y-values for new x-values.
    """
    x_values = np.array(x_values, dtype=float)
    centered_x = x_values - model["x_mean"]

    A_pred = build_polynomial_design_matrix(centered_x, model["degree"])
    # Multiply design matrix by coefficients to get predictions.
    return A_pred @ model["x"]
