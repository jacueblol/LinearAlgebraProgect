import os
import pandas as pd
import matplotlib.pyplot as plt
from station import StationInfo, StationMeasurements, MeasurementField


def clean_value(value):
    """
    Convert pandas missing values (NaN) into None.
    """
    if pd.isna(value):
        return None
    return value


def read_csv_rows(filepath: str) -> list:
    csv = pd.read_csv(filepath)
    measurements_list = []

    for _, row in csv.iterrows():
        date_value = clean_value(row.get("DATE"))

        year = None
        month = None
        if date_value is not None and "-" in str(date_value):
            year_str, month_str = str(date_value).split("-")
            year = int(year_str)
            month = int(month_str)

        station_info = StationInfo(
            name=clean_value(row.get("NAME")),
            year=year,
            month=month,
            longitude=clean_value(row.get("LONGITUDE")),
            latitude=clean_value(row.get("LATITUDE")),
            elevation=clean_value(row.get("ELEVATION"))
        )

        values = {}
        for field in MeasurementField:
            values[field.column_name] = clean_value(row.get(field.column_name))

        measurement = StationMeasurements(
            station_info=station_info,
            values=values
        )

        measurements_list.append(measurement)

    return measurements_list


def graph_one_field(measurements: list, field: MeasurementField, output_dir: str = "graphs"):
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

    dates = []
    values = []

    for measurement in measurements:
        date = measurement.station_info.date
        value = measurement.get(field)

        if date is not None and value is not None:
            dates.append(date)
            values.append(value)

    if len(dates) == 0:
        print(f"No valid data found for {field.column_name}")
        return

    plt.figure(figsize=(12, 6))
    plt.plot(dates, values, marker="o")
    plt.xlabel("Date")
    plt.ylabel(field.column_name)
    plt.title(f"{field.column_name} Over Time")
    plt.xticks(rotation=45)
    plt.tight_layout()

    output_path = os.path.join(output_dir, f"{field.column_name}.png")
    plt.savefig(output_path)
    plt.close()

    print(f"Saved graph: {output_path}")

def is_dir_empty(dir: str) -> bool:
    if not os.path.exists(dir):
        return True

    return len(os.listdir(dir)) == 0

def graph_all_fields(measurements: list, output_dir: str = "graphs"):
    os.makedirs(output_dir, exist_ok=True)

    if not is_dir_empty(output_dir):
        answer = input(f"'{output_dir}' is not empty. Delete all files in it? (y/n): ").strip().lower()

        if answer != "y":
            print("Graph creation canceled.")
            return

        for filename in os.listdir(output_dir):
            file_path = os.path.join(output_dir, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)

        print(f"Deleted all files in '{output_dir}'.")

    for field in MeasurementField:
        if field.graphable:
            graph_one_field(measurements, field, output_dir)

if __name__ == "__main__":
    filename = "weather.csv"

    measurements = read_csv_rows(filename)
    graph_all_fields(measurements)



# --- (~FUNCTIONAL~) THIS VISUAL CHART is just to very quickly look at some data ---
# from visual_chart import VisualChartApp
# if __name__ == "__main__":
#     filename = "weather.csv"
#     measurements = read_csv_rows(filename)
#     app = VisualChartApp(measurements)
#     app.run()