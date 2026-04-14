import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from station import MeasurementField


class VisualChartApp:
    def __init__(self, measurements: list):
        self.measurements = measurements

        self.root = tk.Tk()
        self.root.title("Weather Visual Chart")
        self.root.geometry("1200x800")

        self.field_vars = {}
        self.lines = {}

        self.figure = Figure(figsize=(10, 6), dpi=100)
        self.ax = self.figure.add_subplot(111)

        self.canvas = None
        self.canvas_widget = None

        self._build_ui()
        self._draw_initial_plot()
        self.canvas.draw()

    def _build_ui(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill="both", expand=True)

        control_frame = ttk.Frame(main_frame, padding=10)
        control_frame.pack(side="left", fill="y")

        chart_frame = ttk.Frame(main_frame)
        chart_frame.pack(side="right", fill="both", expand=True)

        title_label = ttk.Label(control_frame, text="Fields", font=("Arial", 12, "bold"))
        title_label.pack(anchor="w", pady=(0, 10))

        for field in MeasurementField:
            if field.graphable:
                var = tk.BooleanVar(value=False)
                self.field_vars[field] = var

                checkbox = ttk.Checkbutton(
                    control_frame,
                    text=field.column_name,
                    variable=var,
                    command=self.update_plot
                )
                checkbox.pack(anchor="w")

        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill="x", pady=(15, 0))

        select_all_button = ttk.Button(button_frame, text="Select All", command=self.select_all)
        select_all_button.pack(fill="x", pady=2)

        clear_all_button = ttk.Button(button_frame, text="Clear All", command=self.clear_all)
        clear_all_button.pack(fill="x", pady=2)

        reset_view_button = ttk.Button(button_frame, text="Reset View", command=self.reset_view)
        reset_view_button.pack(fill="x", pady=2)

        self.canvas = FigureCanvasTkAgg(self.figure, master=chart_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side="top", fill="both", expand=True)

        toolbar = NavigationToolbar2Tk(self.canvas, chart_frame, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side="bottom", fill="x")

    def _get_field_data(self, field: MeasurementField):
        dates = []
        values = []

        for measurement in self.measurements:
            date = measurement.station_info.date
            value = measurement.get(field)

            if date is not None and value is not None:
                dates.append(date)
                values.append(value)

        return dates, values

    def _draw_initial_plot(self):
        first_field = None

        for field in MeasurementField:
            if field.graphable:
                first_field = field
                break

        if first_field is not None:
            self.field_vars[first_field].set(True)

        self.update_plot()

    def update_plot(self):
        current_xlim = self.ax.get_xlim()
        current_ylim = self.ax.get_ylim()
        had_lines = len(self.ax.lines) > 0

        self.ax.clear()
        self.lines.clear()

        selected_fields = [field for field, var in self.field_vars.items() if var.get()]

        if not selected_fields:
            self.ax.set_title("No fields selected")
            self.ax.set_xlabel("Date")
            self.ax.set_ylabel("Value")
            self.canvas.draw()
            return

        for field in selected_fields:
            dates, values = self._get_field_data(field)

            if len(dates) == 0:
                continue

            line, = self.ax.plot(dates, values, marker="o", label=field.column_name)
            self.lines[field] = line

        self.ax.set_title("Weather Measurements Over Time")
        self.ax.set_xlabel("Date")
        self.ax.set_ylabel("Value")
        self.ax.legend()
        self.ax.grid(True)
        self.ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        self.figure.autofmt_xdate()

        if had_lines:
            self.ax.set_xlim(current_xlim)
            self.ax.set_ylim(current_ylim)

        self.canvas.draw()

    def select_all(self):
        for var in self.field_vars.values():
            var.set(True)
        self.update_plot()

    def clear_all(self):
        for var in self.field_vars.values():
            var.set(False)
        self.update_plot()

    def reset_view(self):
        self.update_plot()
        self.ax.relim()
        self.ax.autoscale_view()
        self.canvas.draw()

    def run(self):
        self.root.mainloop()