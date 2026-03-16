import numpy as np
import os
from natsort import natsorted


class DataPoint:
    """Represents a single data point parsed from a CSV export file."""

    def __init__(self, dens_dir, file_name, connectivity_type, concentration, value, std, error):
        self.dens_dir = dens_dir
        self.file_name = file_name
        self.connectivity_type = connectivity_type
        self.concentration = concentration
        self.value = value
        self.std = std
        self.error = error
        self.pressure = 0.0
        self.temperature = 0.0
        self.box_size = 0.0

    def set_conditions(self, pressure, temperature, box_size):
        self.pressure = pressure
        self.temperature = temperature
        self.box_size = box_size

    def get_density_value(self):
        """Extract numerical density from a directory name like 'dens2.240'."""
        return float(self.dens_dir.split("dens")[1])


class DataCollection:
    """Manages a collection of DataPoints and organizes them for export."""

    def __init__(self):
        self.data_points = []

    def add_data_point(self, data_point):
        self.data_points.append(data_point)

    def get_unique_files(self):
        return list(set(dp.file_name for dp in self.data_points))

    def get_connectivity_types_for_file(self, file_name):
        """Return sorted list of all connectivity types present in a given file."""
        types = set(
            dp.connectivity_type
            for dp in self.data_points
            if dp.file_name == file_name
        )
        return sorted(types)

    def get_data_by_variable(self, file_name, connectivity_type, variable):
        """
        Return list of (x, value, std, error) tuples for a given file,
        connectivity type, and x-axis variable, sorted by x.
        """
        relevant = [
            dp for dp in self.data_points
            if dp.file_name == file_name and dp.connectivity_type == connectivity_type
        ]

        x_map = {
            "pressure":     lambda dp: dp.pressure,
            "temperature":  lambda dp: dp.temperature,
            "density":      lambda dp: dp.get_density_value(),
            "box":          lambda dp: dp.box_size,
            "concentration": lambda dp: dp.concentration,
        }

        if variable not in x_map:
            raise ValueError(f"Unknown variable '{variable}'")

        data = [
            (x_map[variable](dp), dp.value, dp.std, dp.error)
            for dp in relevant
        ]
        data.sort(key=lambda row: row[0])
        return data


class DataExporter:
    """Exports organized data to .dat files with proper headers."""

    def __init__(self, data_collection, output_dir="./export"):
        self.data_collection = data_collection
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "concentrations"), exist_ok=True)

    def export_by_variable(self, variable_name):
        """Export data organized by a specific variable (density, pressure, etc.)."""
        for file_name in self.data_collection.get_unique_files():
            connectivity_types = self.data_collection.get_connectivity_types_for_file(file_name)
            if not connectivity_types:
                continue

            all_data = [
                self.data_collection.get_data_by_variable(file_name, ct, variable_name)
                for ct in connectivity_types
            ]

            if not all_data or not all_data[0]:
                continue

            base_path = os.path.join(self.output_dir, f"{variable_name}-{file_name}")

            self._write_data_file(base_path, variable_name, connectivity_types, all_data)
            self._write_error_file(
                base_path.replace(".dat", "--errors.dat"),
                variable_name, connectivity_types, all_data, error_type="error",
            )
            self._write_error_file(
                base_path.replace(".dat", "--std.dat"),
                variable_name, connectivity_types, all_data, error_type="std",
            )

    def _sorted_x_values(self, all_data):
        """Collect and sort all unique x-values across all connectivity-type series."""
        all_x = set()
        for series in all_data:
            for x, *_ in series:
                all_x.add(x)
        return sorted(all_x)

    def _build_matrix(self, sorted_x, all_data, value_index):
        """
        Build a (n_conn_types + 1) × n_x matrix.
        Row 0: x values.  Rows 1+: values at value_index from each series tuple.
        Missing entries are filled with 0.
        """
        n_cols = len(all_data) + 1
        matrix = np.zeros((n_cols, len(sorted_x)))
        x_to_idx = {x: i for i, x in enumerate(sorted_x)}

        for i, x in enumerate(sorted_x):
            matrix[0, i] = x

        for col, series in enumerate(all_data, start=1):
            lookup = {row[0]: row[value_index] for row in series}
            for i, x in enumerate(sorted_x):
                matrix[col, i] = lookup.get(x, 0.0)

        return matrix

    def _write_matrix(self, output_path, variable_name, connectivity_types, matrix, n_x):
        """Write a matrix to a .dat file with column-index headers."""
        with open(output_path, "w") as f:
            f.write(f"# 1 {variable_name}\n")
            for i, ct in enumerate(connectivity_types):
                f.write(f"# {i + 2} {ct}\n")
            for j in range(n_x):
                f.write("\t".join(f"{matrix[col, j]:^10.5f}" for col in range(matrix.shape[0])))
                f.write("\n")

    def _write_data_file(self, output_path, variable_name, connectivity_types, all_data):
        """Write the main data file (values only)."""
        sorted_x = self._sorted_x_values(all_data)
        if not sorted_x:
            print(f"Warning: no data to write for {output_path}")
            return
        matrix = self._build_matrix(sorted_x, all_data, value_index=1)
        self._write_matrix(output_path, variable_name, connectivity_types, matrix, len(sorted_x))
        print(f"Exported: {output_path}")

    def _write_error_file(self, output_path, variable_name, connectivity_types, all_data, error_type):
        """Write an error/std file. error_type must be 'error' or 'std'."""
        if error_type == "error":
            value_index = 3   # (x, value, std, error)
        elif error_type == "std":
            value_index = 2
        else:
            raise ValueError(f"Invalid error_type: '{error_type}'")

        sorted_x = self._sorted_x_values(all_data)
        if not sorted_x:
            return
        matrix = self._build_matrix(sorted_x, all_data, value_index=value_index)
        # Overwrite column 0 with the x values (same as data file)
        for i, x in enumerate(sorted_x):
            matrix[0, i] = x
        self._write_matrix(output_path, variable_name, connectivity_types, matrix, len(sorted_x))

    def export_concentration_files(self):
        """Export one file per (property, connectivity type) pair, sorted by concentration."""
        concentrations_dir = os.path.join(self.output_dir, "concentrations")

        for file_name in self.data_collection.get_unique_files():
            property_name = file_name.replace(".dat", "")
            connectivity_types = self.data_collection.get_connectivity_types_for_file(file_name)

            for ct in connectivity_types:
                data = self.data_collection.get_data_by_variable(file_name, ct, "concentration")
                if not data:
                    continue

                output_path = os.path.join(concentrations_dir, f"concentration-{property_name}-{ct}.dat")
                with open(output_path, "w") as f:
                    f.write("# 1 concentration\n# 2 value\n# 3 std\n# 4 error\n")
                    for concentration, value, std, error in data:
                        f.write(f"{concentration:^10.5f}\t{value:^10.5f}\t{std:^10.5f}\t{error:^10.5f}\n")

                print(f"Exported concentration file: {output_path}")


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_csv_file(filepath):
    """
    Parse a CSV export file.

    Two supported layouts:

    - Standard files (5 columns):
        connectivity_type, concentration, value, std, error

    - concentrations.dat (4 columns):
        connectivity_type, concentration, std, error
      Here the concentration itself is the measured value, so value = concentration.

    'nan' entries are converted to 0.0.
    """
    is_concentrations = os.path.basename(filepath) == "concentrations.dat"

    def parse_float(s):
        return 0.0 if s == "nan" else float(s)

    results = []
    try:
        with open(filepath, "r") as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if not line or line.startswith("#") or "," not in line:
                continue
            parts = [p.strip() for p in line.split(",")]

            if is_concentrations:
                # columns: connectivity_type, concentration, std, error
                if len(parts) < 4:
                    continue
                concentration = float(parts[1])
                results.append({
                    "connectivity_type": parts[0],
                    "concentration":     concentration,
                    "value":             concentration,   # concentration is the value
                    "std":               parse_float(parts[2]),
                    "error":             parse_float(parts[3]),
                })
            else:
                # columns: connectivity_type, concentration, value, std, error
                if len(parts) < 5:
                    continue
                results.append({
                    "connectivity_type": parts[0],
                    "concentration":     float(parts[1]),
                    "value":             float(parts[2]),
                    "std":               parse_float(parts[3]),
                    "error":             parse_float(parts[4]),
                })

    except Exception as e:
        print(f"Error parsing {filepath}: {e}")

    return results


# ---------------------------------------------------------------------------
# Condition loading
# ---------------------------------------------------------------------------

def load_condition_files(list_dens, unloading=False):
    """
    Load pressure, temperature, and box-size data from flat text files.
    Files that are absent are silently skipped.
    """
    conditions = {"pressures": {}, "temperatures": {}, "boxes": {}}

    for filename, key in [("boxes", "boxes"), ("temperature", "temperatures"), ("pressure", "pressures")]:
        if not os.path.exists(filename):
            print(f"Skipping {filename} — file not found")
            continue
        try:
            with open(filename, "r") as f:
                for li, line in enumerate(f):
                    if li >= len(list_dens):
                        break
                    idx = -li - 1 if unloading else li
                    conditions[key][list_dens[idx]] = float(line.strip())
            print(f"Loaded {filename}")
        except Exception as e:
            print(f"Error reading {filename}: {e}")

    return conditions


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    rootdir  = "./"
    pattern  = "dens"
    unloading = False

    target_files = [
        "average_cluster_size.dat",
        "spanning_cluster_size.dat",
        "correlation_length.dat",
        "order_parameter.dat",
        "percolation_probability.dat",
        "largest_cluster_size.dat",
        "concentrations.dat",
    ]

    list_dens = [d for d in natsorted(os.listdir(rootdir)) if pattern in d]
    if not list_dens:
        print(f"No directories matching pattern '{pattern}' found")
        return

    conditions = load_condition_files(list_dens, unloading)

    export_variables = ["density"]
    if conditions["pressures"]:
        export_variables.append("pressure")
    if conditions["temperatures"]:
        export_variables.append("temperature")
    if conditions["boxes"]:
        export_variables.append("box")

    print(f"Export variables: {export_variables}")

    data_collection = DataCollection()

    for dens_dir in list_dens:
        if not os.path.isdir(dens_dir):
            continue
        print(f"Processing: {dens_dir}")
        for filename in os.listdir(dens_dir):
            if filename not in target_files:
                continue
            parsed = parse_csv_file(os.path.join(dens_dir, filename))
            if not parsed:
                continue
            print(f"  {len(parsed)} data points in {filename}")
            for d in parsed:
                dp = DataPoint(
                    dens_dir=dens_dir,
                    file_name=filename,
                    connectivity_type=d["connectivity_type"],
                    concentration=d["concentration"],
                    value=d["value"],
                    std=d["std"],
                    error=d["error"],
                )
                dp.set_conditions(
                    pressure=conditions["pressures"].get(dens_dir, 0.0),
                    temperature=conditions["temperatures"].get(dens_dir, 0.0),
                    box_size=conditions["boxes"].get(dens_dir, 0.0),
                )
                data_collection.add_data_point(dp)

    print(f"\nTotal data points: {len(data_collection.data_points)}")
    print(f"Files found: {data_collection.get_unique_files()}")

    exporter = DataExporter(data_collection)

    for variable in export_variables:
        print(f"\nExporting by {variable}…")
        exporter.export_by_variable(variable)

    print("\nExporting individual concentration files…")
    exporter.export_concentration_files()

    print("\nExport complete!")


if __name__ == "__main__":
    main()
