import numpy as np
import os
import glob
from collections import defaultdict
import re


class SampleData:
    """Represents data from a single sample"""

    def __init__(self, sample_name, file_path):
        self.sample_name = sample_name
        self.file_path = file_path
        self.headers = []
        self.data = None
        self.variable_name = None
        self.connectivity_types = []

    def load_data(self):
        """Load data from file and parse headers"""
        try:
            with open(self.file_path, "r") as f:
                lines = f.readlines()

            # Parse headers
            self.headers = []
            data_start = 0
            for i, line in enumerate(lines):
                if line.startswith("#"):
                    # Parse header like "# 1 pressure" or "# 2 HD"
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        col_num = int(parts[1])
                        col_name = parts[2]
                        self.headers.append(col_name)
                        if col_num == 1:
                            self.variable_name = col_name
                        else:
                            self.connectivity_types.append(col_name)
                else:
                    data_start = i
                    break

            # Load numerical data
            data_lines = [line.strip() for line in lines[data_start:] if line.strip()]
            if data_lines:
                data_matrix = []
                for line in data_lines:
                    values = [float(x) for x in line.split()]
                    data_matrix.append(values)
                self.data = np.array(data_matrix)

            return True

        except Exception as e:
            print(f"Error loading {self.file_path}: {e}")
            return False

    def get_variable_values(self):
        """Get the x-axis values (pressure, density, etc.)"""
        if self.data is not None and len(self.data) > 0:
            return self.data[:, 0]
        return np.array([])

    def get_connectivity_data(self, connectivity_idx):
        """Get data for a specific connectivity type"""
        if self.data is not None and len(self.data) > 0:
            col_idx = connectivity_idx + 1  # +1 because first column is variable
            if col_idx < self.data.shape[1]:
                return self.data[:, col_idx]
        return np.array([])


class ConcentrationSampleData:
    """Represents concentration data from a single sample"""

    def __init__(self, sample_name, file_path):
        self.sample_name = sample_name
        self.file_path = file_path
        self.data = None

    def load_data(self):
        """Load concentration data (concentration, value, error)"""
        try:
            with open(self.file_path, "r") as f:
                lines = f.readlines()

            # Skip header lines
            data_lines = [
                line.strip()
                for line in lines
                if not line.startswith("#") and line.strip()
            ]

            if data_lines:
                data_matrix = []
                for line in data_lines:
                    values = [float(x) for x in line.split()]
                    if len(values) >= 3:  # concentration, value, error
                        data_matrix.append(values)
                self.data = np.array(data_matrix)

            return True

        except Exception as e:
            print(f"Error loading {self.file_path}: {e}")
            return False

    def get_concentration_values(self):
        """Get concentration values"""
        if self.data is not None:
            return self.data[:, 0]
        return np.array([])

    def get_data_values(self):
        """Get data values"""
        if self.data is not None:
            return self.data[:, 1]
        return np.array([])

    def get_error_values(self):
        """Get error values"""
        if self.data is not None:
            return self.data[:, 2]
        return np.array([])


class MultiSampleProcessor:
    """Main class for processing multiple samples"""

    def __init__(self, sample_dirs, output_dir="./averaged_export"):
        self.sample_dirs = sample_dirs
        self.output_dir = output_dir
        self.ensure_output_dirs()

    def ensure_output_dirs(self):
        """Create output directories"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        concentrations_dir = os.path.join(self.output_dir, "concentrations")
        if not os.path.exists(concentrations_dir):
            os.makedirs(concentrations_dir)

    def find_common_files(self):
        """Find files that exist in all sample directories"""
        all_files = {}

        for sample_dir in self.sample_dirs:
            export_dir = os.path.join(sample_dir, "export")
            if not os.path.exists(export_dir):
                print(f"Warning: {export_dir} not found")
                continue

            # Find regular export files (not in concentrations/)
            regular_files = glob.glob(os.path.join(export_dir, "*.dat"))
            regular_files = [
                os.path.basename(f)
                for f in regular_files
                if not f.endswith("--errors.dat")
            ]

            # Find concentration files
            concentrations_dir = os.path.join(export_dir, "concentrations")
            concentration_files = []
            if os.path.exists(concentrations_dir):
                concentration_files = glob.glob(
                    os.path.join(concentrations_dir, "*.dat")
                )
                concentration_files = [os.path.basename(f) for f in concentration_files]

            sample_name = os.path.basename(sample_dir.rstrip("/"))
            all_files[sample_name] = {
                "regular": regular_files,
                "concentration": concentration_files,
            }

        # Find common files across all samples
        if not all_files:
            return {}, {}

        sample_names = list(all_files.keys())
        common_regular = set(all_files[sample_names[0]]["regular"])
        common_concentration = set(all_files[sample_names[0]]["concentration"])

        for sample_name in sample_names[1:]:
            common_regular &= set(all_files[sample_name]["regular"])
            common_concentration &= set(all_files[sample_name]["concentration"])

        return sorted(list(common_regular)), sorted(list(common_concentration))

    def process_regular_files(self, common_files):
        """Process regular export files (pressure-, density-, etc.)"""
        print(f"\nProcessing {len(common_files)} regular files...")

        for filename in common_files:
            print(f"Processing: {filename}")

            # Collect data from all samples
            samples_data = []
            for sample_dir in self.sample_dirs:
                file_path = os.path.join(sample_dir, "export", filename)
                if os.path.exists(file_path):
                    sample_name = os.path.basename(sample_dir.rstrip("/"))
                    sample_data = SampleData(sample_name, file_path)
                    if sample_data.load_data():
                        samples_data.append(sample_data)

            if not samples_data:
                print(f"  No valid data found for {filename}")
                continue

            # Process the data
            self._average_regular_file(filename, samples_data)

    def _average_regular_file(self, filename, samples_data):
        """Average data from multiple samples for a regular file - averaging both x and y"""

        # Get reference sample for structure
        ref_sample = samples_data[0]
        variable_name = ref_sample.variable_name
        connectivity_types = ref_sample.connectivity_types

        # Find the maximum number of data points across all samples
        max_points = max(len(sample.get_variable_values()) for sample in samples_data)

        if max_points == 0:
            print(f"  No data points found for {filename}")
            return

        # Create output matrices
        n_connectivities = len(connectivity_types)

        averaged_data = np.zeros(
            (max_points, n_connectivities + 1)
        )  # +1 for x variable
        std_data = np.zeros((max_points, n_connectivities + 1))

        # For each data point index, collect values from all samples
        for point_idx in range(max_points):
            # Collect x-values at this point index from all samples
            x_values_at_point = []
            for sample in samples_data:
                x_values = sample.get_variable_values()
                if point_idx < len(x_values):
                    x_values_at_point.append(x_values[point_idx])

            # Average x-values
            if x_values_at_point:
                averaged_data[point_idx, 0] = np.mean(x_values_at_point)
                std_data[point_idx, 0] = (
                    np.std(x_values_at_point, ddof=1)
                    if len(x_values_at_point) > 1
                    else 0.0
                )

            # Process each connectivity type
            for conn_idx, connectivity in enumerate(connectivity_types):
                # Collect y-values at this point index from all samples
                y_values_at_point = []
                for sample in samples_data:
                    conn_data = sample.get_connectivity_data(conn_idx)
                    if point_idx < len(conn_data):
                        y_values_at_point.append(conn_data[point_idx])

                # Average y-values
                if y_values_at_point:
                    averaged_data[point_idx, conn_idx + 1] = np.mean(y_values_at_point)
                    std_data[point_idx, conn_idx + 1] = (
                        np.std(y_values_at_point, ddof=1)
                        if len(y_values_at_point) > 1
                        else 0.0
                    )
                else:
                    averaged_data[point_idx, conn_idx + 1] = 0.0
                    std_data[point_idx, conn_idx + 1] = 0.0

        # Write output files
        self._write_averaged_file(
            filename, variable_name, connectivity_types, averaged_data, "average"
        )
        self._write_averaged_file(
            filename.replace(".dat", "--std.dat"),
            variable_name,
            connectivity_types,
            std_data,
            "std",
        )

    def _write_averaged_file(
        self, filename, variable_name, connectivity_types, data_matrix, data_type
    ):
        """Write averaged or std data file"""

        output_path = os.path.join(self.output_dir, filename)

        with open(output_path, "w") as f:
            # Write headers
            f.write(f"# Multi-sample {data_type} data\n")
            f.write(f"# 1 {variable_name}\n")
            for i, conn_type in enumerate(connectivity_types):
                f.write(f"# {i + 2} {conn_type}\n")

            # Write data
            for row in data_matrix:
                for val in row:
                    f.write(f"{val:^12.6f}\t")
                f.write("\n")

        print(f"  Exported: {output_path}")

    def process_concentration_files(self, common_files):
        """Process concentration files from concentrations/ directory"""
        print(f"\nProcessing {len(common_files)} concentration files...")

        for filename in common_files:
            print(f"Processing: {filename}")

            # Collect data from all samples
            samples_data = []
            for sample_dir in self.sample_dirs:
                file_path = os.path.join(
                    sample_dir, "export", "concentrations", filename
                )
                if os.path.exists(file_path):
                    sample_name = os.path.basename(sample_dir.rstrip("/"))
                    sample_data = ConcentrationSampleData(sample_name, file_path)
                    if sample_data.load_data():
                        samples_data.append(sample_data)

            if not samples_data:
                print(f"  No valid data found for {filename}")
                continue

            # Process the concentration data
            self._average_concentration_file(filename, samples_data)

    def _average_concentration_file(self, filename, samples_data):
        """Average concentration data from multiple samples - averaging both x and y"""

        # Find the maximum number of data points across all samples
        max_points = max(
            len(sample.get_concentration_values()) for sample in samples_data
        )

        if max_points == 0:
            print(f"  No concentration data found for {filename}")
            return

        # Create output arrays
        averaged_data = np.zeros((max_points, 3))  # concentration, averaged_value, std

        # For each data point index, collect values from all samples
        for point_idx in range(max_points):
            # Collect concentration values (x) at this point index
            conc_values_at_point = []
            data_values_at_point = []

            for sample in samples_data:
                concentrations = sample.get_concentration_values()
                data_values = sample.get_data_values()

                if point_idx < len(concentrations) and point_idx < len(data_values):
                    conc_values_at_point.append(concentrations[point_idx])
                    data_values_at_point.append(data_values[point_idx])

            # Average both concentration (x) and data (y) values
            if conc_values_at_point and data_values_at_point:
                # Average x-values (concentrations)
                averaged_data[point_idx, 0] = np.mean(conc_values_at_point)

                # Average y-values (data values)
                data_array = np.array(data_values_at_point)
                averaged_data[point_idx, 1] = np.mean(data_array)
                averaged_data[point_idx, 2] = (
                    np.std(data_array, ddof=1) if len(data_array) > 1 else 0.0
                )
            else:
                averaged_data[point_idx, 0] = 0.0
                averaged_data[point_idx, 1] = 0.0
                averaged_data[point_idx, 2] = 0.0

        # Write output file
        output_path = os.path.join(
            self.output_dir,
            "concentrations",
            filename.replace(".dat", "--averaged.dat"),
        )

        with open(output_path, "w") as f:
            f.write("# Multi-sample averaged concentration data\n")
            f.write("# 1 concentration_averaged\n")
            f.write("# 2 data_averaged\n")
            f.write("# 3 data_std_deviation\n")

            for row in averaged_data:
                f.write(f"{row[0]:^12.6f}\t{row[1]:^12.6f}\t{row[2]:^12.6f}\n")

        print(f"  Exported: {output_path}")


def main():
    """Main function"""

    # Configuration
    base_dir = "./"  # Current directory
    sample_patterns = ["4a", "4b", "4c", "4d", "4e"]  # Sample directory names

    # Find sample directories
    sample_dirs = []
    for pattern in sample_patterns:
        sample_path = os.path.join(base_dir, pattern)
        if os.path.exists(sample_path):
            sample_dirs.append(sample_path)
        else:
            print(f"Warning: Sample directory {sample_path} not found")

    if not sample_dirs:
        print("Error: No sample directories found!")
        return

    print(
        f"Found {len(sample_dirs)} sample directories: {[os.path.basename(d) for d in sample_dirs]}"
    )

    # Initialize processor
    processor = MultiSampleProcessor(sample_dirs)

    # Find common files across all samples
    common_regular, common_concentration = processor.find_common_files()

    print(f"Found {len(common_regular)} common regular files")
    print(f"Found {len(common_concentration)} common concentration files")

    if not common_regular and not common_concentration:
        print("No common files found across all samples!")
        return

    # Process files
    if common_regular:
        processor.process_regular_files(common_regular)

    if common_concentration:
        processor.process_concentration_files(common_concentration)

    print(f"\nProcessing complete! Results saved to: {processor.output_dir}")
    print(f"- Regular averaged files: {len(common_regular)}")
    print(f"- Standard deviation files: {len(common_regular)}")
    print(f"- Concentration averaged files: {len(common_concentration)}")


if __name__ == "__main__":
    main()
