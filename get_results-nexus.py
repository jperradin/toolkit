import numpy as np
import os
from natsort import natsorted


def transform_fraction_files(path):
    for subdir, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".dat") and file.startswith("fraction"):
                filename = os.path.join(subdir, file)
                print(filename)
                with open(filename, "r") as f:
                    header = []
                    for li, l in enumerate(f):
                        if l[0] == "#":
                            if li >= 1:
                                line = l.strip().split()
                                header.append(line[-1])
                            else:
                                continue
                        else:
                            break

                    data = np.loadtxt(f)

                    # old name is like "fraction-spanning_cluster_size-SiO6-SiO6-stishovite.dat"
                    # new name is like "SiO6-SiO6-stishovite_spanning_cluster_size--average.csv"
                    # fetch connectivity from name in header and write to new file
                    for head in header:
                        if head in filename:
                            connectivity = head
                            if "stishovite" in filename:
                                connectivity = head + "-stishovite"
                            print(connectivity)
                            break
                    property = file.split("-")[1]
                    new_filename = filename.replace(
                        file, connectivity + "_" + property + "--average.csv"
                    )
                    print(new_filename)
                    # # fetch errors from another file
                    # with open(os.path.join('../../sio2-96000at-1x/xrho/', new_filename), 'r') as f:
                    #     errors = np.loadtxt(f)
                    #     this_error = errors[:, 2]
                    #     # reverse if connectivity is OSi2-OSi2 or SiO4-SiO4
                    #     this_error = this_error[::-1]
                    #     if 'OSi2-OSi2' in connectivity or 'SiO4-SiO4' in connectivity:
                    with open(new_filename, "w") as f:
                        f.write("#Concentration	correlation_length	Error\n")
                        for i, d in enumerate(data):
                            f.write(
                                str(d[0]) + "\t" + str(d[1]) + "\t" + str(1.0) + "\n"
                            )

                    # write the error file with random values
                    with open(
                        new_filename.replace("--average.csv", "--errors.csv"), "w"
                    ) as f:
                        f.write("#Concentration	correlation_length	Error\n")
                        for i, d in enumerate(data):
                            dx = np.random.uniform(0.0, 0.01)
                            if d[1] > 0.0:
                                dy = np.random.uniform(0.0, 0.1)
                                de = np.random.uniform(0.0, 0.1)
                            else:
                                dy = 0.0
                                de = 0.0

                            f.write(f"{dx}\t{dy}\t{de}\n")


# rootdir = sys.argv[1]
rootdir = "./"

pattern = "dens"

unloading = False
# or unloading = False if loading material


class Printer:
    def __init__(self, data: list, names: list, output_path: str) -> None:
        # data should be like
        # data = [
        # [ (x1, y1), (x2, y2), ... ], # set 1
        # [ (x1, y1), (x2, y2), ... ],  # set 1
        # ]
        #
        self.data = data
        self.names = names
        self.variable = names[0]
        self.print_nxy(output_path)

    def set_data(self, this_set):
        self.x = []
        self.y = []
        self.dy = []
        for t in this_set:
            self.x.append(t[0])
            self.y.append(t[1])
            self.dy.append(t[2])

        self.x = np.array(self.x)
        self.y = np.array(self.y)
        self.dy = np.array(self.dy)

        if self.variable == "pressure":
            sorted_indices = np.argsort(self.x)
            self.x = self.x[sorted_indices]
            self.y = self.y[sorted_indices]
            self.dy = self.dy[sorted_indices]

        elif self.variable == "temperature":
            sorted_indices = np.argsort(self.x)
            self.x = self.x[sorted_indices]
            self.y = self.y[sorted_indices]
            self.dy = self.dy[sorted_indices]

    def print_nxy(self, output_path):
        data_to_export = np.zeros((len(self.data) + 1, len(self.data[0])))
        error_to_export = np.zeros((len(self.data) + 1, len(self.data[0])))
        if self.variable != "fraction":
            for c, s in enumerate(self.data):
                self.set_data(s)
                counter = 0
                for x, y, dy in zip(self.x, self.y, self.dy):
                    data_to_export[0][counter] = x
                    data_to_export[c + 1][counter] = y
                    error_to_export[0][counter] = x
                    error_to_export[c + 1][counter] = dy
                    counter += 1

            with open(output_path, "w") as f:
                # write the header of the file
                for i, n in enumerate(names):
                    f.write(f"# {i + 1} {n}\n")
                # write the data in the file
                for i in range(data_to_export.shape[1]):
                    for j in range(data_to_export.shape[0]):
                        f.write(f"{data_to_export[j][i]:^10.5f}\t")
                    f.write("\n")
            f.close()

            with open(output_path.replace(".dat", "--errors.dat"), "w") as f:
                # write the header of the file
                for i, n in enumerate(names):
                    f.write(f"# {i + 1} {n}\n")
                # write the data in the file
                for i in range(error_to_export.shape[1]):
                    for j in range(error_to_export.shape[0]):
                        f.write(f"{error_to_export[j][i]:^10.5f}\t")
                    f.write("\n")

        else:
            # print x y dy in the same file
            for c, s in enumerate(self.data):
                self.set_data(s)
                counter = 0
                for x, y, dy in zip(self.x, self.y, self.dy):
                    data_to_export[0][counter] = x
                    data_to_export[c + 1][counter] = y
                    error_to_export[0][counter] = x
                    error_to_export[c + 1][counter] = dy
                    counter += 1

            with open(output_path, "w") as f:
                # write the header of the file
                for i, n in enumerate(names):
                    f.write(f"# {i + 1} {n}\n")
                # write the data in the file
                for i in range(data_to_export.shape[1]):
                    for j in range(data_to_export.shape[0]):
                        f.write(f"{data_to_export[j][i]:^10.5f}\t")
                    f.write(f"{error_to_export[-1][i]:^10.5f}\t")
                    f.write("\n")

        print(f"file {output_path} printed !")


class Result:
    def __init__(self, dens: str, file: str, key: str) -> None:
        self.dens = dens
        self.file = file
        self.key = key
        self.fraction = 0
        self.result = 0
        self.error = 0
        self.box = 0
        self.pressure = 0
        self.temperature = 0

    def set_result(self, value):
        self.result = value

    def set_error(self, value):
        self.error = value

    def set_key(self, value):
        self.key = value

    def set_fraction(self, value):
        self.fraction = value

    def set_box(self, value):
        self.box = value

    def set_pressure(self, value):
        self.pressure = value

    def set_temperature(self, value):
        self.temperature = value

    def get_result(self):
        return self.result

    def get_file(self):
        return self.file

    def get_error(self):
        return self.error

    def get_dens(self):
        return self.dens

    def get_dens_value(self):
        return np.float64(self.dens.split("s")[1])

    def get_key(self):
        return self.key

    def get_box(self):
        return self.box

    def get_pressure(self):
        return self.pressure

    def get_temperature(self):
        return self.temperature

    def get_fraction(self):
        return self.fraction


class Results:
    def __init__(self) -> None:
        self.list = []

    def add_to_list(self, result: Result):
        self.list.append(result)

    def return_all_results(self, file):
        return [r for r in self.list if r.get_file() == file]

    def return_key_results(self, file, key, x):
        if x == "pressure":
            return [
                (r.get_pressure(), r.get_result(), r.get_error())
                for r in self.list
                if (r.get_file() == file and r.get_key() == key)
            ]
        elif x == "temperature":
            return [
                (r.get_temperature(), r.get_result(), r.get_error())
                for r in self.list
                if (r.get_file() == file and r.get_key() == key)
            ]
        elif x == "box":
            return [
                (r.get_box(), r.get_result(), r.get_error())
                for r in self.list
                if (r.get_file() == file and r.get_key() == key)
            ]
        elif x == "dens":
            return [
                (r.get_dens_value(), r.get_result(), r.get_error())
                for r in self.list
                if (r.get_file() == file and r.get_key() == key)
            ]
        elif x == "fraction":
            return [
                (r.get_fraction(), r.get_result(), r.get_error())
                for r in self.list
                if (r.get_file() == file and r.get_key() == key)
            ]

    def return_keys_of_file(self, file):
        keys = np.unique([r.get_key() for r in self.list if (r.get_file() == file)])
        # print(f"keys in {file} are : ")
        # print(keys)

        return keys

    def __str__(self) -> str:
        n_results = len(self.list)
        n_unique_files = len(np.unique([r.get_file() for r in self.list]))
        n_unique_keys = len(np.unique([r.get_key() for r in self.list]))
        to_return = f"{n_results} are stored for \n - {n_unique_files} different files \n - {n_unique_keys} different keys ... "
        return to_return


if __name__ == "__main__":
    files_to_look_for = [
        # GSPC files
        # "SiOz.dat",
        # "OSiz.dat",
        # "connectivity.dat",
        # "polyhedricity.dat",
        # "switch_probability.dat",
        # Nexus files
        "average_cluster_size.dat",
        "spanning_cluster_size.dat",
        "correlation_length.dat",
        "order_parameter.dat",
        "percolation_probability.dat",
        "biggest_cluster_size.dat",
    ]

    results = Results()

    dirs = natsorted(os.listdir(rootdir))

    pressures = {}
    boxes = {}
    temperatures = {}

    list_dens = []
    for i in dirs:
        if pattern in i:
            list_dens.append(i)
            pressures[i] = 0.0
            boxes[i] = 0.0
            temperatures[i] = 0.0

    with open("boxes", "r") as f:
        for li, l in enumerate(f):
            if unloading:
                # the first one is the last one
                boxes[list_dens[-li - 1]] = np.float64(l)
            else:
                boxes[list_dens[li]] = np.float64(l)
    hold = boxes
    f.close()

    with open("temperature", "r") as f:
        for li, l in enumerate(f):
            if unloading:
                temperatures[list_dens[-li - 1]] = np.float64(l)
            else:
                temperatures[list_dens[li]] = np.float64(l)
    f.close()

    with open("pressure", "r") as f:
        for li, l in enumerate(f):
            if unloading:
                pressures[list_dens[-li - 1]] = np.float64(l)
            else:
                pressures[list_dens[li]] = np.float64(l)
    f.close()

    for subdir in dirs:
        if pattern in subdir:
            files = os.listdir(subdir)
            print("subdir : ", subdir)
            for file in files:
                if file in files_to_look_for:
                    with open(os.path.join(subdir, file), "r") as f:
                        for li, l in enumerate(f):
                            if l[0] == "#":
                                continue
                            else:
                                # line should be like this for GSPC files
                                # result +/- error # key
                                #
                                # line should be like this for Nexus files
                                # concententration \u27c result +/- error # key
                                parts = l.split()

                                if len(parts) == 5:
                                    key = parts[-1]
                                    value = np.float64(parts[0])
                                    error = np.float64(parts[2])
                                    result = Result(dens=subdir, file=file, key=key)
                                    result.set_key(key)
                                    result.set_result(value)
                                    result.set_error(error)
                                    result.set_box(boxes[subdir])
                                    result.set_pressure(pressures[subdir])
                                    result.set_temperature(temperatures[subdir])
                                    results.add_to_list(result)
                                elif len(parts) == 7:
                                    key = parts[-1]
                                    value = np.float64(parts[2])
                                    fraction = np.float64(parts[0])
                                    error = np.float64(parts[4])
                                    result = Result(dens=subdir, file=file, key=key)
                                    result.set_key(key)
                                    result.set_fraction(fraction)
                                    result.set_result(value)
                                    result.set_error(error)
                                    result.set_box(boxes[subdir])
                                    result.set_pressure(pressures[subdir])
                                    result.set_temperature(temperatures[subdir])
                                    results.add_to_list(result)
                                elif len(parts) == 9:
                                    if parts[-1] == "1D":
                                        key = parts[6]
                                        value = np.float64(parts[2])
                                        fraction = np.float64(parts[0])
                                        error = np.float64(parts[4])
                                        result = Result(dens=subdir, file=file, key=key)
                                        result.set_key(key)
                                        result.set_fraction(fraction)
                                        result.set_result(value)
                                        result.set_error(error)
                                        result.set_box(boxes[subdir])
                                        result.set_pressure(pressures[subdir])
                                        result.set_temperature(temperatures[subdir])
                                        results.add_to_list(result)
                                    else:
                                        continue
                                else:
                                    continue

    print(results)
    if not os.path.exists("./export"):
        os.mkdir("./export")
        os.mkdir("./export/fractions")

    for file in files_to_look_for:
        x = "pressure"
        keys = results.return_keys_of_file(file)
        data = [results.return_key_results(file, key, x) for key in keys]
        names = [x]
        names.extend(keys)
        p = Printer(data, names, f"./export/{x}-{file}")
    for file in files_to_look_for:
        x = "dens"
        keys = results.return_keys_of_file(file)
        data = [results.return_key_results(file, key, x) for key in keys]
        names = [x]
        names.extend(keys)
        p = Printer(data, names, f"./export/{x}-{file}")
    for file in files_to_look_for:
        x = "box"
        keys = results.return_keys_of_file(file)
        data = [results.return_key_results(file, key, x) for key in keys]
        names = [x]
        names.extend(keys)
        p = Printer(data, names, f"./export/{x}-{file}")
    for file in files_to_look_for:
        x = "temperature"
        keys = results.return_keys_of_file(file)
        data = [results.return_key_results(file, key, x) for key in keys]
        names = [x]
        names.extend(keys)
        p = Printer(data, names, f"./export/{x}-{file}")
    for file in files_to_look_for:
        x = "fraction"
        keys = results.return_keys_of_file(file)
        data = [results.return_key_results(file, key, x) for key in keys]
        names = [x]
        names.extend(keys)
        for i, d in enumerate(data):
            this_name = [names[0], names[i + 1]]
            p = Printer(
                [d],
                this_name,
                f"./export/fractions/{x}-{file.split('.')[0]}-{names[i + 1]}.dat",
            )
    # transform_fraction_files('./export/')
    # remove ./export/fraction-* files
    # for subdir, dirs, files in os.walk("./export/"):
    #     for file in files:
    #         if file.startswith("fraction"):
    #             os.remove(os.path.join(subdir, file))
