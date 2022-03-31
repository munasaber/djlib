import numpy as np
import os
import pathlib
import math as m
from glob import glob
import json

libpath = pathlib.Path(__file__).parent.resolve()


def casm_query_reader(casm_query_json_path="pass", casm_query_json_data=None):
    """Reads keys and values from casm query json dictionary. 
    Parameters:
    -----------
    casm_query_json_path: str
        Absolute path to casm query json file.
        Defaults to 'pass' which means that the function will look to take a dictionary directly.
    casm_query_json_data: dict
        Can also directly take the casm query json dictionary.
        Default is None.
    
    Returns:
    results: dict
        Dictionary of all data grouped by keys (not grouped by configuraton)
    """
    if casm_query_json_data is None:
        with open(casm_query_json_path) as f:
            data = json.load(f)
    else:
        data = casm_query_json_data
    keys = data[0].keys()
    data_collect = []
    for i in range(len(keys)):
        data_collect.append([])

    for element_dict in data:
        for index, key in enumerate(keys):
            data_collect[index].append(element_dict[key])

    results = dict(zip(keys, data_collect))
    if "comp" in results.keys():
        comp = np.array(results["comp"])
        if len(comp.shape) > 2:
            results["comp"] = np.squeeze(comp).tolist()
    return results


def get_dj_dir():
    libpath = pathlib.Path(__file__).parent.resolve()
    return libpath


def column_sort(matrix, column_index):
    """Sorts a matrix by the values of a specific column. Far left column is column 0.
    Args:
        matrix(numpy_array): mxn numpy array.
        column_index(int): Index of the column to sort by.
    """
    column_index = int(column_index)
    sorted_matrix = matrix[np.argsort(matrix[:, column_index])]
    return sorted_matrix


def find(lst, a):
    """Finds the index of an element that matches a specified value.
    Args:
        a(float): The value to search for
    Returns:
        match_list[0](int): The index that a match occurrs, assuming there is only one match.
    """
    tolerance = 1e-14
    match_list = [
        i for i, x in enumerate(lst) if np.isclose([x], [a], rtol=tolerance)[0]
    ]
    if len(match_list) == 1:
        return match_list[0]
    elif len(match_list) > 1:
        print("Found more than one match. This is not expected.")
    elif len(match_list) == 0:
        print("Search value does not match any value in the provided list.")


def update_properties_files(casm_root_dir):
    """Updates json key of property.calc.json files to allow imports.

    Parameters
    ----------
    training_data_dir : str
        Path to a training_data directory in a casm project.

    Returns
    -------
    None.

    Notes
    -----
    Currently, only modifies "coord_mode" -> "coordinate_mode"
    """
    training_data_dir = os.path.join(casm_root_dir, "training_data")
    scels = glob(os.path.join(training_data_dir, "SCEL*"))
    for scel in scels:
        configs = glob(os.path.join(scel, "*"))

        for config in configs:
            properties_path = os.path.join(
                config, "calctype.default/properties.calc.json"
            )

            if os.path.isfile(properties_path):
                with open(properties_path) as f:
                    properties = json.load(f)
                if "coord_mode" in properties:
                    properties["coordinate_mode"] = properties["coord_mode"]
                if properties["atom_properties"]["force"]["value"] == []:
                    properties["atom_properties"]["force"]["value"] = [[0.0, 0.0, 0.0]]
                    print("Fixed empty forces in %s" % config)
                with open(
                    os.path.join(config, "calctype.default/properties.calc.json"), "w"
                ) as f:
                    json.dump(properties, f, indent="")
            else:
                print("Could not find %s" % properties_path)


def move_calctype_dirs(casm_root_dir):
    """Meant to fix casm import issue where calctype_default is copied within new calctype_default directory. Shifts all the data up one directory.

    Parameters
    ----------
    casm_root_dir : str
        Path to casm project root.

    Returns
    -------
    None.
    """
    scels = glob(os.path.join(casm_root_dir, "training_data/SCEL*"))
    for scel in scels:
        configs = glob(os.path.join(scel, "*"))

        for config in configs:

            if os.path.isdir(os.path.join(config, "calctype.default/calctype.default")):
                nested_calctype_data = os.path.join(
                    config, "calctype.default/calctype.default/*"
                )
                calctype_path = os.path.join(config, "calctype.default")
                os.system("mv %s %s" % (nested_calctype_data, calctype_path))
                os.system(
                    "rm -r %s"
                    % os.path.join(config, "calctype.default/calctype.default")
                )

