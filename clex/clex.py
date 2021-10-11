import json
import os
from cupy._indexing.generate import ravel_multi_index
import numpy as np
from scipy.sparse.construct import rand
from scipy.spatial import ConvexHull
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error
import csv
from glob import glob
from tqdm import tqdm
import pickle
import cupy as cp


def read_comp_and_energy_points(datafile):
    """Generates points in composition and energy space for use in convex hull algorithms.

    Parameters
    ----------
    datafile : str
        path to the casm query output json file.

    Returns
    -------
    points: ndarray,  shape(number_configurations, number_comp_axes+1)
        Points in composition-energy space.
    """
    with open(datafile) as f:
        data = json.load(f)
    points = [
        [x[0] for x in entry["comp"]] + [entry["formation_energy"]] for entry in data
    ]
    points = np.array(points)
    return points


def read_corr_comp_formation(datafile):
    """
    read_corr_and_formation_energy(datafile)

    Reads and returns data from json containing correlation functions and formation energies.
    Args:
        datafile(str): Path to the json file containing the correlation functions and formation energies.
    Returns:
        tuple(
            corr,                   (ndarray): Correlation functions: nxm matrix of correlation funcitons: each row corresponds to a configuration.
            formation_energy,       (ndarray): Formation energies: vecrtor of n elements: one for each configuration.
            scel_names              (ndarray): The name for a given configuration. Vector of n elements.
        )
    """
    with open(datafile) as f:
        data = json.load(f)

    corr = []
    formation_energy = []
    scel_names = []
    comp = []
    # TODO: add flexibility for absence of some keys in the json file
    for entry in data:
        if "corr" in entry.keys():
            corr.append(np.array(entry["corr"]).flatten())
        if "formation_energy" in entry.keys():
            formation_energy.append(entry["formation_energy"])
        if "name" in entry.keys():
            scel_names.append(entry["name"])
        if "comp" in entry.keys():
            comp.append(entry["comp"][0])  # Assumes a binary
    corr = np.array(corr)
    formation_energy = np.array(formation_energy)
    scel_names = np.array(scel_names)
    comp = np.array(comp)
    results = {
        "corr": corr,
        "formation_energy": formation_energy,
        "names": scel_names,
        "comp": comp,
    }
    return results


def lower_hull(hull, energy_index=-2):
    """Returns the lower convex hull (with respect to energy direction) given  complete convex hull.
    Parameters
    ----------
        hull : scipy.spatial.ConvexHull
            Complete convex hull object.
        energy_index : int
            index of energy dimension of points within 'hull'.
    Returns
    -------
        lower_hull_simplices : np.ndarray of ints, shape (nfacets, ndim)
            indices of the facets forming the lower convex hull.

        lower_hull_vertices : np.ndarray of ints, shape (nvertices,)
            Indices of the vertices forming the vertices of the lower convex hull.
    """
    # Note: energy_index is used on the "hull.equations" output, which has a scalar offset.
    # (If your energy is the last column of "points", either use the direct index, or use "-2". Using "-1" as energy_index will not give the proper lower hull.)
    lower_hull_simplices = hull.simplices[hull.equations[:, energy_index] < 0]
    lower_hull_vertices = np.unique(np.ravel(lower_hull_simplices))
    return (lower_hull_simplices, lower_hull_vertices)


def checkhull(hull_comps, hull_energies, test_comp, test_energy):
    """Find if specified coordinates are above, on or below the specified lower convex hull.
    Args:
        hull_vertex(ndarray): 2D array, shape nxm where n = # of points, m = # of composition dimensions + 1 energy as the last column.
        test_coords(ndarray): 2D array, shape lxm where l = # of points to test, m = # of composition dimensions + 1 energy as the last column.
    Returns:
        tuple(
            above_hull(ndarray): 2D array, shape p x m where m = # of composition dimensions + 1 energy as the last column.
            on_hull(ndarray): 2D array, shape q x m where m = # of composition dimensions + 1 energy as the last column.
            below_hull(ndarray): 2D array, shape r x m where m = # of composition dimensions + 1 energy as the last column.
        )
    """
    # Need to reshape to column vector to work properly.
    # Test comp should also be a column vector.
    test_energy = np.reshape(test_energy, (-1, 1))
    # Fit linear grid
    interp_hull = griddata(hull_comps, hull_energies, test_comp, method="linear")

    # Check if the test_energy points are above or below the hull
    hull_dist = test_energy - interp_hull
    return np.ravel(np.array(hull_dist))


def checkhull_gpu(hull_comps, hull_energies, test_comp, test_energy):
    """Find if specified coordinates are above, on or below the specified lower convex hull. (GPU cupy version)
    Args:
        hull_vertex(ndarray): 2D array, shape nxm where n = # of points, m = # of composition dimensions + 1 energy as the last column.
        test_coords(ndarray): 2D array, shape lxm where l = # of points to test, m = # of composition dimensions + 1 energy as the last column.
    Returns:
        tuple(
            above_hull(ndarray): 2D array, shape p x m where m = # of composition dimensions + 1 energy as the last column.
            on_hull(ndarray): 2D array, shape q x m where m = # of composition dimensions + 1 energy as the last column.
            below_hull(ndarray): 2D array, shape r x m where m = # of composition dimensions + 1 energy as the last column.
        )
    """
    # Need to reshape to column vector to work properly.
    # Test comp should also be a column vector.
    test_energy = cp.reshape(test_energy, (-1, 1))
    # Fit linear grid
    interp_hull = griddata(hull_comps, hull_energies, test_comp, method="linear")

    # Check if the test_energy points are above or below the hull
    hull_dist = test_energy - interp_hull
    return cp.ravel(cp.array(hull_dist))


def run_lassocv(corr, formation_energy):
    reg = LassoCV(fit_intercept=False, n_jobs=4).fit(corr, formation_energy)
    eci = reg.coef_
    return eci


def generate_rand_eci_vec(num_eci: int, stdev: float, normalization: float):
    """Generates a random, normalized vector in eci space. Each element is drawn from a standard normal distribution.

    Parameters
    ----------
    num_eci : int
        The number of ECI.
    stdev : float
        Standard deviation defining the standard normal distribution for each element.
    normalization : float
        Magnitude to scale the random vector by.

    Returns
    -------
    eci_vec : np.ndarray
        Random, scaled vector in ECI space.
    """
    eci_vec = np.random.normal(scale=stdev, size=num_eci)
    eci_vec = (eci_vec / np.linalg.norm(eci_vec)) * normalization
    return eci_vec


def metropolis_hastings_ratio(
    current_eci: np.ndarray,
    proposed_eci: np.ndarray,
    current_energy: np.ndarray,
    proposed_energy: np.ndarray,
    formation_energy: np.ndarray,
):
    """Acceptance probability ratio defined in Zabaras et. al, https://doi.org/10.1016/j.cpc.2014.07.013. First part of equation (12)
    Parameters
    ----------
    current_eci : np.ndarray shape(number_eci)
        Vector of current ECI values.
    proposed_eci : np.ndarray shape(number_eci)
        Vector of proposed eci values, differing from current_eci by a random vector in ECI space.
    current_energy : np.ndarray, shape(number_dft_computed_configs_)
        Energy calculated with current_eci.
    proposed_energy : np.ndarray, shape(number_dft_computed_configs_)
        Energy calculated using proposed_eci.
    formation_energy : np.ndarray, shape(number_dft_computed_configs_)

    Returns
    -------
    mh_ratio : float
        Ratio defined in paper listed above- used in deciding whether to accept or reject proposed_eci.
    """

    left_term = (
        np.linalg.norm(proposed_eci, ord=1) / np.linalg.norm(current_eci, ord=1)
    ) ** (-1 * current_eci.shape[0])

    right_term_numerator = np.linalg.norm(formation_energy - proposed_energy)
    right_term_denom = np.linalg.norm(formation_energy - current_energy)

    right_term = (right_term_numerator / right_term_denom) ** (
        -1 * formation_energy.shape[0]
    )

    mh_ratio = left_term * right_term
    if mh_ratio > 1:
        mh_ratio = 1
    return mh_ratio


def metropolis_hastings_ratio_gpu(
    current_eci: cp.ndarray,
    proposed_eci: cp.ndarray,
    current_energy: cp.ndarray,
    proposed_energy: cp.ndarray,
    formation_energy: cp.ndarray,
):
    """Acceptance probability ratio defined in Zabaras et. al, https://doi.org/10.1016/j.cpc.2014.07.013. First part of equation (12)
    Parameters
    ----------
    current_eci : cp.ndarray shape(number_eci)
        Vector of current ECI values.
    proposed_eci : cp.ndarray shape(number_eci)
        Vector of proposed eci values, differing from current_eci by a random vector in ECI space.
    current_energy : cp.ndarray, shape(number_dft_computed_configs_)
        Energy calculated with current_eci.
    proposed_energy : cp.ndarray, shape(number_dft_computed_configs_)
        Energy calculated using proposed_eci.
    formation_energy : cp.ndarray, shape(number_dft_computed_configs_)

    Returns
    -------
    mh_ratio : float
        Ratio defined in paper listed above- used in deciding whether to accept or reject proposed_eci.
    """
    left_term = (
        cp.linalg.norm(proposed_eci, ord=1) / cp.linalg.norm(current_eci, ord=1)
    ) ** (-1 * current_eci.shape[0])

    right_term_numerator = cp.linalg.norm(formation_energy - proposed_energy)
    right_term_denom = cp.linalg.norm(formation_energy - current_energy)

    right_term = (right_term_numerator / right_term_denom) ** (
        -1 * formation_energy.shape[0]
    )

    mh_ratio = left_term * right_term
    if mh_ratio > 1:
        mh_ratio = 1
    return mh_ratio


def run_eci_monte_carlo(
    corr_comp_energy_file: str,
    eci_walk_step_size: float,
    iterations: int,
    sample_frequency: int,
    burn_in=1000000,
    output_file_path=False,
    use_gpu=False,
):
    """Samples ECI space according to Metropolis Monte Carlo, recording ECI values and most likely ground state configurations.

    Parameters
    ----------
    corr_comp_energy_file : str
        Path to json casm query output file (output of: "casm query -k corr comp formation_energy -j -o filename.json")
    eci_walk_step_size : int
        Magnitude of the random step vector in ECI space. (Try for a value that gives an acceptance rate of ~ 0.24)
    iterations : int
        Number of steps to perform in the monte carlo algorithm.
    sample_frequency : int
        The number of steps that pass before ECI and proposed ground states are recorded.
    burn_in : int
        The number of steps to "throw away" before ECI and proposed ground states are recorded.
    output_dir : str
        Path to the directory where monte carlo results should be written. By default, results are not written to a file.
    use_gpu : bool
        If the computer has a compatible nvidia gpu and use_gpu is True, uses gpu for matrix multiplications. Good for large matrices

    Returns
    -------
    results : dict
        sampled_eci : np.ndarray
            Matrix of recorded ECI. M rows of sampled ECI, where M = (Number of iterations / sample frequency). Each row is a set of N ECI, where N is the number of correlations.
        acceptance : np.ndarray
            Vector of booleans dictating whether a step was accepted (True) or rejected (False)
        acceptance_prob : float
            Number of accepted steps divided by number of total steps.
        proposed_ground_state_indices : np.ndarray
            Vector of indices denoting configurations which appeared below the DFT hull across all of the Monte Carlo steps.
        rms : np.ndarray
            Root Mean Squared Error of the calculated energy vs DFT energy for each Monte Carlo step.
        names : list
            List of configuraton names used in the Monte Carlo calculations.
    """
    # Read data from casm query json output
    data = read_corr_comp_formation(corr_comp_energy_file)
    corr = data["corr"]
    formation_energy = data["formation_energy"]
    comp = data["comp"]

    # Dealing with compatibility: Different descriptors for un-calculated formation energy (1.1.2->{}, 1.2-> null (i.e. None))
    uncalculated_energy_descriptor = None
    if {} in formation_energy:
        uncalculated_energy_descriptor = {}

    # downsampling only the calculated configs:
    downsample_selection = formation_energy != uncalculated_energy_descriptor
    corr_calculated = corr[downsample_selection]
    formation_energy_calculated = formation_energy[downsample_selection]
    if use_gpu:
        formation_energy_calculated_gpu = formation_energy_calculated.tolist()
        formation_energy_calculated_gpu = cp.array(formation_energy_calculated_gpu)
    comp_calculated = comp[downsample_selection]

    # Find and store the DFT hull:
    points = np.zeros(
        (formation_energy_calculated.shape[0], comp_calculated.shape[1] + 1)
    )
    points[:, 0:-1] = comp_calculated
    points[:, -1] = formation_energy_calculated
    hull = ConvexHull(points)
    dft_hull_simplices, dft_hull_vertices = lower_hull(hull, energy_index=-2)
    dft_hull_vertices = np.array(hull.points[dft_hull_vertices])

    # Run lassoCV to get expected eci values
    lasso_eci = run_lassocv(corr_calculated, formation_energy_calculated)

    if use_gpu:
        corr = cp.array(corr)
    # Instantiate lists
    acceptance = []
    rms = []
    sampled_eci = []
    proposed_ground_states_indices = np.array([])

    # Perform MH Monte Carlo
    current_eci = lasso_eci
    if use_gpu:
        current_eci = cp.array(current_eci)

    sampled_eci.append(current_eci)
    for i in tqdm(range(iterations), desc="Monte Carlo Progress"):
        eci_random_vec = generate_rand_eci_vec(
            num_eci=lasso_eci.shape[0], stdev=1, normalization=eci_walk_step_size
        )

        if use_gpu:
            corr_calculated = cp.array(corr_calculated)
            proposed_eci = current_eci + cp.array(eci_random_vec)
            current_energy = cp.matmul(corr_calculated, current_eci)
            proposed_energy = cp.matmul(corr_calculated, proposed_eci)

        else:
            proposed_eci = current_eci + eci_random_vec
            current_energy = np.matmul(corr_calculated, current_eci)
            proposed_energy = np.matmul(corr_calculated, proposed_eci)

        if use_gpu:
            mh_ratio = metropolis_hastings_ratio_gpu(
                current_eci,
                proposed_eci,
                current_energy,
                proposed_energy,
                formation_energy_calculated_gpu,
            )
        else:
            mh_ratio = metropolis_hastings_ratio(
                current_eci,
                proposed_eci,
                current_energy,
                proposed_energy,
                formation_energy_calculated,
            )

        acceptance_comparison = np.random.uniform()
        if mh_ratio >= acceptance_comparison:
            acceptance.append(True)
            current_eci = proposed_eci
            if use_gpu:
                energy_for_error = cp.asnumpy(proposed_energy)
            else:
                energy_for_error = proposed_energy
        else:
            acceptance.append(False)
            if use_gpu:
                energy_for_error = cp.asnumpy(current_energy)
            else:
                energy_for_error = current_energy

        # Calculate and append rms:
        mse = mean_squared_error(formation_energy_calculated, energy_for_error)
        rms.append(mse ** (1 / 2))

        # Compare to DFT hull
        if use_gpu:
            full_predicted_energy = cp.asnumpy(cp.matmul(corr, current_eci))

        else:
            full_predicted_energy = np.matmul(corr, current_eci)
        hulldist = checkhull(
            dft_hull_vertices[:, 0:-1],
            dft_hull_vertices[:, -1],
            comp,
            full_predicted_energy,
        )

        below_hull_selection = hulldist < 0
        below_hull_indices = np.ravel(np.array(below_hull_selection.nonzero()))

        # Only record a subset of all monte carlo steps to avoid excessive correlation
        if (i > burn_in) and (i % sample_frequency == 0):
            sampled_eci.append(current_eci)
            proposed_ground_states_indices = np.concatenate(
                (proposed_ground_states_indices, below_hull_indices)
            )

    acceptance = np.array(acceptance)
    acceptance_prob = np.count_nonzero(acceptance) / acceptance.shape[0]
    if use_gpu:
        sampled_eci = cp.array(sampled_eci)
    else:
        sampled_eci = np.array(sampled_eci)

    results = {
        "sampled_eci": sampled_eci,
        "acceptance": acceptance,
        "acceptance_prob": acceptance_prob,
        "proposed_ground_states_indices": proposed_ground_states_indices,
        "rms": rms,
        "names": data["names"],
        "lasso_eci": lasso_eci,
    }
    if output_file_path:
        print("Saving results to %s" % output_file_path)
        with open(output_file_path, "wb") as f:
            pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

    return results


def plot_eci_hist(eci_data):
    plt.hist(x=eci_data, bins="auto", color="xkcd:crimson", alpha=0.7, rwidth=0.85)

    plt.xlabel("ECI value (eV)", fontsize=18)
    plt.ylabel("Count", fontsize=18)
    # plt.show()
    fig = plt.gcf()
    return fig


def plot_eci_covariance(eci_data_1, eci_data_2):
    plt.scatter(eci_data_1, eci_data_2, color="xkcd:crimson")
    plt.xlabel("ECI 1 (eV)", fontsize=18)
    plt.ylabel("ECI 2 (eV)", fontsize=18)
    fig = plt.gcf()
    return fig


def plot_clex_hull_data_1_x(
    fit_dir,
    hall_of_fame_index,
    full_formation_energy_file="full_formation_energies.txt",
):
    """plot_clex_hull_data_1_x(fit_dir, hall_of_fame_index, full_formation_energy_file='full_formation_energies.txt')

    Function to plot DFT energies, cluster expansion energies, DFT convex hull and cluster expansion convex hull.

    Args:
        fit_dir (str): absolute path to a casm cluster expansion fit.
        hall_of_fame_index (int or str): Integer index. "hall of fame" index for a specific fit (corresponding to a set of "Effective Cluster Interactions" or ECI).
        full_formation_energy_file (str): filename that contains the formation energy of all configurations of interest. Generated using a casm command

    Returns:
        fig: a python figure object.
    """
    # TODO: Definitely want to re-implement this with json input
    # Pre-define values to pull from data files
    # title is intended to be in the form of "casm_root_name_name_of_specific_fit_directory".
    title = fit_dir.split("/")[-3] + "_" + fit_dir.split("/")[-1]
    dft_scel_names = []
    clex_scel_names = []
    dft_hull_data = []
    clex_hull_data = []
    cv = None
    rms = None
    wrms = None
    below_hull_exists = False
    hall_of_fame_index = str(hall_of_fame_index)

    # Read necessary files
    os.chdir(fit_dir)
    files = glob("*")
    for f in files:
        if "_%s_dft_gs" % hall_of_fame_index in f:
            dft_hull_path = os.path.join(fit_dir, f)
            dft_hull_data = np.genfromtxt(
                dft_hull_path, skip_header=1, usecols=list(range(1, 10))
            ).astype(float)
            with open(dft_hull_path, "r") as dft_dat_file:
                dft_scel_names = [
                    row[0] for row in csv.reader(dft_dat_file, delimiter=" ")
                ]
                dft_scel_names = dft_scel_names[1:]

        if "_%s_clex_gs" % hall_of_fame_index in f:
            clex_hull_path = os.path.join(fit_dir, f)
            clex_hull_data = np.genfromtxt(
                clex_hull_path, skip_header=1, usecols=list(range(1, 10))
            ).astype(float)
            with open(clex_hull_path, "r") as clex_dat_file:
                clex_scel_names = [
                    row[0] for row in csv.reader(clex_dat_file, delimiter=" ")
                ]
                clex_scel_names = clex_scel_names[1:]

        if "_%s_below_hull" % hall_of_fame_index in f:
            below_hull_exists = True
            below_hull_path = os.path.join(fit_dir, f)
            below_hull_data = np.reshape(
                np.genfromtxt(
                    below_hull_path, skip_header=1, usecols=list(range(1, 10))
                ).astype(float),
                ((-1, 9)),
            )
            with open(below_hull_path, "r") as below_hull_file:
                below_hull_scel_names = [
                    row[0] for row in csv.reader(below_hull_file, delimiter=" ")
                ]
                below_hull_scel_names = below_hull_scel_names[1:]

        if "check.%s" % hall_of_fame_index in f:
            checkfile_path = os.path.join(fit_dir, f)
            with open(checkfile_path, "r") as checkfile:
                linecount = 0
                cv_rms_wrms_info_line = int
                for line in checkfile.readlines():
                    if (
                        line.strip() == "-- Check: individual 0  --"
                    ):  # % hall_of_fame_index:
                        cv_rms_wrms_info_line = linecount + 3

                    if linecount == cv_rms_wrms_info_line:
                        cv = float(line.split()[3])
                        rms = float(line.split()[4])
                        wrms = float(line.split()[5])
                    linecount += 1

    # Generate the plot
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.text(
        0.80,
        0.80 * min(dft_hull_data[:, 4]),
        "CV:      %.10f\nRMS:    %.10f\nWRMS: %.10f" % (cv, rms, wrms),
        fontsize=15,
    )
    labels = []
    plt.title(title, fontsize=30)
    plt.xlabel(r"Composition", fontsize=20)
    plt.ylabel(r"Energy $\frac{eV}{prim}$", fontsize=20)
    plt.plot(dft_hull_data[:, 1], dft_hull_data[:, 5], marker="o", color="xkcd:crimson")
    labels.append("DFT Hull")
    plt.plot(
        clex_hull_data[:, 1],
        clex_hull_data[:, 8],
        marker="o",
        linestyle="dashed",
        color="b",
    )
    labels.append("ClEx Hull")
    plt.scatter(dft_hull_data[:, 1], dft_hull_data[:, 8], color="k")
    labels.append("Clex Prediction of DFT Hull")

    if full_formation_energy_file:
        # format:
        # run casm query -k comp formation_energy hull_dist clex clex_hull_dist -o full_formation_energies.txt
        #            configname    selected           comp(a)    formation_energy    hull_dist(MASTER,atom_frac)        clex()    clex_hull_dist(MASTER,atom_frac)
        datafile = full_formation_energy_file
        data = np.genfromtxt(datafile, skip_header=1, usecols=list(range(2, 7))).astype(
            float
        )
        composition = data[:, 0]
        dft_formation_energy = data[:, 1]
        clex_formation_energy = data[:, 3]
        plt.scatter(composition, dft_formation_energy, color="salmon")
        labels.append("DFT energies")
        plt.scatter(composition, clex_formation_energy, marker="x", color="skyblue")
        labels.append("ClEx energies")

    # TODO: This implementation is wrong. This is the distance below the hull (energy difference) not the actual energy.
    if below_hull_exists:
        plt.scatter(below_hull_data[:, 1], below_hull_data[:, 7], marker="+", color="k")
        labels.append("Clex Below Clex Prediction of DFT Hull Configs")
    else:
        print("'_%s_below_hull' file doesn't exist" % hall_of_fame_index)

    plt.legend(labels, loc="lower left", fontsize=10)

    fig = plt.gcf()
    return fig
