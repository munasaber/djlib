import json
import matplotlib.pyplot as plt
import os
from scipy import integrate
import scipy.interpolate
import scipy.optimize
import numpy as np
from glob import glob
from djlib import *


def read_mc_results_file(results_file_path):
    """Function to parse mc results.json files.
    Args;
        results_file_path(str): Path to the results.json file for the given monte carlo simulation.
    Returns:
        x(list): Vector of compostitions
        b(list): Vector of beta values
        temperature(list): Vecor of temperature values (K)
        potential_energy(list): Vector of potential energy values (E-mu*x)
    """
    with open(results_file_path) as f:
        results = json.load(f)

    mu = np.array(results["param_chem_pot(a)"])
    x = np.array(results["<comp(a)>"])
    b = np.array(results["Beta"])
    temperature = np.array(results["T"])
    potential_energy = np.array(results["<potential_energy>"])
    return (mu, x, b, temperature, potential_energy)


def read_superdupercell(mc_settings_file):
    """Function to read mu / temperature values as well as superdupercell from a monte carlo settings.json file.
    Args:
        mc_settings_file(str): Path to mc_settings.json file.
    Returns:
        superdupercell(list): Matrix (list of 3 lists) that describes a supercell for the monte carlo simulation
    """
    with open(mc_settings_file) as f:
        settings = json.load(f)
    superdupercell = settings["supercell"]
    return superdupercell


class lte_run:
    def __init__(self, lte_dir):
        self.path = lte_dir
        self.read_lte_results()

    def read_lte_results(self):
        results_file = os.path.join(self.path, "results.json")
        with open(results_file) as f:
            results = json.load(f)
        self.mu = results["param_chem_pot(a)"]
        self.b = results["Beta"]
        self.t = results["T"]
        self.x = results["gs_comp(a)"]
        self.pot_eng = results["phi_LTE"]
        self.superdupercell = read_superdupercell(
            os.path.join(self.path, "mc_settings.json")
        )


class constant_t_run:
    def __init__(self, const_t_dir):
        self.path = const_t_dir
        results_file_path = os.path.join(self.path, "results.json")
        self.mu, self.x, self.b, self.t, self.pot_eng = read_mc_results_file(
            results_file_path
        )
        self.integrate_constant_temp_grand_canonical()
        self.superdupercell = read_superdupercell(
            os.path.join(self.path, "mc_settings.json")
        )

    def integrate_constant_temp_grand_canonical(self):
        """Function to integrate across mu values at a constant temperature
        Args:
            x(list): Vector of compostitions
            b(list): Vector of beta values
            potential_energy(list): Vector of potential energy values (E-mu*x)
            mu(list): Vector of mu values

        Returns:
            integrated_potential(list): List of grand canonical free energy values corresponding to a fixed temperature / beta value.
        """
        free_energy_reference = self.pot_eng[0]
        integrated_potential = []
        for index, value in enumerate(self.mu):
            index = index + 1
            # if index == 0:
            #    integrated_potential.append(free_energy_reference)
            if index > 0:
                current_mu = self.mu[0:index]
                current_b = self.b[0:index]
                current_x = self.x[0:index]
                integrated_potential.append(
                    (1 / current_b[-1])
                    * (
                        self.b[0] * free_energy_reference
                        + integrate.simpson((-1 * current_b * current_x), current_mu)
                    )
                )
        self.integ_grand_canonical = np.asarray(integrated_potential)


class heating_run:
    def __init__(self, heating_dir, lte_run):
        self.path = heating_dir
        results_file_path = os.path.join(self.path, "results.json")
        self.mu, self.x, self.b, self.t, self.pot_eng = read_mc_results_file(
            results_file_path
        )
        self.get_lte_reference_energy(lte_run)
        self.integrate_heating_grand_canonical_from_lte()
        self.superdupercell = read_superdupercell(
            os.path.join(self.path, "mc_settings.json")
        )

    def get_lte_reference_energy(self, lte_run):
        mu_index = find(lte_run.mu, self.mu[0])
        self.energy_reference = lte_run.pot_eng[mu_index]

    def integrate_heating_grand_canonical_from_lte(self):
        """Function to integrate the grand canonical free energy from monte carlo heating run results.
        Args:
            x(list): Vector of compostitions
            b(list): Vector of beta values
            potential_energy(list): Vector of potential energy values (E-mu*x)
            mu(list): Vector of mu values

        Returns:
            integrated_potential(list): List of grand canonical free energy values corresponding to a fixed mu value.
        """

        self.pot_eng[0] = self.energy_reference

        integrated_potential = []
        for index in range(len(self.b)):
            index = index + 1
            # if index == 0:
            #    integrated_potential.append(free_energy_reference)
            if index > 0:
                current_b = self.b[0:index]
                current_potential_energy = self.pot_eng[0:index]
                integrated_potential.append(
                    (1 / current_b[-1])
                    * (
                        self.b[0] * self.energy_reference
                        + integrate.simpson(current_potential_energy, current_b)
                    )
                )
        self.integ_grand_canonical = np.asarray(integrated_potential)


class cooling_run:
    def __init__(self, cooling_dir, constant_t_run):
        self.path = cooling_dir
        results_file_path = os.path.join(self.path, "results.json")
        self.mu, self.x, self.b, self.t, self.pot_eng = read_mc_results_file(
            results_file_path
        )
        self.get_constant_t_reference_energy(constant_t_run)
        self.integrate_cooling_from_const_t_run()
        self.superdupercell = read_superdupercell(
            os.path.join(self.path, "mc_settings.json")
        )

    def get_constant_t_reference_energy(self, constant_t_run):
        mu_index = find(constant_t_run.mu, self.mu[0])
        self.energy_reference = constant_t_run.integ_grand_canonical[mu_index]

    def integrate_cooling_from_const_t_run(self):
        free_energy_reference = self.energy_reference
        integrated_potential = []
        for index, value in enumerate(self.b):
            index = index + 1
            # if index == 0:
            #    integrated_potential.append(free_energy_reference)
            if index > 0:
                current_b = self.b[0:index]
                current_potential_energy = self.pot_eng[0:index]
                integrated_potential.append(
                    (1 / current_b[-1])
                    * (
                        self.b[0] * free_energy_reference
                        + integrate.simpson(current_potential_energy, current_b)
                    )
                )
        self.integ_grand_canonical = np.asarray(integrated_potential)


def plot_heating_and_cooling(heating_run, cooling_run):
    bullet_size = 3
    if heating_run.mu[0] != cooling_run.mu[0]:
        print(
            "WARNING: Chemical potentials do not match between the heating and cooling runs"
        )
    plt.title("Constant Mu: %.4f" % heating_run.mu[0])
    plt.xlabel("Temperature (K)", fontsize=18)
    plt.ylabel("Grand Canonical Free Energy", fontsize=18)
    plt.scatter(cooling_run.t, cooling_run.integ_grand_canonical, s=bullet_size)
    plt.scatter(heating_run.t, heating_run.integ_grand_canonical, s=bullet_size)
    plt.legend(["Cooling", "Heating"])
    fig = plt.gcf()
    fig.set_size_inches(15, 19)
    return fig


def compile_all_gibbs_from_mc_heating_runs(heating_runs_dir):
    """Computes gibbs free energies for heating runs, and returns gibbs and composition vectors for each fixed T

    Args:
        heating_runs_dir(str): Path to the directory containing the monte carlo simulations of heating runs.

    Returns:
        t_indexed_gibbs_results_list(list): List of "n" dictionaries, where "n" is the number of fiexed temperatures in the monte carlo simulation
                                            Each dictionary takes the form:
                                            {'T':t,                             #Temperature value: a scalar
                                            'mu':mu_at_fixed_t,                 #chemical potential values: a vector of length "m"
                                            'x':x_at_fixed_t,                   #composition values: a vector of length "m"
                                            'gibbs':gibbs_at_fixed_t}           #Gibbs free energy values: a vector of length "m"

    """
    mc_directory_list = glob(os.path.join(heating_runs_dir, "mu_*"))

    gibbs_results_list = []
    for run in mc_directory_list:

        # read mc results file
        results_file = os.path.join(run, "results.json")
        settings_file = os.path.join(run, "mc_settings.json")
        with open(results_file) as f:
            results = json.load(f)
        with open(settings_file) as f:
            settings = json.load(f)
        mu_value = settings["driver"]["initial_conditions"]["param_chem_pot"]["a"]

        x = results["<comp_n(N)>"]
        b = np.asarray(results["Beta"])
        temperature = results["T"]
        E = results["<formation_energy>"]
        potential_energy = results["<potential_energy>"]
        mu = np.ones(b.shape[0]) * mu_value

        # Calculate gibbs
        integrated_potential = integrate_heating_grand_canonical_free_energy(
            x, b, potential_energy, mu
        )
        gibbs = integrated_potential + mu * x

        # Format and return data (indexed by mu, not by temperature):
        gibbs_results_list.append(
            {"mu": mu[0], "temperature": temperature, "x": x, "gibbs": gibbs}
        )

    # Re-format to indexing by temperature
    t_indexed_gibbs_results_list = []
    for t_index, t in enumerate(temperature):
        print(t)
        mu_at_fixed_t = []
        x_at_fixed_t = []
        gibbs_at_fixed_t = []

        # collect mu's, x's and gibbs energies consistent with fixed temperature
        for fixed_mu_run in gibbs_results_list:
            mu_at_fixed_t.append(fixed_mu_run["mu"])
            x_at_fixed_t.append(fixed_mu_run["x"][t_index])
            gibbs_at_fixed_t.append(fixed_mu_run["gibbs"][t_index])
        t_indexed_gibbs_results_list.append(
            {"T": t, "mu": mu_at_fixed_t, "x": x_at_fixed_t, "gibbs": gibbs_at_fixed_t}
        )
    return t_indexed_gibbs_results_list


def predict_free_energy_crossing(
    heating_integrated_free_energy,
    temp_heating,
    cooling_integrated_free_energy,
    temp_cooling,
):
    """Function to find crossing point between two (energy vs T) curves.
    Args:
        heating_integrated_free_energy(list): Vector of integrated free energy values from heating monte carlo simulation.
        temp_heating(list): Vector of temperature values (K) from the heating monte carlo simulation.
        cooling_integrated_free_energy(list): Vector of integrated free energy values from cooling monte carlo simulation.
        temp_cooling(list): Vector of temperature values (K) from the cooling monte carlo simulation.

    Returns:
        tuple(
            t_intersect_predict

        )

    """
    # Check that lengths of all vectors match and that temp_heating == temp_cooling (i.e., they're not the reverse of each other)
    if (
        len(heating_integrated_free_energy)
        == len(temp_heating)
        == len(cooling_integrated_free_energy)
        == len(temp_cooling)
    ):

        find_intersection = False
        if temp_heating.all() == temp_cooling.all():
            find_intersection = True
        elif temp_heating != temp_cooling:
            # If the temperature axes arent the same, try swapping the order of temp_cooling and cooling_integrated_free_energy.
            temp_cooling = np.flip(np.asarray(temp_heating))
            cooling_integrated_free_energy = np.flip(
                np.asarray(cooling_integrated_free_energy)
            )

            # If the temperature axes still aren't the same, cancel the function.
            if temp_heating == temp_cooling:
                find_intersection = True
            elif temp_heating != temp_cooling:
                print(
                    "temp_heating and temp_cooling are the same length but do not match. See printout below:\ntemp_heating  temp_cooling"
                )
                for idx, value in enumerate(temp_heating):
                    print("%.3f  %.3f" % temp_heating[idx], temp_cooling[idx])

        if find_intersection:
            # TODO: Check that there isn't more than one intersection (complete overlap) or no intersection.

            # fit spline to each dataset, calculate intersection
            interp_heating = scipy.interpolate.InterpolatedUnivariateSpline(
                temp_heating, heating_integrated_free_energy
            )
            interp_cooling = scipy.interpolate.InterpolatedUnivariateSpline(
                temp_cooling, cooling_integrated_free_energy
            )

            # define a difference function to calculate the root
            def difference(t):
                return np.abs(interp_heating(t) - interp_cooling(t))

            # Provide a composition x0 as a guess for the root finder
            # This will break if there are multiple identical minimum values
            t0_index = np.argmin(
                abs(heating_integrated_free_energy - cooling_integrated_free_energy)
            )
            t0_guess = temp_heating[t0_index]

            # Calculate the intersection point
            t_intersect_predict = scipy.optimize.fsolve(difference, x0=x0_guess)
            energy_intersect_predict = interp_heating(x_intersect_predict)

            return (t_intersect_predict, energy_intersect_predict)

    else:
        print(
            "The free energies and composition vectors do not have the same lengths.\nCurrent lengths are:"
        )
        print("length of temp_heating: %d" % len(temp_heating))
        print(
            "length of heating_integrated_free_energy: %d"
            % len(heating_integrated_free_energy)
        )
        print("length of temp_cooling: %d" % len(temp_cooling))
        print(
            "length of cooling_integrated_free_energy: %d"
            % len(cooling_integrated_free_energy)
        )


def find_crossing_compositions(
    integrated_energies, temperature, x, t_intersect_predict, energy_intersect_predict
):
    """Given an interpolated point in (energy vs temperature) space, find the closest existing (energy, temperature) and return the corresponding composition x and corresponding temperature.
    Args:
        integrated_energies(list): Vector of integrated energy values.
        temperature(list): Vector of temperature values (K).
        x(list): Vector of composition values.
        t_intersect_predict(float): Interpolated prediction of the free energy crossing temperature between a heating and cooling grand canonical monte carlo simulation.
        energy_intersect_predict(float): Interpolated prediction of the free energy at the crossing temperature between a heating and cooling grand canonical monte carlo simulation.

    Returns:
        tuple(
            x_at_crossing(float): Composition at the actual coordinates closest to the predicted
            t_at_crossing(float): Temperature (K) at the actual coordinates closest to the predicted
        )
    """

    temperature_and_energy = np.zeros((len(temperature), 2))
    temperature_and_energy[:, 0] = temperature
    temperature_and_energy[:, 1] = integrated_energies

    prediction_point = np.array([t_intersect_predict, energy_intersect_predict])

    difference = temperature_and_energy - prediction_point

    distance = np.sum(np.abs(temperature_and_energy) ** 2, axis=-1) ** (1 / 2)
    closest_point_index = np.argmin(distance)

    x_at_crossing = x[closest_point_index]
    t_at_crossing = temperature[closest_point_index]
    return (x_at_crossing, t_at_crossing)


def simulation_is_complete(mc_run_dir):
    """Check that a grand canonical monte carlo simulation has finished
    Args:
        mc_run_dir(str): Path to a monte carlo simulation directory.

    Returns:
        simulation_status(bool): simulation is complete (True) or simulation is not complete (False)
    """
    # Check the number of conditions (mu, temp) that should be completed
    mc_settings_file = os.path.join(mc_run_dir, "mc_settings.json")
    (
        mu_values,
        temperature_values,
        superdupercell,
    ) = read_mu_temperature_and_superdupercell(mc_settings_file)
    number_of_conditions = len(mu_values) * len(temperature_values)

    # Check that the final condition exists
    final_conditions_path = os.path.join(
        mc_run_dir, "conditions.%d" % number_of_conditions
    )
    simulation_status = None
    if os.path.isdir(final_conditions_path):
        simulation_status = True
    else:
        simulation_status = False

    return simulation_status


def plot_mc_results(mc_runs_directory, save_image=False, show_labels=False):
    """plot_mc_results(mc_runs_directory, save_image_path=False, show_labels=False)

    Generate a single (T vs composition) plot using all monte carlo runs in mc_runs_directory.
    Args:
        mc_runs_directory(str): Path to the directory containing all grand canonical monte carlo runs.
        same_image(bool): Whether the image will be saved to the run directory or not.
        show_labels(bool): Whether or not the plot legend displays.

    Returns:
        fig(matplotlib.pyplot figure object): 2D plot object. Can do fig.show() to display the plot.
    """
    labels = []
    for subdir, dirs, files in os.walk(mc_runs_directory):
        for filename in files:
            if filename == "results.json":
                datafile = os.path.join(subdir, "results.json")
                with open(datafile) as f:
                    data = json.load(f)
                    f.close()
                    current_mc = subdir.split("/")[-1]
                    labels.append(current_mc)
                    composition = data["<comp(a)>"]
                    temperature = data["T"]
                    plt.scatter(composition, temperature)

    if show_labels:
        plt.legend(labels)
    plt.xlabel("Composition", fontsize=18)
    plt.ylabel("Temperature (K)", fontsize=18)
    title = (
        "Chemical Potential and Temperature Sweep Rain Plot: %s"
        % mc_runs_directory.split("/")[-1]
    )
    plt.title("Chemical Potential and Temperature Sweep Rain Plot", fontsize=30)
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10)
    if save_image:
        fig.savefig(os.path.join(mc_runs_directory, title + ".png"), dpi=100)
        print("Saving image to file: ", end="")
        print(os.path.join(mc_runs_directory, title + ".png"))
    return fig
