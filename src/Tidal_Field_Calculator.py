"""
Tidal_Field_Calculator.py
-------------------------------

Author: Asit Dave
Date: 31-05-2024
License: MIT

Description:
    This script is designed to compute the density field, tidal tensor, potential field, and traceless tidal
    shear tensor for a given cosmological N-body simulation snapshot (pynbody.snapshot). Users have the option
    to smooth the density field using a Gaussian filter and can select whether to compute the potential field
    and the traceless tidal shear tensor.

Instructions:
    1. Make sure the config/input_params.txt file is present in the current working directory.
    2. Make sure the LSS_TWeb_BlackBox.py script is present in the current working directory.
    3. The script expects you to have a simulation snapshot for the analysis.
    4. Parameters that can be modified in this script are:
        - Projection of the box for the plots (default: xy)
        - Thickness of the slice to average the density field over. (default: half box thickness)

Execution:
    You can either run the script for all the smoothing scales you want to perform the calculations for in one go
    OR
    you can run the script for all smoothing scales one by one and then run the Tweb_Structure_Classifier.py script.

"""

# ----------------------------------------- IMPORT LIBRARIES ----------------------------------------------#

import pynbody
import numpy as np
from tqdm import tqdm

import os
import logging
import time

from LSS_TWeb_BlackBox import *

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Start the timer
start_time = time.time()

# ------------------------------ READ INPUT FILE ----------------------------------#


def read_input_txt_file(file_path: str) -> tuple:
    """
    Read the config/input_params.txt file and extract the user inputs.

    Parameters:
    - file_path (str): The path to the input parameters file.

    Returns:
    - tuple: A tuple containing extracted parameters.
    """
    try:
        (
            snapshot_path,
            save_path,
            grid_size,
            create_density,
            own_density_path,
            smoothing_scales,
            calculate_potential,
            calculate_traceless,
        ) = read_input_file(file_path)

        logging.info("Input parameters read successfully.")

        return (
            snapshot_path,
            save_path,
            grid_size,
            create_density,
            own_density_path,
            smoothing_scales,
            calculate_potential,
            calculate_traceless,
        )

    except Exception as e:
        logging.error(f"Error reading input file: {e}")
        raise


# ------------------------------ LOAD THE SNAPSHOT --------------------------------#


def get_snapshot(
    snapshot_path: str,
) -> tuple[pynbody.snapshot.SimSnap, pynbody.simdict.SimDict]:
    """
    Load the simulation snapshot from the specified path.

    Parameters:
    - snapshot_path (str): The file path to the snapshot.

    Returns:
    - tuple: A tuple containing the loaded snapshot and its header.
    """
    try:
        logging.info("Loading the snapshot...")
        snap, header = load_snapshot(snapshot_path)
        logging.info("Snapshot loaded successfully.")
        return snap, header
    except Exception as e:
        logging.error(f"Error loading snapshot: {e}")
        raise


# ------------------------------ EXTRACT SIMULATION PROPERTIES --------------------------------#


def extract_simulation_params(
    snapshot_header: pynbody.simdict.SimDict,
) -> dict[str, float]:
    """
    Extract values from the simulation dictionary.

    Parameters:
    - snapshot_header (pynbody.simdict.SimDict): The simulation dictionary.

    Returns:
    - dict: A dictionary containing the extracted simulation properties
            --> keys: ['omegaM0', 'omegaL0', 'a', 'h']; values: [float]
    """
    try:
        extracted_values = extract_simdict_values(simdict=snapshot_header)
        return extracted_values
    except Exception as e:
        logging.error(f"Error extracting simulation parameters: {e}")
        raise


# ------------------------------ SAVE THE SIMULATION PROPERTIES --------------------------------#


def save_simulation_properties(
    snapshot_header: pynbody.simdict.SimDict,
    snapshot: pynbody.snapshot.SimSnap,
    output_dir: str,
) -> None:
    """
    Save the simulation properties to a file.

    Parameters:
    - snapshot_header (pynbody.simdict.SimDict): The simulation header.
    - snapshot (pynbody.snapshot.SimSnap): The loaded snapshot.
    - output_dir (str): Directory where simulation_properties.txt will be saved.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        sim_props_path = os.path.join(output_dir, "simulation_properties.txt")

        with open(sim_props_path, "w") as f:
            f.write("-------------------------------------------------------------\n")
            f.write(" The file contains the simulation properties and parameters \n")
            f.write("-------------------------------------------------------------\n\n")
            f.write(f"Box size: {snapshot_header['boxsize']}\n")
            f.write(f"Omega_M0: {snapshot_header['omegaM0']}\n")
            f.write(f"Omega_L0: {snapshot_header['omegaL0']}\n")
            f.write(f"a: {snapshot_header['a']}\n")
            f.write(f"h: {snapshot_header['h']}\n")
            f.write(f"time: {snapshot_header['time']}\n\n")
            mean_mass = snapshot["mass"].mean()
            f.write(f"Mass of each particle: {mean_mass} * {mean_mass.units}\n")
            total_particles = snapshot["pos"].shape[0]
            f.write(f"Total number of particles: {total_particles}\n")
        logging.info(f"Simulation properties saved successfully at {sim_props_path}.")
    except Exception as e:
        logging.error(f"Error saving simulation properties: {e}")
        raise


# ------------------------------ SAVE THE PARTICLE POSITIONS AND VELOCITIES --------------------------------#


def save_particle_positions_and_velocities(snapshot: pynbody.snapshot.SimSnap) -> None:
    """
    Save the particle positions and velocities from the snapshot.

    Parameters:
    - snapshot (pynbody.snapshot.SimSnap): The loaded snapshot.
    """
    try:
        save_path_particle_position = os.path.join(save_path, "particle_positions.npy")
        if not os.path.exists(save_path_particle_position):
            save_data(data=snapshot["pos"], file_path=save_path_particle_position)
            logging.info(
                "Particle position file ('particle_positions.npy') saved successfully."
            )
        else:
            logging.info("Particle position file already exists, skipping...")
    except Exception as e:
        logging.error(f"Error saving particle positions: {e}")

    try:
        save_path_particle_velocity = os.path.join(save_path, "particle_velocity.npy")
        if not os.path.exists(save_path_particle_velocity):
            save_data(data=snapshot["vel"], file_path=save_path_particle_velocity)
            logging.info(
                "Particle velocity file ('particle_velocity.npy') saved successfully."
            )
        else:
            logging.info("Particle velocity file already exists, skipping...")
    except Exception as e:
        logging.error(f"Error saving particle velocities: {e}")


# ------------------------------ CREATE A DENSITY FIELD --------------------------------#


def get_density_field(snapshot: pynbody.simdict, mas: str, verbose: bool) -> np.ndarray:
    """
    Compute the density field or load an existing one.

    Parameters:
    - snapshot (pynbody.simdict): The loaded snapshot.
    - mas (str): The mass-assignment scheme. Options: 'NGP', 'CIC', 'TSC', 'PCS', 'gaussian'.
    - verbose (bool): Print information on progress.

    Returns:
    - np.ndarray: The computed or loaded density field.
    """
    if not create_density:
        logging.info("Loading the density field...")
        try:
            rho = load_data(own_density_path)
            logging.info("Density field loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading density field: {e}")
            raise
    else:
        save_path_density = os.path.join(save_path, "density_field.npy")

        # Check if density field already exists
        if os.path.exists(save_path_density):
            logging.info("Loading existing density field...")
            try:
                rho = load_data(save_path_density)
                logging.info("Density field loaded successfully.")
            except Exception as e:
                logging.error(f"Error loading existing density field: {e}")
                raise
        else:
            logging.info("Computing the density field...")
            try:
                rho = compute_density_field(
                    snapshot=snapshot,
                    grid_size=grid_size,
                    box_size=box_size,
                    mas=mas,
                    verbose=verbose,
                )
                logging.info("Density field computed successfully.")
                save_data(data=rho, file_path=save_path_density)
                logging.info("Density field ('density_field.npy') saved successfully.")
            except Exception as e:
                logging.error(f"Error computing density field: {e}")
                raise

    return rho


# ------------------------------ EXTRACT SMOOTHING SCALES --------------------------------#


def extract_smoothing_scales(
    smoothing_scales: list[float],
) -> tuple[list[float], list[str]]:
    """
    Extract and truncate (remove decimals) the smoothing scales.

    Parameters:
    - smoothing_scales (list[float]): List of smoothing scales.

    Returns:
    - tuple: A tuple containing the original floating-point smoothing scales and their truncated string forms.
    """
    try:
        smth_scales, truncated_scales = extract_scales(smoothing_scales)
        return smth_scales, truncated_scales
    except Exception as e:
        logging.error(f"Error extracting smoothing scales: {e}")
        raise


# ------------------------------ SMOOTH THE DENSITY FIELD --------------------------------#


def get_smoothed_field(
    input_field: np.ndarray, smoothing_scales: list[float], truncated_scales: list[str]
) -> list[np.ndarray]:
    """
    Smooth the density field using the specified smoothing scales.

    Parameters:
    - input_field (np.ndarray): The input density field.
    - smoothing_scales (list[float]): List of smoothing scales.
    - truncated_scales (list[str]): List of truncated smoothing scale names.

    Returns:
    - list[np.ndarray]: List of smoothed density fields.
    """
    logging.info("Smoothing the density field....")
    try:
        create_directory(
            os.path.join(f"{save_path}", "smoothed_density_fields"), overwrite=False
        )
        smoothened_rho = []
        for i, smth_scale in enumerate(tqdm(smoothing_scales)):
            save_path_smooth = os.path.join(
                save_path,
                "smoothed_density_fields",
                f"smoothed_density_field_{truncated_scales[i]}.npy",
            )

            # Check if smoothed field already exists
            if os.path.exists(save_path_smooth):
                logging.info(
                    f"Loading existing smoothed density field for scale {truncated_scales[i]}"
                )
                smooth_rho = load_data(save_path_smooth)
            else:
                logging.info(
                    f"Computing smoothed density field for scale {truncated_scales[i]}"
                )
                smooth_rho = smooth_field(
                    input_field=input_field,
                    smoothing_scale=smth_scale,
                    box_size=box_size,
                    grid_size=grid_size,
                )
                save_data(data=smooth_rho, file_path=save_path_smooth)

            smoothened_rho.append(smooth_rho)
        logging.info(
            "Density field smoothed for all the smoothing scales and files saved successfully."
        )
    except Exception as e:
        logging.error(f"Error smoothing density field: {e}")
        raise

    return smoothened_rho


# ------------------------------ PLOT THE DENSITY FIELD --------------------------------#


def plot_density_field(
    input_field: np.ndarray,
    smoothing_scales: list[float],
    truncated_scales: list[str],
    projection: str,
    slice_thickness: list[int, int],
    save_path: str,
) -> None:
    """
    Plot the smoothed density fields.

    Parameters:
    - input_field (np.ndarray): The smoothed density fields.
    - smoothing_scales (list[float]): List of smoothing scales.
    - truncated_scales (list[str]): List of truncated smoothing scale names.
    - projection (str): The projection type ('xy', 'yz', 'xz').
    - slice_thickness (list[int, int]): The range of slices to average over [Start index, End index].
    """
    try:
        create_directory(os.path.join(f"{save_path}", "density_plots"), overwrite=False)

        logging.info("Plotting the density field for respective smoothing scales...")

        for i, sm_scale in enumerate(smoothing_scales):
            save_sm_path = os.path.join(save_path, "density_plots")
            plot_field(
                input_field=input_field[i],
                sm_scale=sm_scale,
                name_sm_scale=truncated_scales[i],
                projection=projection,
                slice_index=slice_thickness,
                filepath=save_sm_path,
            )

        logging.info("Density field plots saved successfully.")

    except Exception as e:
        logging.error(f"Error plotting density field: {e}")
        raise


# ------------------------------ CALCULATE TIDAL SHEAR TENSOR --------------------------------#


def get_tidal_tensor(smoothed_density_field: np.ndarray) -> None:
    """
    Calculate the tidal tensor and potential field.

    Parameters:
    - smoothed_density_field (np.ndarray): The smoothed density field.
    """

    try:
        create_directory(os.path.join(f"{save_path}", "tidal_fields"), overwrite=False)
        for i in tqdm(range(len(smoothed_density_field))):
            if calculate_potential:
                create_directory(
                    os.path.join(f"{save_path}", "potential_field"), overwrite=False
                )

                save_path_tidal_tensor = os.path.join(
                    save_path, "tidal_fields", f"tidal_tensor_{truncated_scales[i]}.npy"
                )
                save_path_tidal_potential = os.path.join(
                    save_path,
                    "potential_field",
                    f"potential_field_{truncated_scales[i]}.npy",
                )
                save_path_traceless = (
                    os.path.join(
                        save_path,
                        "tidal_fields",
                        f"traceless_tidal_shear_{truncated_scales[i]}.npy",
                    )
                    if calculate_traceless
                    else None
                )

                # Check if files already exist
                files_exist = os.path.exists(save_path_tidal_tensor) and os.path.exists(
                    save_path_tidal_potential
                )
                if calculate_traceless:
                    files_exist = files_exist and os.path.exists(save_path_traceless)

                if files_exist:
                    logging.info(
                        f"Tidal tensor files already exist for scale {truncated_scales[i]}, skipping..."
                    )
                    continue

                logging.info(
                    f"Calculating the tidal tensor and potential field for scale {truncated_scales[i]}..."
                )
                tidal_tensor, Grav_potential = calculate_tidal_tensor(
                    density_field=smoothed_density_field[i], calculate_potential=True
                )

                save_data(data=tidal_tensor, file_path=save_path_tidal_tensor)
                save_data(data=Grav_potential, file_path=save_path_tidal_potential)

                if calculate_traceless:
                    traceless_tidal_shear = calculate_traceless_tidal_shear(
                        tidal_tensor, grid_size
                    )
                    save_data(data=traceless_tidal_shear, file_path=save_path_traceless)
            else:
                save_path_tidal_tensor = os.path.join(
                    save_path, "tidal_fields", f"tidal_tensor_{truncated_scales[i]}.npy"
                )
                save_path_traceless = (
                    os.path.join(
                        save_path,
                        "tidal_fields",
                        f"traceless_tidal_shear_{truncated_scales[i]}.npy",
                    )
                    if calculate_traceless
                    else None
                )

                # Check if files already exist
                files_exist = os.path.exists(save_path_tidal_tensor)
                if calculate_traceless:
                    files_exist = files_exist and os.path.exists(save_path_traceless)

                if files_exist:
                    logging.info(
                        f"Tidal tensor files already exist for scale {truncated_scales[i]}, skipping..."
                    )
                    continue

                logging.info(
                    f"Calculating the tidal tensor for scale {truncated_scales[i]}..."
                )
                tidal_tensor = calculate_tidal_tensor(
                    density_field=smoothed_density_field[i], calculate_potential=False
                )

                save_data(data=tidal_tensor, file_path=save_path_tidal_tensor)

                if calculate_traceless:
                    traceless_tidal_shear = calculate_traceless_tidal_shear(
                        tidal_tensor=tidal_tensor, grid_size=grid_size
                    )

                    save_data(data=traceless_tidal_shear, file_path=save_path_traceless)
        logging.info("Tidal tensor calculations completed.")
    except Exception as e:
        logging.error(f"Error calculating tidal tensor: {e}")
        raise


# ----------------------------------------- MAIN EXECUTION ----------------------------------------------#

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Tidal Field Calculator")
    parser.add_argument("--mas", type=str, default="TSC", help="Mass Assignment Scheme")
    parser.add_argument(
        "--force", action="store_true", help="Force recalculation even if files exist"
    )
    args = parser.parse_args()
    mas = args.mas
    force_recalc = args.force

    try:
        (
            snapshot_path,
            save_path,
            grid_size,
            create_density,
            own_density_path,
            smoothing_scales,
            calculate_potential,
            calculate_traceless,
        ) = read_input_txt_file("config/input_params.txt")

        smth_scales, truncated_scales = extract_smoothing_scales(smoothing_scales)

        # Check if files already exist (unless forced recalculation)
        if not force_recalc:
            scale_status = check_tidal_field_files_exist(
                save_path, truncated_scales, calculate_potential, calculate_traceless
            )
            missing_scales = get_missing_scales(scale_status)

            if not missing_scales:
                logging.info(
                    "All tidal field files already exist. Skipping computation."
                )
                logging.info("Use --force flag to force recalculation.")
                exit(0)
            else:
                existing_scales = [
                    scale for scale, exists in scale_status.items() if exists
                ]
                if existing_scales:
                    logging.info(
                        f"Files already exist for smoothing scales: {existing_scales}"
                    )
                logging.info(
                    f"Computing missing files for smoothing scales: {missing_scales}"
                )
                # Filter scales to only compute missing ones
                smth_scales = [
                    smth_scales[i]
                    for i, scale in enumerate(truncated_scales)
                    if scale in missing_scales
                ]
                truncated_scales = missing_scales

        snap, snap_header = get_snapshot(snapshot_path)

        extracted_values = extract_simulation_params(snapshot_header=snap_header)

        box_size = int(extracted_values["boxsize"])

        save_simulation_properties(
            snapshot_header=snap_header,
            snapshot=snap,
            output_dir=save_path,
        )

        save_particle_positions_and_velocities(snapshot=snap)

        rho = get_density_field(snapshot=snap, mas=mas, verbose=True)

        smoothened_rho = get_smoothed_field(
            input_field=rho,
            smoothing_scales=smth_scales,
            truncated_scales=truncated_scales,
        )

        plot_density_field(
            input_field=smoothened_rho,
            smoothing_scales=smth_scales,
            truncated_scales=truncated_scales,
            projection="xy",  # Specify the projection of the box for classification overlay
            slice_thickness=[
                100,
                grid_size // 2,
            ],  # Specify the range of slices to average over [Start index, End index]
            save_path=save_path,
        )

        get_tidal_tensor(smoothed_density_field=smoothened_rho)

        logging.info("All calculations are done. Exiting the program...")

    except Exception as e:
        logging.error(f"An error occurred: {e}")

# Stop the timer
end_time = time.time()
logging.info(f"Time taken to run the complete script: {end_time - start_time} seconds")


# ----------------------------------------------END-OF-THE-SCRIPT------------------------------------------------------------#
