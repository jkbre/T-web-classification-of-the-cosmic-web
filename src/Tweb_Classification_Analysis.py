"""
Tweb_Classification_Analysis.py
-------------------------------

Author: Asit Dave
Date: 31-05-2024
License: MIT

Description:
    This script analyses the classification of structures based on the T-web classification. The volume
    fractions of different structures are plotted against the smoothing scales. The classification overlay
    on the density field is plotted for the given smoothing scales. The changes in the classification of
    particles are also plotted if the smoothing scales are changed.

Instructions:
    1. Make sure config/input_params.txt and LSS_TWeb_BlackBox.py are present in the current working directory.
    2. Make sure the Tidal_Field_Calculator.py & Tweb_Structure_Classifier.py script is ran before running this script.
    3. Parameters that can be modified in this script are:
        - Slice index for the structure classification (default: middle slice)
        - Projection of the box for the plots (default: xy)
        - Thickness of the slice (to average the density field over) for the classification
          overlay plots (default: half box thickness)
     4. Make sure to change the smoothing scales in config/input_params.txt as per the requirement before running this script.
         The code will perform calculations for the smoothing scales specified in config/input_params.txt.
    5. The script assumes that you have not changed the default directory structure of the output files.

"""

# ----------------------------------------- IMPORT LIBRARIES ----------------------------------------------#

import numpy as np

import time
import os
import logging

from LSS_TWeb_BlackBox import *


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Start the timer
start_time = time.time()

# ----------------------------------------- READ FILE TO GET THE BOX SIZE ----------------------------------------------#


def get_box_size(output_dir: str) -> int:
    """
    Extract the box size from the 'simulation_properties.txt' file.

    Returns:
    - int: The extracted box size.

    Raises:
    - FileNotFoundError: If 'simulation_properties.txt' does not exist in the output or current directory.
    - ValueError: If the box size cannot be found or parsed correctly.
    """
    try:
        preferred_path = os.path.join(output_dir, "simulation_properties.txt")
        fallback_path = "simulation_properties.txt"
        file_path = preferred_path if os.path.exists(preferred_path) else fallback_path

        with open(file_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            if line.startswith("Box size"):
                box_size = int(float(line.split(":")[1].strip().split(" ")[0]))
                return box_size

        raise logging.error(
            "Box size not found in the 'simulation_properties.txt' file."
        )

    except FileNotFoundError as e:
        logging.error(
            f"'simulation_properties.txt' file does not exist in output directory '{output_dir}' or current working directory: {e}"
        )
        raise

    except ValueError as e:
        logging.error(f"Error in parsing box size: {e}")
        raise


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


# ------------------------------ LOAD THE DENSITY FIELD FILES --------------------------------#


def load_density_fields() -> list[np.ndarray]:
    """
    Load smoothed density fields from the specified directory.

    Returns:
    - np.ndarray: An list containing all loaded smoothed density fields corresponding to the given input smoothing scales.
    """
    logging.info("Loading the smoothed density field files....")

    density_fields = load_all_npy_files(
        folder_path=os.path.join(save_path, "smoothed_density_fields"),
        filename_prefix="smoothed_density_field_",
        str_smoothing_scales=truncated_scales,
    )

    logging.info("Smoothed density field files loaded successfully.\n")

    return density_fields


# ------------------------------ LOAD THE CLASSIFICATION MATRICES --------------------------------#


def load_classification_matrices(lambda_th: float) -> list[np.ndarray]:
    """
    Load classification matrices from the specified directory.

    Returns:
    - np.ndarray: An list containing all loaded classification matrices corresponding to the given input smoothing scales.
    """
    classification_path = os.path.join(save_path, "Classification_matrices")

    logging.info("Loading the classification matrix files....")

    classifications = load_all_npy_files(
        folder_path=classification_path,
        filename_prefix="classification_matrix_",
        str_smoothing_scales=truncated_scales,
        lambda_th=lambda_th,
    )
    total_files = len(classifications)

    logging.info(f"Number of files found: {total_files}")
    logging.info("Classification matrix files loaded successfully.\n")

    return classifications


# ------------------------------ CREATE A DIRECTORY FOR PLOTS --------------------------------#


def create_directory_for_plots() -> str:
    """
    Create a directory to save the classification analysis plots.

    Returns:
    - str: The path to the created or existing directory for classification analysis plots.
    """
    new_directory_name = "Classification analysis plots"
    new_directory_path = os.path.join(save_path, new_directory_name)
    create_directory(new_directory_path, overwrite=False)

    return new_directory_path


# ------------------------------ PLOT VOLUME FRACTIONS VS RS --------------------------------#


def plot_volfrac_vs_rs(
    classifications: list[np.ndarray], smooth_scales: list[float], dir_path: str
):
    """
    Plot the volume fractions vs Smoothing scale (Rs).

    Parameters:
    - classifications (np.ndarray): An array containing the classification matrices.
    - smooth_scales (list[float]): List of smoothing scales.
    - dir_path (str): The directory path to save the plot.

    Returns:
    - None
    """
    plot_volfrac_rs(
        classifications=classifications,
        smooth_scales=smooth_scales,
        save_path=os.path.join(dir_path, "Volume_fractions_vs_Rs.png"),
    )


# ------------------------------ PLOT CLASSIFICATION OVERLAY ON DENSITY FIELD --------------------------------#


def plot_structure_classification(
    classifications: np.ndarray,
    density_fields: np.ndarray,
    smooth_scales: list[float],
    truncated_scales: list[str],
    dir_path: str,
    slice_thickness: list[int, int],
    slice_index: int,
    projection: str,
    lambda_th: float = None,
):
    """
    Plot the structure classification overlay on the density field for all smoothing scales.

    This function plots the classification overlay on the density field for each smoothing scale.
    It saves the plots with filenames indicating the corresponding smoothing scales in the specified directory path.

    Parameters:
    - classifications (np.ndarray): An array containing the classification matrices for all smoothing scales.
    - density_fields (np.ndarray): An array containing the density fields for all smoothing scales.
    - smooth_scales (list[float]): List of smoothing scales.
    - truncated_scales (list[str]): List of truncated smoothing scale names.
    - dir_path (str): The directory path to save the plots.
    - slice_thickness (list[int, int]): The thickness of the slice for the classification overlay plots. [start_index, end_index]
    - slice_index (int): The index of the slice for the classification overlay plots.
    - projection (str): The projection of the box for the classification overlay plots ('xy', 'yz', or 'zx').

    Returns:
    - None
    """
    # Plot the classification overlay on density field for all smoothing scales
    lambda_th_str = (
        str(lambda_th).replace(".", "p") if lambda_th is not None else "default"
    )

    for i in range(len(smooth_scales)):
        plot_classification_overlay(
            smth_scale=smooth_scales[i],
            lambda_th=lambda_th,
            classification_matrix=classifications[i],
            smoothed_rho=density_fields[i],
            slice_thickness=slice_thickness,
            slice_index=slice_index,
            projection=projection,
            grid_size=grid_size,
            box_size=box_size,
            save_path=os.path.join(
                dir_path,
                f"Classification_overlay_{truncated_scales[i]}_{lambda_th_str}_{projection}.png",
            ),
        )


# ------------------------------ PLOT CLASSIFICATION OVERLAY ON DENSITY FIELD --------------------------------#


def plot_all_environments_overlay(
    classifications: np.ndarray,
    density_fields: np.ndarray,
    smooth_scales: list[float],
    truncated_scales: list[str],
    slice_thickness: list[int, int],
    projection: str,
    slice_index: int,
    dir_path: str,
    lambda_th: float = None,
):
    """
    Plot and save overlay images of all environments on density fields for multiple smoothing scales.

    Parameters:
    - classifications (np.ndarray): Array of 3D classification matrices for different smoothing scales.
                                    Shape (n_scales, grid_size, grid_size, grid_size).
    - density_fields (np.ndarray): Array of 3D smoothed density fields for different smoothing scales.
                                   Shape (n_scales, grid_size, grid_size, grid_size).
    - smooth_scales (list[float]): List of smoothing scales used for generating the classifications and density fields.
    - truncated_scales (list[str]): List of truncated string representations of smoothing scales for filenames.
    - slice_thickness (list[int, int]): List containing the start and end indices defining the slice thickness.
    - projection (str): Axis along which the slice is taken ('xy', 'yz', or 'zx').
    - slice_index (int): Index of the slice to be taken along the specified projection axis.
    - dir_path (str): Directory path where the resulting overlay images will be saved.

    Returns:
    - None: This function saves the overlay images to the specified directory.

    The function iterates over each smoothing scale, generating and saving an overlay image for each one.
    It calls the overlay_all_envs function to create the overlay for each scale.
    """

    lambda_th_str = (
        str(lambda_th).replace(".", "p") if lambda_th is not None else "default"
    )

    for i in range(len(smooth_scales)):
        overlay_all_envs(
            classification_matrix=classifications[i],
            lambda_th=lambda_th,
            smoothed_rho=density_fields[i],
            slice_thickness=slice_thickness,
            smoothing_scale=smooth_scales[i],
            slice_index=slice_index,
            projection=projection,
            grid_size=grid_size,
            box_size=box_size,
            save_path=os.path.join(
                dir_path,
                f"All_Environments_Overlay_{truncated_scales[i]}_{lambda_th_str}_{projection}.png",
            ),
        )


# ------------------------------ GET STRUCTURE CHANGES --------------------------------#


def get_structure_changes(
    classifications: np.ndarray, slice_index: int, projection: str
) -> dict[str, np.ndarray]:
    """
    Get structure changes between two classification matrices.

    This function calculates the structure changes between two classification matrices, typically representing the beginning
    and end of a simulation. It returns a dictionary containing the changes in the cosmic web structures.

    Parameters:
    - classifications (np.ndarray): An array containing the classification matrices for different time steps.
    - slice_index (int): The index of the slice to analyze for structure changes.
    - projection (str): The projection type ('xy', 'yz', 'xz') to analyze for structure changes.

    Returns:
    - dict: A dictionary containing the changes in the cosmic web structures.
    """
    # Get all environment changes
    env_changes = get_env_changes(
        clf1=classifications[0],
        clf2=classifications[-1],
        slice_index=slice_index,
        projection=projection,
        grid_size=grid_size,
        box_size=box_size,
    )

    return env_changes


# ------------------------------ GENERATE TRANSFORMATION DICTIONARY --------------------------------#


def generate_transformation_dictionary(
    env_changes: dict,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    Generate two transformation dictionaries based on environmental changes.

    Parameters:
    - env_changes (dict): A dictionary containing environmental changes.

    Returns:
    - tuple: A tuple containing two dictionaries representing transformations.
             Each dictionary has string keys indicating the type of transformation
             and NumPy arrays as values representing the changed environments.
    """
    # Generate the transformation dictionary
    transformations_1 = {
        "Void to Sheet": env_changes["Void to Sheet"],
        "Sheet to Filament": env_changes["Sheet to Filament"],
        "Filament to Node": env_changes["Filament to Node"],
        "Sheet to Void": env_changes["Sheet to Void"],
        "Filament to Sheet": env_changes["Filament to Sheet"],
        "Node to Filament": env_changes["Node to Filament"],
    }

    # Plot the second set of transformations
    transformations_2 = {
        "Void to Filament": env_changes["Void to Filament"],
        "Void to Node": env_changes["Void to Node"],
        "Sheet to Node": env_changes["Sheet to Node"],
        "Filament to Void": env_changes["Filament to Void"],
        "Node to Void": env_changes["Node to Void"],
        "Node to Sheet": env_changes["Node to Sheet"],
    }

    return transformations_1, transformations_2


# ------------------------------ PLOT STRUCTURE CHANGES --------------------------------#


def plot_structure_changes(
    transformations: dict[str, np.ndarray],
    density_field: np.ndarray,
    truncated_scales: list[str],
    dir_path: str,
    slice_thickness: list[int, int],
    projection: str,
    unique_num: int,
) -> None:
    """
    Plot the changes in classification of halo environments for given transformations.

    Parameters:
    - transformations (dict): A dictionary containing environmental changes.
    - density_fields (np.ndarray): An array containing density fields.
    - truncated_scales (list[str]): A list of truncated smoothing scale names (without decimals).
    - dir_path (str): The directory path to save the plots.
    - slice_thickness (list[int, int]): The range of slices to average over [Start index, End index]. (used for density field plot)
    - slice_index (int): The index of the slice to plot. (used for structure classification)
    - projection (str): The projection type ('xy', 'yz', 'xz').
    - unique_num (int): An unique identifier to differentiate between plots.

    Returns:
    - None
    """
    # Plot the changes in classification of Halo environments for above transformations
    plot_env_changes(
        transformations=transformations,
        density_slice=slice_density_field(
            density_field, slice_thickness=slice_thickness, projection=projection
        ),
        box_size=box_size,
        title="Changes in classification of Halo environments",
        save_path=os.path.join(
            dir_path,
            f"Classification_change_{truncated_scales[0]}_{truncated_scales[-1]}_{unique_num}.png",
        ),
    )


# ----------------------------------------- MAIN EXECUTION ----------------------------------------------#

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate TWEB DAVE configuration file"
    )
    parser.add_argument(
        "--lambda_th",
        "-l",
        type=float,
        default=0.0,
        help="Threshold for T-web classification",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force regeneration of plots even if they exist",
    )
    args = parser.parse_args()
    lambda_th = args.lambda_th
    force_recalc = args.force
    try:
        (
            snapshot_path,
            save_path,
            grid_size,
            create_density,
            own_density_path,
            smoothing_scales,
            yn_potential,
            yn_traceless,
        ) = read_input_file("config/input_params.txt")
        box_size = get_box_size(save_path)
        smooth_scales, truncated_scales = extract_smoothing_scales(smoothing_scales)
        # Load the density fields that were saved using LSS_Tidal_shear.py
        smoothed_density_fields = load_density_fields()
        # Load the classification matrices that were saved using LSS_Classification.py
        classifications = load_classification_matrices(lambda_th=lambda_th)
        new_directory_path = create_directory_for_plots()

        # Parameters that can be modified
        slice_index = (
            grid_size // 2
        )  # Specify the slice index you want to consider for structure classification
        projection = (
            "xy"  # Specify the projection to be used. Options: 'xy', 'yz', 'xz'
        )
        slice_thickness = [
            grid_size // 2,
            grid_size // 2 + 1,
        ]  # Specify the range of slices to average the density field over [Start index, End index]

        # Plot the classification overlay on density field for all smoothing scales
        plot_structure_classification(
            classifications,
            smoothed_density_fields,
            smooth_scales,
            truncated_scales,
            new_directory_path,
            slice_thickness,
            slice_index,
            projection,
            lambda_th,
        )

        plot_all_environments_overlay(
            classifications,
            smoothed_density_fields,
            smooth_scales,
            truncated_scales,
            slice_thickness,
            projection,
            slice_index,
            new_directory_path,
            lambda_th,
        )

        if len(smooth_scales) > 1:
            # Plot the volume fractions vs Rs
            plot_volfrac_vs_rs(classifications, smooth_scales, new_directory_path)

            # Get all environment changes
            env_changes = get_structure_changes(
                classifications, slice_index, projection
            )

            # Generate the transformation dictionary
            transformations_1, transformations_2 = generate_transformation_dictionary(
                env_changes
            )

            # Plot the changes in classification of Halo environments for above transformations
            plot_structure_changes(
                transformations_1,
                smoothed_density_fields[len(smoothed_density_fields) // 2],
                truncated_scales,
                new_directory_path,
                slice_thickness,
                projection,
                unique_num=1,
            )

            plot_structure_changes(
                transformations_2,
                smoothed_density_fields[len(smoothed_density_fields) // 2],
                truncated_scales,
                new_directory_path,
                slice_thickness,
                projection,
                unique_num=2,
            )

            logging.info(
                "Note: The filename as 'Classification_change_Rs1_Rs2_1.png' indicates the changes in classification of "
                "Halo environments between two smoothing scales Rs1 and Rs2.\n"
            )

        else:
            logging.info(
                "Volume fraction plot requires more than one smoothing scale to compare."
            )
            logging.info(
                "Skipping the volume fraction plot & changes in structure classification plots....\n"
            )

        logging.info("Execution completed successfully.")

    except Exception as e:
        logging.error(f"An error occurred: {e}")


# Stop the timer
end_time = time.time()
logging.info(f"Time taken to run the complete script: {end_time - start_time} seconds")

# ----------------------------------------------END-OF-THE-SCRIPT------------------------------------------------------------#
