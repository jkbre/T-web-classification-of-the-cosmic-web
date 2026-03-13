"""
Tweb_Structure_Classifier.py
-------------------------------

Author: Asit Dave
Date: 31-05-2024
License: MIT

Description:
    This script classifies the structures based on T-web classification for the given tidal shear tensor(s).
    The tidal shear tensor files are loaded and the eigenvalues and eigenvectors are calculated for respective
    smoothing scales. The structures are classified based on T-web classification and the classification matrices,
    depicting the Large-scale structure environment of the voxels, are saved.

Instructions:
    1. Make sure the config/input_params.txt and LSS_TWeb_BlackBox.py files are present in the current working directory.
    2. Make sure the Tidal_Field_Calculator.py script is run before running this script.
    3. Make sure to change the smoothing scales in config/input_params.txt as per the requirement before running this script.
    4. The script assumes that you have not changed the default directory structure of the output files.


"""

# ----------------------------------------- IMPORT LIBRARIES ----------------------------------------------#

from tqdm import tqdm

import os
import time
import logging

from LSS_TWeb_BlackBox import *


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Start the timer
start_time = time.time()

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


# ------------------------------ LOAD THE TIDAL SHEAR TENSOR FILES --------------------------------#


def load_tidal_shear_files() -> list[np.ndarray]:
    """
    Load the tidal shear files from the specified directory.

    Raises:
    - ValueError: If the tidal tensor files do not exist.

    Returns:
    - list[np.ndarray]: A list of loaded tidal shear files corresponding to the given input smoothing scales.
    """
    try:
        tidal_path = os.path.join(save_path, "tidal_fields")

        if not os.path.exists(tidal_path):
            logging.error(
                "The tidal tensor files do not exist. Please run the LSS_Tidal_shear.py script first."
                "Exiting the program...."
            )
            raise
            exit()

        logging.info("Loading the tidal tensor files...")
        tidal_shears = load_all_npy_files(
            folder_path=tidal_path,
            filename_prefix="tidal_tensor_",
            str_smoothing_scales=truncated_scales,
        )
        logging.info("Tidal tensor files loaded successfully.")

        return tidal_shears

    except Exception as e:
        logging.error(f"Error loading tidal shear files: {e}")
        raise


def load_filtered_tidal_shear_files(scales_to_load: list[str]) -> list[np.ndarray]:
    """
    Load specific tidal shear files for given scales.

    Parameters:
    - scales_to_load (list[str]): List of scales to load files for.

    Returns:
    - list[np.ndarray]: A list of loaded tidal shear files for the specified scales.
    """
    try:
        tidal_path = os.path.join(save_path, "tidal_fields")

        if not os.path.exists(tidal_path):
            logging.error(
                "The tidal tensor files do not exist. Please run the LSS_Tidal_shear.py script first."
                "Exiting the program...."
            )
            raise
            exit()

        logging.info(f"Loading tidal tensor files for scales: {scales_to_load}")
        tidal_shears = []
        for scale in scales_to_load:
            file_path = os.path.join(tidal_path, f"tidal_tensor_{scale}.npy")
            if os.path.exists(file_path):
                tidal_shears.append(load_data(file_path))
            else:
                logging.error(
                    f"Tidal tensor file not found for scale {scale}: {file_path}"
                )
                raise FileNotFoundError(
                    f"Required tidal tensor file missing: {file_path}"
                )

        logging.info("Tidal tensor files loaded successfully.")
        return tidal_shears

    except Exception as e:
        logging.error(f"Error loading tidal shear files: {e}")
        raise


# ------------------------------ CALCULATE EIGENVALUES AND EIGENVECTORS OF TIDAL SHEAR TENSOR --------------------------------#


def calculate_tidal_eigenvalues_and_eigenvectors(
    tidal_shears: list[np.ndarray], truncated_scales: list[str]
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    Calculate and save the eigenvalues and eigenvectors of the tidal shear tensor for each smoothing scale.

    Parameters:
    - tidal_shears (list of np.ndarray): List of tidal shear tensors for different smoothing scales.
    - truncated_scales (list of str): List of truncated smoothing scale strings.

    Returns:
    - tuple: A tuple containing dictionaries of eigenvalues and eigenvectors for each smoothing scale.
    """
    try:
        # Create a directory to save the eigenvalues and eigenvectors
        new_directory_name = "Tidal_eigenvalues_and_eigenvectors"
        new_directory_path = os.path.join(save_path, new_directory_name)
        create_directory(new_directory_path, overwrite=False)

        all_tidal_eigenvalues = {}
        all_tidal_eigenvectors = {}

        logging.info(
            "Calculating the eigenvalues and eigenvectors of the tidal shear tensor..."
        )

        for i, sm_scale in enumerate(tqdm(truncated_scales)):
            save_path_eigenvalues = os.path.join(
                new_directory_path, f"tidal_eigenvalues_{sm_scale}.npy"
            )
            save_path_eigenvectors = os.path.join(
                new_directory_path, f"tidal_eigenvectors_{sm_scale}.npy"
            )

            # Check if files already exist
            if os.path.exists(save_path_eigenvalues) and os.path.exists(
                save_path_eigenvectors
            ):
                logging.info(
                    f"Loading existing eigenvalues and eigenvectors for scale {sm_scale}"
                )
                eigenvalues = load_data(save_path_eigenvalues)
                eigenvectors = load_data(save_path_eigenvectors)
            else:
                logging.info(
                    f"Computing eigenvalues and eigenvectors for scale {sm_scale}"
                )
                eigenvalues, eigenvectors = calculate_eigenvalues_and_vectors(
                    tidal_shear_tensor=tidal_shears[i], grid_size=grid_size
                )
                save_data(data=eigenvalues, file_path=save_path_eigenvalues)
                save_data(data=eigenvectors, file_path=save_path_eigenvectors)

            all_tidal_eigenvalues[sm_scale] = eigenvalues
            all_tidal_eigenvectors[sm_scale] = eigenvectors

        logging.info("Eigenvalues and eigenvectors saved successfully.")

        logging.info(
            "Note: The eigenvalues are sorted in the following order: [λ1, λ2, λ3] where λ1 > λ2 > λ3. "
            "And therefore, their corresponding eigenvectors are sorted likewise."
        )

        return all_tidal_eigenvalues, all_tidal_eigenvectors

    except Exception as e:
        logging.error(f"Error calculating tidal eigenvalues and eigenvectors: {e}")
        raise


# ------------------------------ T WEB CLASSIFICATION --------------------------------#


def classify_structures(
    all_tidal_eigenvalues: dict[str, np.ndarray], lambda_th: float
) -> None:
    """
    Classify structures based on the T-web classification scheme using the tidal shear tensor eigenvalues.

    Parameters:
    - all_tidal_eigenvalues (dict[str, np.ndarray]): Dictionary containing eigenvalues for each smoothing scale.
    - save_path (str): Directory path to save the output files.

    Returns:
    - None
    """
    lambda_th_str = str(lambda_th).replace(".", "p")
    try:
        # Create a directory to save the classification matrices
        new_directory_path = os.path.join(save_path, "Classification_matrices")
        create_directory(new_directory_path, overwrite=False)

        logging.info("Classifying the structures based on T-web classification...")

        for scale, eigenvalues in all_tidal_eigenvalues.items():
            save_path_classification = os.path.join(
                new_directory_path, f"classification_matrix_{scale}_{lambda_th_str}.npy"
            )

            # Check if classification file already exists
            if os.path.exists(save_path_classification):
                logging.info(
                    f"Classification matrix already exists for scale {scale} and λ_th={lambda_th}, skipping..."
                )
                continue

            logging.info(
                f"Computing classification matrix for scale {scale} and λ_th={lambda_th}"
            )
            classification_matrix = classify_structure(
                eigenvalues=eigenvalues, lambda_th=lambda_th
            )
            save_data(data=classification_matrix, file_path=save_path_classification)

        logging.info("Structures classified successfully.")

        logging.info(
            "Note: The classification matrices consist of the following values:\n"
            "0: Void\n"
            "1: Sheet\n"
            "2: Filament\n"
            "3: Cluster\n"
        )

    except Exception as e:
        logging.error(f"Error classifying structures: {e}")
        raise


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
        "--force", action="store_true", help="Force recalculation even if files exist"
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

        smth_scales, truncated_scales = extract_smoothing_scales(smoothing_scales)

        # Check if files already exist (unless forced recalculation)
        if not force_recalc:
            scale_status = check_classification_files_exist(
                save_path, truncated_scales, lambda_th
            )
            missing_scales = get_missing_scales(scale_status)

            if not missing_scales:
                logging.info(
                    "All classification files already exist. Skipping computation."
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

                # Load only the tidal shear files for missing scales
                tidal_shears = load_filtered_tidal_shear_files(missing_scales)
        else:
            # Load all tidal shear files when forcing recalculation
            tidal_shears = load_tidal_shear_files()

        all_tidal_eigenvalues, all_tidal_eigenvectors = (
            calculate_tidal_eigenvalues_and_eigenvectors(
                tidal_shears=tidal_shears, truncated_scales=truncated_scales
            )
        )

        classify_structures(
            all_tidal_eigenvalues=all_tidal_eigenvalues, lambda_th=lambda_th
        )

        logging.info("All calculations are done. Exiting the program...")

    except Exception as e:
        logging.error(f"An error occurred: {e}")


# Stop the timer
end_time = time.time()
logging.info(f"Time taken to run the complete script: {end_time - start_time} seconds")


# ----------------------------------------------END-OF-THE-SCRIPT------------------------------------------------------------#
