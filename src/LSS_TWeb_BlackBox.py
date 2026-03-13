"""
LSS_TWeb_BlackBox.py
-------------------------------

Author: Asit Dave
Date: 31-05-2024
License: MIT

Description:

    This Python script, LSS_TWeb_BlackBox.py, serves as a comprehensive library designed to facilitate
    various analyses of cosmological simulation data. It provides a suite of functions to compute,
    process, and visualize large-scale structures within the universe, based on density fields and
    tidal tensor calculations.

    Key functionalities include:

      1. Density Field Computation: Extract the density field from cosmological simulation snapshots.
      2. Gaussian Smoothing: Apply Gaussian filtering to smooth the computed density field.
      3. Visualization: Plot and save visual representations of the density field.
      4. Tidal Tensor Calculation: Derive the tidal tensor and potential field from the density field.
      5. Tidal Tensor Traceless Transformation: Modify the tidal tensor to make it traceless.
      6. Eigenvalue and Eigenvector Analysis: Calculate the eigenvalues and eigenvectors of the tidal shear tensor.
      7. Large-scale Structure Classification: Classify large-scale structures based on the T-web classification scheme.
      8. Result Visualization: Overlay and visualize the classification results on the density field.

# Note: All files are saved in the .npy format.

"""

# ----------------------------------------- IMPORT LIBRARIES ----------------------------------------------#

import numpy as np
import pynbody
import MAS_library as MASL
import pynbody.simdict
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
import smoothing_library as SL

import os
import shutil
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# -------------------------------------------------------------------------------------------------------------#


def read_input_file(file_path: str) -> list:
    """
    Reads and parses the input parameter file to extract simulation settings.

    Args:
        file_path (str): Path to the input parameter file.

    Returns:
        list: A list containing the following elements in order:
            - snapshot_path (str): Path to the (pynbody) snapshot file.
            - save_path (str): Path where all the results will be saved.
            - grid_size (int): Grid size for the simulation.
            - create_density (bool): Whether to create the density field.
            - density_path (str or None): Path to load the density field (optional).
            - smoothing_scales (list of float or None): Smoothing scales for the density field.
            - calculate_potential (bool): Whether to calculate the potential field.
            - calculate_traceless_tidal (bool): Whether to calculate the traceless tidal tensor.

    Raises:
        ValueError: If any required path is invalid or a required value is not provided.

    Example:
        >>> params = read_input_file('simulation_params.txt')
        >>> print(params)
        ['/path/to/snapshot', '/path/to/save', 256, True, None, [1.0, 2.0], False, False]
    """

    # Initialize variables to store the parsed values
    snapshot_path = ""
    save_path = ""
    create_density = True
    density_path = None
    smoothing_scales = None
    grid_size = 0
    calculate_potential = False
    calculate_traceless_tidal = False

    # Read the input file and parse the values
    with open(file_path, "r") as file:
        for line in file:
            line = line.strip().split("#")[0]

            if line.startswith("Path to the snapshot file"):
                snapshot_path = line.split(":")[1].strip()

            elif line.startswith("Path where you want to save the results"):
                save_path = line.split(":")[1].strip()

            elif line.startswith("Enter the grid size"):
                grid_size = int(line.split(":")[1].strip())

            elif line.startswith("Create the density field"):
                create_density = line.split(":")[1].strip().lower() == "yes"

            elif line.startswith("Path to load the density field"):
                if ":" in line:
                    density_path = (
                        None
                        if (line.split(":")[1].strip()) == ""
                        else int(line.split(":")[1].strip())
                    )

            elif line.startswith("Smoothing scales"):
                smoothing_scales = [
                    float(x) for x in line.split(":")[1].strip().split()
                ]

            elif line.startswith("Calculate potential field"):
                calculate_potential = line.split(":")[1].strip().lower() == "yes"

            elif line.startswith("Calculate traceless tidal tensor"):
                calculate_traceless_tidal = line.split(":")[1].strip().lower() == "yes"

    # Validate the input parameters
    if density_path is None and not create_density:
        raise ValueError("No density field provided!")

    # Check if snapshot path exists - handle both single files and Gadget multi-file format
    if not os.path.exists(snapshot_path) and not os.path.exists(f"{snapshot_path}.0"):
        raise FileNotFoundError(
            "Please enter a valid snapshot path. File does not exist."
        )

    if not os.path.exists(save_path):
        raise FileNotFoundError(
            "Please enter a valid file path to save the result. File does not exist."
        )

    if grid_size <= 0:
        raise ValueError("Please enter a positive value.")

    # Final validation - check paths exist (handle Gadget multi-file format for snapshot)
    if not os.path.exists(snapshot_path) and not os.path.exists(f"{snapshot_path}.0"):
        raise FileNotFoundError(
            f"Please enter a valid snapshot path. {snapshot_path} does not exist."
        )

    if not os.path.exists(save_path):
        raise FileNotFoundError(
            f"Please enter a valid save path. {save_path} does not exist."
        )

    if density_path is not None and not os.path.exists(density_path):
        raise FileNotFoundError(
            f"Please enter a valid path. {density_path} does not exist."
        )

    return [
        snapshot_path,
        save_path,
        grid_size,
        create_density,
        density_path,
        smoothing_scales,
        calculate_potential,
        calculate_traceless_tidal,
    ]


# -------------------------------------------------------------------------------------------------------------#


def load_snapshot(
    snapshot_path: str,
) -> tuple[pynbody.snapshot.SimSnap, pynbody.simdict.SimDict]:
    """
    Load the cosmological simulation snapshot and extract its properties.

    Parameters:
      - snapshot_path: str
        The path to the snapshot file.

    Returns:
      - snap: pynbody.snapshot.SimSnap object
        The loaded cosmological simulation snapshot.
      - header: pynbody.simdict.SimDict object
        The properties of the simulation snapshot.
    """

    snap = pynbody.load(snapshot_path)

    header = snap.properties

    return snap, header


# -------------------------------------------------------------------------------------------------------------#


def extract_simdict_values(simdict: pynbody.simdict.SimDict) -> dict[str, float]:
    """
    Extracts values from a pynbody SimDict in float format.

    Args:
        simdict: A pynbody.simdict.SimDict object.

    Returns:
        A dictionary containing the following keys and values in float format:
            boxsize: (float) Box size in [Mpc a / h]
            time: (float) Time in [s Mpc a**1/2 h**-1 km**-1]
            a: (float) Scale factor
            h: (float) Hubble parameter
            omega_m: (float) Matter density parameter
            omega_l: (float) Dark energy density parameter
    """

    results = {}
    for key, value in simdict.items():
        if key in ["omegaM0", "omegaL0", "a", "h"]:
            results[key] = float(value)
        else:
            results[key] = float(str(value).split()[0])

    return results


# -------------------------------------------------------------------------------------------------------------#


def extract_scales(input_scales: list) -> tuple[list[float], list[str]]:
    """
    Extracts and processes scales from a list of input smoothing scales.

    Parameters:
    - input_scales: list
      A list containing space-separated scales to be extracted and processed.

    Returns:
    - sm_scales: list
      A list of smoothing scales converted to float.
    - truncated_scales: list
      A list of smoothing scales with decimal points removed.
    """

    # Sort the input scales in ascending order
    scales = sorted(input_scales)

    # Initialize lists to store processed scales
    truncated_scales = []
    sm_scales = []

    # Process each scale
    for scale in scales:
        # Convert scale to float and append to sm_scales
        sm_scales.append(float(scale))

        # Remove decimal points from scale and append to truncated_scales
        str_split = str(scale).split(".")
        truncated_scales.append("".join(str_num for str_num in str_split))

    return sm_scales, truncated_scales


# -------------------------------------------------------------------------------------------------------------#


def load_data(file_path: str) -> np.ndarray:
    """
    Load the data from the specified file path.

    Parameters:
    - file_path: str
      The file path where the data is stored.

    Returns:
    - data: numpy.ndarray
      The loaded data.
    """

    try:
        loaded_data = np.load(file_path)

    except ImportError as e:
        logging.error(
            "Error loading the data from the file path. Make sure that the file type is .npy.\n"
            f"Error message: {e}"
        )

    return loaded_data


# -------------------------------------------------------------------------------------------------------------#


def create_directory(directory_path: str, overwrite: bool) -> None:
    """
    Creates a directory, overwriting any existing directory with the same name.

    Args:
        directory_path: str
            Path to the directory to create.
        overwrite: bool
            Whether to overwrite the directory if it already exists.

    Returns:
        None
    """

    if os.path.exists(directory_path) and os.path.isdir(directory_path):

        if overwrite:
            logging.warning(
                f"Warning: Overwriting existing directory: {directory_path}\n"
            )

            # Overwrite existing directory if it's a directory
            shutil.rmtree(directory_path)  # Remove the existing directory
            os.makedirs(directory_path)  # Create the directory again

        else:
            logging.warning(
                f"Warning: Directory already exists: {directory_path}\n"
                "The files will continue to be saved in the existing directory.\n"
            )

    else:
        os.makedirs(directory_path)  # Attempt to create directories recursively


# -------------------------------------------------------------------------------------------------------------#


def save_data(data: np.ndarray, file_path: str) -> None:
    """
    Save the given data to the specified file path.

    Parameters:
    - data: numpy.ndarray
      The data to be saved.
    - file_path: str
      The file path where the data will be saved.
    """

    np.save(file_path, data)

    return None


# -------------------------------------------------------------------------------------------------------------#


def check_tidal_field_files_exist(
    save_path: str,
    truncated_scales: list[str],
    calculate_potential: bool,
    calculate_traceless: bool,
) -> dict[str, bool]:
    """
    Check if tidal field calculation output files already exist for given parameters.

    Parameters:
    - save_path: str
      The base path where results are saved.
    - truncated_scales: list[str]
      List of truncated smoothing scale strings.
    - calculate_potential: bool
      Whether potential field files should exist.
    - calculate_traceless: bool
      Whether traceless tidal shear files should exist.

    Returns:
    - dict[str, bool]: Dictionary mapping scale to whether all required files exist for that scale.
    """
    scale_status = {}

    for scale in truncated_scales:
        files_exist = True

        # Check core files that are always generated
        required_files = [
            os.path.join(save_path, "particle_positions.npy"),
            os.path.join(save_path, "particle_velocity.npy"),
            os.path.join(save_path, "density_field.npy"),
            os.path.join(
                save_path,
                "smoothed_density_fields",
                f"smoothed_density_field_{scale}.npy",
            ),
            os.path.join(save_path, "tidal_fields", f"tidal_tensor_{scale}.npy"),
        ]

        # Add conditional files
        if calculate_potential:
            required_files.append(
                os.path.join(
                    save_path, "potential_field", f"potential_field_{scale}.npy"
                )
            )

        if calculate_traceless:
            required_files.append(
                os.path.join(
                    save_path, "tidal_fields", f"traceless_tidal_shear_{scale}.npy"
                )
            )

        # Check if all required files exist
        for file_path in required_files:
            if not os.path.exists(file_path):
                files_exist = False
                break

        scale_status[scale] = files_exist

    return scale_status


def check_classification_files_exist(
    save_path: str, truncated_scales: list[str], lambda_th: float
) -> dict[str, bool]:
    """
    Check if structure classification output files already exist for given parameters.

    Parameters:
    - save_path: str
      The base path where results are saved.
    - truncated_scales: list[str]
      List of truncated smoothing scale strings.
    - lambda_th: float
      Lambda threshold value for classification.

    Returns:
    - dict[str, bool]: Dictionary mapping scale to whether all required files exist for that scale.
    """
    scale_status = {}
    lambda_th_str = str(lambda_th).replace(".", "p")

    for scale in truncated_scales:
        files_exist = True

        # Check required files for this scale
        required_files = [
            os.path.join(
                save_path,
                "Tidal_eigenvalues_and_eigenvectors",
                f"tidal_eigenvalues_{scale}.npy",
            ),
            os.path.join(
                save_path,
                "Tidal_eigenvalues_and_eigenvectors",
                f"tidal_eigenvectors_{scale}.npy",
            ),
            os.path.join(
                save_path,
                "Classification_matrices",
                f"classification_matrix_{scale}_{lambda_th_str}.npy",
            ),
        ]

        # Check if all required files exist
        for file_path in required_files:
            if not os.path.exists(file_path):
                files_exist = False
                break

        scale_status[scale] = files_exist

    return scale_status


def get_missing_scales(scale_status: dict[str, bool]) -> list[str]:
    """
    Get list of scales for which files are missing.

    Parameters:
    - scale_status: dict[str, bool]
      Dictionary mapping scale to file existence status.

    Returns:
    - list[str]: List of scales with missing files.
    """
    return [scale for scale, exists in scale_status.items() if not exists]


# -------------------------------------------------------------------------------------------------------------#


def compute_density_field(
    snapshot: pynbody.snapshot.SimSnap,
    grid_size: int,
    box_size: int,
    mas: str,
    verbose: bool,
) -> np.ndarray:
    """
    Compute the density field from a cosmological simulation snapshot.

    Parameters:
      - snapshot: pynbody.snapshot.SimSnap object
        The cosmological simulation snapshot.
      - grid_size: int
        The size of the 3D grid.
      - box_size: int
        The size of the simulation box in [Mpc a / h].
      - mas: str
        The mass-assignment scheme. Options: 'NGP', 'CIC', 'TSC', 'PCS', 'gaussian'.
      - verbose: bool
        Print information on progress.

    Returns:
      - delta: numpy.ndarray
        The computed density field representing density contrast in each voxel.
    """

    pos = snapshot["pos"]  # Position of particles

    # Construct a 3D grid for the density field
    delta = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)
    MASL.MA(pos, delta, box_size, mas, verbose=verbose)  # Generate the density field

    # Calculate the density contrast
    delta /= np.mean(delta, dtype=np.float64)
    delta -= 1.0

    return delta


# -------------------------------------------------------------------------------------------------------------#


def smooth_field(
    input_field: np.ndarray, smoothing_scale: float, box_size: int, grid_size: int
) -> np.ndarray:
    """
    Smooth the given field using Gaussian filtering.

    Parameters:
      - input_field: numpy.ndarray
        The field to be smoothed.
      - smoothing_scale: float
        The scale of Gaussian smoothing.
      - box_size: int
        The size of the simulation box in [Mpc a/h].
      - grid_size: int
        The size of the 3D grid.

    Returns:
      - smoothed_field: numpy.ndarray
        The smoothed field.
    """

    # --------------------------- Uncomment the following code to use the smoothing_library --------------------------- #
    # Filter  = 'Gaussian'
    # threads = 16

    # # compute FFT of the filter
    # W_k = SL.FT_filter(box_size, smoothing_scale, grid_size, Filter, threads)

    # # smooth the field
    # smoothed_density = SL.field_smoothing(input_field, W_k, threads)

    # ------------------------------------------------------------------------------------------------------------------ #
    sigma = smoothing_scale
    pixel_scale = box_size / grid_size
    sigma_pixels = sigma / pixel_scale

    # # mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}, optional
    smoothed_density = gaussian_filter(
        input_field, sigma=sigma_pixels, mode="wrap", cval=0.0
    )

    return smoothed_density


# -------------------------------------------------------------------------------------------------------------#


def plot_field(
    input_field: np.ndarray,
    sm_scale: float,
    projection: str,
    slice_index: list[int, int],
    name_sm_scale: str,
    filepath: str,
) -> None:
    """
    Plot the given field.

    Parameters:
    - input_field: numpy.ndarray
      The field to be plotted.
    - sm_scale: float
      The smoothing scale.
    - name_sm_scale: str
      Smoothing scale for the plot (used for saving the plot).
    - projection: str
      The dimension of the field ('xy', 'yz', 'zx').
    - slice_index: list
      The slice indices for the plot: [start_index, end_index].
    - filepath: str
      The file path where the plot will be saved.
    """

    eps = 1e-15  # so that log doesn't get a value 0
    N_start = slice_index[0]
    N_end = slice_index[1]

    # Plot for zero smoothening density field
    plt.figure(figsize=(2, 2), dpi=200)
    delplot1 = np.log10(input_field + 1 + eps)

    if projection == ("yz" or "zy"):
        slic1 = np.mean(delplot1[N_start:N_end, :, :], axis=0)

    if projection == ("xz" or "zx"):
        slic1 = np.mean(delplot1[:, N_start:N_end, :], axis=1)

    if projection == ("xy" or "yx"):
        slic1 = np.mean(delplot1[:, :, N_start:N_end], axis=2)

    plt.imshow(slic1, cmap="inferno")
    plt.axis("off")
    plt.title(r"$R_s = $" + f"{sm_scale}" + r"$~h^{-1} Mpc$")

    save_path = os.path.join(
        filepath, f"density_field_{projection}_plane_{name_sm_scale}.png"
    )
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)


# -------------------------------------------------------------------------------------------------------------#


def calculate_tidal_tensor(density_field: np.ndarray, calculate_potential: bool):
    """
    Calculate the tidal tensor from the given density field.

    Parameters:
    - density_field: numpy.ndarray
      The density field.
    - calculate_potential: bool
      Calculate the potential field.

    Returns:
    - tidal_tensor: numpy.ndarray
      Tidal tensor (3x3 matrix) for each voxel in the simulation box.
      Shape: (grid_size, grid_size, grid_size, 3, 3).
    - (optional) potential: numpy.ndarray
      The gravitational potential field. Only returned if calculate_potential=True.
      Shape: (grid_size, grid_size, grid_size).
    """

    # Fast Fourier Transform of the density field
    density_field_fft = np.fft.fftn(density_field)
    shape = density_field.shape[0]

    # Generate the k-space grid
    k_modes = np.fft.fftfreq(shape)
    kx, ky, kz = np.meshgrid(k_modes, k_modes, k_modes, indexing="ij")

    # Calculate the square of k
    k_sq = kx**2 + ky**2 + kz**2

    # Calculate the tidal tensor in Fourier space
    with np.errstate(divide="ignore", invalid="ignore"):
        potential_k = -np.divide(density_field_fft, k_sq, where=k_sq != 0)

    # Calculate the potential if required
    if calculate_potential:
        potential = np.fft.ifftn(potential_k).real

    # Calculate the components of the tidal tensor in Fourier space
    tk_components = -potential_k * np.array(
        [kx**2, kx * ky, kx * kz, ky**2, ky * kz, kz**2]
    )

    # Inverse Fourier Transform to get the tidal tensor components
    tidal_tensor_components = np.fft.ifftn(tk_components, axes=(1, 2, 3))

    # Assemble the tensor field (assigning the values to symmetric counterparts)
    tidal_tensor = np.zeros((3, 3) + density_field.shape, dtype=np.float64)
    tidal_tensor[0, 0, ...] = tidal_tensor_components[0].real
    tidal_tensor[1, 0, ...] = tidal_tensor_components[1].real
    tidal_tensor[0, 1, ...] = tidal_tensor_components[1].real
    tidal_tensor[1, 1, ...] = tidal_tensor_components[3].real
    tidal_tensor[0, 2, ...] = tidal_tensor_components[2].real
    tidal_tensor[2, 0, ...] = tidal_tensor_components[2].real
    tidal_tensor[2, 2, ...] = tidal_tensor_components[5].real
    tidal_tensor[1, 2, ...] = tidal_tensor_components[4].real
    tidal_tensor[2, 1, ...] = tidal_tensor_components[4].real

    if calculate_potential:
        return tidal_tensor.transpose(2, 3, 4, 0, 1), potential

    else:
        return tidal_tensor.transpose(2, 3, 4, 0, 1)


# -------------------------------------------------------------------------------------------------------------#


def make_traceless(matrix: np.ndarray) -> np.ndarray:
    """
    Make the given matrix traceless.

    Parameters:
    - matrix: numpy.ndarray
      The matrix to be made traceless.

    Returns:
    - traceless_matrix: numpy.ndarray
      The traceless matrix.
    """
    traceless_matrix = matrix - (np.trace(matrix) / 3) * np.identity(3)

    return traceless_matrix


# -------------------------------------------------------------------------------------------------------------#


def calculate_traceless_tidal_shear(
    tidal_tensor: np.ndarray, grid_size: int
) -> np.ndarray:
    """
    Calculate the traceless tidal shear from the given tidal tensor.

    Parameters:
    - tidal_tensor: numpy.ndarray
      The tidal tensor.
    - grid_size: int
      The size of the 3D grid.

    Returns:
    - tidal_shear: numpy.ndarray
      The traceless tidal shear.
    """

    # Reshape to (grid_size*grid_size*grid_size, 3, 3) to make calculations faster
    tid = tidal_tensor.reshape(grid_size * grid_size * grid_size, 3, 3)
    traceless = make_traceless(tid)
    return traceless.reshape(grid_size, grid_size, grid_size, 3, 3)


# -------------------------------------------------------------------------------------------------------------#


def load_all_npy_files(
    folder_path: str,
    filename_prefix: str,
    str_smoothing_scales: list[str],
    lambda_th: float = None,
) -> list:
    """
    Loads all files with the .npy extension from a folder using concurrent processing.

    Args:
        folder_path: str
            Path to the folder containing the NumPy files.

    Returns:
        list:
            A list of loaded NumPy arrays from the folder.
    """

    # Create a list of all the tidal tensor files (tidal_tensor_*.npy)
    if lambda_th is not None:
        lambda_th_str = str(lambda_th).replace(".", "p")
        tidal_files = [
            os.path.join(folder_path, f"{filename_prefix}{name}_{lambda_th_str}.npy")
            for name in str_smoothing_scales
        ]
    else:
        tidal_files = [
            os.path.join(folder_path, f"{filename_prefix}{name}.npy")
            for name in str_smoothing_scales
        ]

    if not tidal_files:
        raise logging.error("No .npy files found in the folder.")
        return []

    data_list = []

    # Use ThreadPoolExecutor to load files concurrently
    with ThreadPoolExecutor() as executor:
        future_to_filepath = {
            executor.submit(load_data, filepath): filepath for filepath in tidal_files
        }
        for future in as_completed(future_to_filepath):
            try:
                data = future.result()
                data_list.append(data)
            except Exception as e:
                logging.error(f"Error loading {future_to_filepath[future]}: {e}")
                exit()

    return data_list


# -------------------------------------------------------------------------------------------------------------#


def calculate_eigenvalues_and_vectors(
    tidal_shear_tensor: np.ndarray, grid_size: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate eigenvalues and eigenvectors of the tidal shear tensor for each grid point or voxel.

    Parameters:
    - tidal_shear_tensor (ndarray): 3D array representing the tidal shear tensor with shape (grid_size, grid_size, grid_size, 3, 3).
    - grid_size (int): Size of the grid in each dimension.

    Returns:
    - eigenvalues (ndarray): Array containing the eigenvalues for each grid point.
      Shape (grid_size, grid_size, grid_size, 3).
    - eigenvectors (ndarray): Array containing the corresponding eigenvectors for each grid point.
      Shape (grid_size, grid_size, grid_size, 3, 3).

    Note:
    - The eigenvalues, and therefore the eigenvectors, are sorted in descending order.
      That is, the first eigenvalue corresponds to the largest eigenvalue.
      And the first eigenvector corresponds to the eigenvector associated with the largest eigenvalue.

    """

    # Reshape the tidal shear tensor to have a shape of (grid_size*grid_size*grid_size, 3, 3)
    reshaped_tensor = tidal_shear_tensor.reshape((-1, 3, 3))

    # Calculate eigenvalues and eigenvectors for all tensors simultaneously
    eigenvalues, eigenvectors = np.linalg.eig(reshaped_tensor)

    # Sort eigenvalues and corresponding eigenvectors in descending order
    sorted_indices = np.argsort(eigenvalues, axis=-1)[:, ::-1]

    # Apply sorting to eigenvalues and eigenvectors arrays
    eigenvalues = np.take_along_axis(eigenvalues, sorted_indices, axis=-1)
    eigenvectors = np.take_along_axis(
        eigenvectors, sorted_indices[..., np.newaxis], axis=1
    )

    # Reshape eigenvalues and eigenvectors arrays back to the original shape
    eigenvalues = eigenvalues.reshape((grid_size, grid_size, grid_size, 3))
    eigenvectors = eigenvectors.reshape((grid_size, grid_size, grid_size, 3, 3))

    return eigenvalues, eigenvectors


# -------------------------------------------------------------------------------------------------------------#


def classify_structure(eigenvalues: np.ndarray, lambda_th: float) -> np.ndarray:
    """
    Uses T-web classification scheme.

    Classifies each voxel as a part of Void, Sheet, Filament, or Node
    based on the eigenvalues of the tidal tensor.

    This counts the number of positive eigenvalues for each voxel.

    - If all three eigenvalues are positive, the voxel is classified as a Node.
    - If two eigenvalues are positive, the voxel is classified as a Filament.
    - If one eigenvalue is positive, the voxel is classified as a Sheet.
    - If no eigenvalues are positive, the voxel is classified as a Void.

    Args:
    - eigenvalues (np.ndarray): Array containing the eigenvalues for each grid point.
      Shape: (grid_size, grid_size, grid_size, 3).

    Returns:
    - classified_array (np.ndarray): Array containing the classification for each grid point.
      Shape: (grid_size, grid_size, grid_size).

    """

    # Count the number of positive eigenvalues
    classified_array = np.sum(
        eigenvalues > lambda_th, axis=-1
    )  # Shape of eigenvalues is (grid_size, grid_size, grid_size, 3)

    return classified_array  # Shape of classified_array is (grid_size, grid_size, grid_size)


# -------------------------------------------------------------------------------------------------------------#


def calculate_volume_fraction(classification_matrix: np.ndarray, label: int) -> float:
    """
    Calculate the volume fraction of a particular environment in a 3D classification matrix.

    Parameters:
    - classification_matrix (numpy.ndarray): 3D array containing voxel labels.
    - label (int): The label for which the volume fraction is calculated.

    Returns:
    float: The volume fraction of the specified label in the matrix.
    """
    return np.sum(classification_matrix.flatten() == label) / np.prod(
        classification_matrix.shape
    )


# -------------------------------------------------------------------------------------------------------------#


def calculate_volume_fractions(classification_matrix: np.ndarray) -> np.ndarray:
    """
    Calculate the volume fractions for all the environments in a 3D classification matrix.

    Parameters:
    - classification_matrix (numpy.ndarray): 3D array containing voxel labels.
    - num_labels (numpy.ndarray or list): Array or list of labels for which volume fractions are calculated.

    Returns:
    numpy.ndarray: Array of volume fractions corresponding to each label.
    """

    num_labels = np.arange(4)  # Labels: 0 = Void, 1 = Sheet, 2 = Filament, 3 = Cluster

    volume_fractions = np.zeros_like(num_labels, dtype=float)

    for i, label in enumerate(num_labels):
        volume_fractions[i] = calculate_volume_fraction(classification_matrix, label)

    return volume_fractions


# -------------------------------------------------------------------------------------------------------------#


def plot_volfrac_rs(
    classifications: np.ndarray, smooth_scales: list, save_path: str
) -> None:

    labels = ["Void", "Sheets", "Filaments", "Clusters"]

    vol_fracs = []

    for class_i in classifications:
        vol_frac = calculate_volume_fractions(class_i)
        vol_fracs.append(vol_frac)

    vol_fracs = np.array(vol_fracs)

    # Plot the volume fractions
    plt.figure(figsize=(5, 5), dpi=300)

    for i in range(len(vol_fracs.T)):
        plt.semilogx(smooth_scales, vol_fracs[:, i], label=labels[i])

    plt.axhline(
        0.43, ls="--", lw=1, color="black", alpha=0.4
    )  # Attains gaussian random field
    plt.axhline(
        0.072, ls="--", lw=1, color="black", alpha=0.4
    )  # Attains gaussian random field
    # plt.axvline(0.54, ls = '--', lw = 1, color = 'black', alpha = 0.8) # Choose a value to draw a vertical line at a particlar scale

    plt.xlabel("$R_s~[h^{-1}~ Mpc] $")
    plt.ylabel("Volume fraction")
    plt.tick_params(
        axis="both",
        which="both",
        left=True,
        right=True,
        top=True,
        bottom=True,
        direction="in",
        labelsize=7,
    )
    plt.legend(bbox_to_anchor=(1, 0.6), fontsize=9, fancybox=True, handlelength=1)
    plt.savefig(save_path)


# -------------------------------------------------------------------------------------------------------------#


def slice_density_field(
    density_field: np.ndarray, slice_thickness: list[int, int], projection: str
) -> np.ndarray:
    """
    Extracts a 2D slice from a 3D density field and applies logarithmic transformation.

    Parameters:
    - density_field (numpy.ndarray): 3D array representing the density field.
    - projection (str): The dimension of the slice ('xy', 'yz', 'zx').
    - slice_thickness (list): The thickness of the slice. [start_index, end_index]

    Returns:
    - numpy.ndarray: 2D array representing the logarithmic transformation of the specified slice.

    Notes:
    - The logarithmic transformation is applied using the formula np.log10(density_field + 1 + eps),
      where eps is a small constant (1e-15) to prevent taking the logarithm of zero.

    Example:
    >>> density_field = np.random.rand(100, 100, 100)
    >>> slice_index = [0, 256] # (for a grid size of 512)
    >>> projection = 'xy'
    >>> result = slice_density_field(density_field, slice_index, projection)
    """

    eps = 1e-15  # so that log doesn't get a value 0
    N_start = slice_thickness[0]
    N_end = slice_thickness[1]

    # Plot for zero smoothening density field
    plt.figure(figsize=(2, 2), dpi=200)
    delplot1 = np.log10(density_field + 1 + eps)

    if projection == ("yz" or "zy"):
        d_field = np.mean(delplot1[N_start:N_end, :, :], axis=0)

    if projection == ("xz" or "zx"):
        d_field = np.mean(delplot1[:, N_start:N_end, :], axis=1)

    if projection == ("xy" or "yx"):
        d_field = np.mean(delplot1[:, :, N_start:N_end], axis=2)

    return d_field


# -------------------------------------------------------------------------------------------------------------#


def get_structure_positions(
    structure: str,
    projection: str,
    slice_index: int,
    classification_matrix: np.ndarray,
    grid_size: int,
    box_size: int,
) -> np.ndarray:
    """
    Extracts positions of points in a 2D slice based on the specified structure, projection, and classification.

    Parameters:
    - structure (str): A string representing the structure type ('v', 's', 'f', 'n').
      'v' = Voids, 's' = Sheets, 'f' = Filaments, 'n' = Nodes.
    - projection (str): A string representing the projection type ('xy', 'yx', 'yz', 'zy', 'zx', 'xz').
    - slice_index (int): Index of the slice to be extracted.
    - classification (numpy.ndarray): 3D array representing the classification of points.
    - grid_size (int): Size of the grid in the 3D array.
    - box_size (int): Size of the box in physical units.

    Returns:
    - numpy.ndarray: 2D array representing the positions of points in the specified slice.

    Notes:
    - Points are classified into different structures based on the structure parameter.
    - The projection parameter determines the orientation of the slice (e.g., 'xy', 'yz', 'zx').
    - The classification parameter is a 3D array where points are classified into different structures.
    - The positions are returned in physical units, scaled based on the grid size and box size.
    - The function prints 'ValueError' if invalid structure or projection is provided.

    Example:
    >>> structure = 'v' # for Voids
    >>> projection = 'xy'
    >>> slice_index = 2
    >>> grid_size = 512
    >>> box_size = 100
    >>> classification = np.random.randint(0, 4, size=(512, 512, 512))
    >>> result = get_structure_positions(structure, projection, slice_index, classification, 512, 100)
    """

    if str(structure)[0].lower() == "v":

        if str(projection) == "xy" or "yx":
            mask_r, mask_c = np.where(classification_matrix[:, :, slice_index] == 0)
            mask_r = mask_r * box_size / grid_size
            mask_c = mask_c * box_size / grid_size
            position = np.column_stack((mask_r, mask_c))
            return position

        elif str(projection) == "yz" or "zy":
            mask_r, mask_c = np.where(classification_matrix[slice_index, :, :] == 0)
            mask_r = mask_r * box_size / grid_size
            mask_c = mask_c * box_size / grid_size
            position = np.column_stack((mask_r, mask_c))
            return position

        elif str(projection) == "zx" or "xz":
            mask_r, mask_c = np.where(classification_matrix[:, slice_index, :] == 0)
            mask_r = mask_r * box_size / grid_size
            mask_c = mask_c * box_size / grid_size
            position = np.column_stack((mask_r, mask_c))
            return position

        else:
            raise ValueError("Environment not found!")

    elif str(structure)[0].lower() == "s":

        if str(projection) == "xy" or "yx":
            mask_r, mask_c = np.where(classification_matrix[:, :, slice_index] == 1)
            mask_r = mask_r * box_size / grid_size
            mask_c = mask_c * box_size / grid_size
            position = np.column_stack((mask_r, mask_c))
            return position

        elif str(projection) == "yz" or "zy":
            mask_r, mask_c = np.where(classification_matrix[slice_index, :, :] == 1)
            mask_r = mask_r * box_size / grid_size
            mask_c = mask_c * box_size / grid_size
            position = np.column_stack((mask_r, mask_c))
            return position

        elif str(projection) == "zx" or "xz":
            mask_r, mask_c = np.where(classification_matrix[:, slice_index, :] == 1)
            mask_r = mask_r * box_size / grid_size
            mask_c = mask_c * box_size / grid_size
            position = np.column_stack((mask_r, mask_c))
            return position

        else:
            raise ValueError("Environment not found!")

    elif str(structure)[0].lower() == "f":

        if str(projection) == "xy" or "yx":
            mask_r, mask_c = np.where(classification_matrix[:, :, slice_index] == 2)
            mask_r = mask_r * box_size / grid_size
            mask_c = mask_c * box_size / grid_size
            position = np.column_stack((mask_r, mask_c))
            return position

        elif str(projection) == "yz" or "zy":
            mask_r, mask_c = np.where(classification_matrix[slice_index, :, :] == 2)
            mask_r = mask_r * box_size / grid_size
            mask_c = mask_c * box_size / grid_size
            position = np.column_stack((mask_r, mask_c))
            return position

        elif str(projection) == "zx" or "xz":
            mask_r, mask_c = np.where(classification_matrix[:, slice_index, :] == 2)
            mask_r = mask_r * box_size / grid_size
            mask_c = mask_c * box_size / grid_size
            position = np.column_stack((mask_r, mask_c))
            return position

        else:
            raise ValueError("Environment not found!")

    elif str(structure)[0].lower() == "n":

        if str(projection) == "xy" or "yx":
            mask_r, mask_c = np.where(classification_matrix[:, :, slice_index] == 3)
            mask_r = mask_r * box_size / grid_size
            mask_c = mask_c * box_size / grid_size
            position = np.column_stack((mask_r, mask_c))
            return position

        elif str(projection) == "yz" or "zy":
            mask_r, mask_c = np.where(classification_matrix[slice_index, :, :] == 3)
            mask_r = mask_r * box_size / grid_size
            mask_c = mask_c * box_size / grid_size
            position = np.column_stack((mask_r, mask_c))
            return position

        elif str(projection) == "zx" or "xz":
            mask_r, mask_c = np.where(classification_matrix[:, slice_index, :] == 3)
            mask_r = mask_r * box_size / grid_size
            mask_c = mask_c * box_size / grid_size
            position = np.column_stack((mask_r, mask_c))
            return position

        else:
            raise ValueError("Environment not found!")

    else:
        raise ValueError("Environment not found!")


# -------------------------------------------------------------------------------------------------------------#


def get_all_env_pos(
    classification_matrix: np.ndarray,
    slice_index: int,
    projection: str,
    grid_size: int,
    box_size: int,
) -> list:
    """
    Extracts positions of points for all environment types ('v', 's', 'f', 'n') in a 2D slice.

    Parameters:
    - classification (numpy.ndarray): 3D array representing the classification of points.
    - slice_index (int): Index of the slice to be extracted.
    - projection (str): A string representing the projection type ('xy', 'yx', 'yz', 'zy', 'zx', 'xz').
    - grid_size (int): Size of the grid in the 3D array.
    - box_size (int): Size of the box in physical units.

    Returns:
    - list: List containing 2D arrays representing positions of points for each environment type.

    Notes:
    - Uses the get_structure_positions function to extract positions for each environment type.
    - The classification parameter is a 3D array where points are classified into different structures.
    - The slice_index parameter determines the index of the slice to be extracted.
    - Positions are returned in physical units, scaled based on the default grid size and box size.

    Example:
    >>> classification = np.random.randint(0, 4, size=(512, 512, 512))
    >>> slice_index = 256
    >>> projection = 'xy'
    >>> grid_size = 512
    >>> box_size = 100
    >>> result = get_class_env_pos(classification, slice_index, projection, grid_size, box_size)
    """

    envs = ["v", "s", "f", "n"]  # Voids, Sheets, Filaments, Nodes

    all_env_pos = [
        get_structure_positions(
            env, projection, slice_index, classification_matrix, grid_size, box_size
        )
        for env in envs
    ]

    return all_env_pos


# -------------------------------------------------------------------------------------------------------------#


def plot_classification_overlay(
    smth_scale: float,
    lambda_th: float,
    classification_matrix: np.ndarray,
    smoothed_rho: np.ndarray,
    slice_thickness: list[int, int],
    slice_index: int,
    projection: str,
    grid_size: int,
    box_size: int,
    save_path: str,
):
    """
    Plot the overlay of structure classifications (void, sheet, filament, cluster) on a density field slice.

    Parameters:
    - smth_scale (float): The smoothing scale used in the classification.
    - classification (numpy.ndarray): 3D array representing the classification of points.
    - rho (numpy.ndarray): 3D array representing the density field.
    - slice_thickness (list): Thickness of the slice. [start_index, end_index]
    - slice_index (int): Index of the slice to be plotted.
    - projection (str): Projection type ('xy', 'yz', 'zx').
    - grid_size (int): Size of the grid in the 3D array.
    - box_size (int): Size of the simulation box in physical units [Mpc a/h].
    - save_path (str): Path to save the plot.

    Returns:
    - None

    Notes:
    - The function plots the overlay of structure classifications on a density field slice.
    - Four different structures are represented: Void, Sheet, Filament, Cluster.
    - The classification is based on the structure type assigned to each voxel.
    """

    labels = ["Void", "Sheet", "Filament", "Cluster"]
    alphas = [0.1, 0.1, 0.1, 0.7]  # Transparency of points for each classification
    sizes = [0.3, 0.3, 0.3, 0.8]  # Size of points for each classification

    # Get positions (in physical units) in a 2D slice for each environment type
    env_pos = get_all_env_pos(
        classification_matrix, slice_index, projection, grid_size, box_size
    )

    # Get the density field slice
    slice_ = slice_density_field(smoothed_rho, slice_thickness, projection)

    # Plot the overlay of structure classifications on the density field slice
    fig, ax = plt.subplots(2, 2, figsize=(10, 8), dpi=300)

    ax = ax.flatten()

    for i in range(len(labels)):

        ax[i].imshow(
            slice_, cmap="Greys", extent=[0, box_size, 0, box_size], origin="lower"
        )
        ax[i].scatter(
            env_pos[i][:, 0],
            env_pos[i][:, 1],
            s=sizes[i],
            alpha=alphas[i],
            label=labels[i],
        )
        ax[i].set_title(f"{labels[i]} classification (λ={lambda_th})", fontsize=13)
        ax[i].set_xlabel("$h^{-1}~Mpc$")
        ax[i].set_ylabel("$h^{-1}~Mpc$")
        ax[i].tick_params(
            axis="both",
            which="both",
            left=True,
            right=True,
            top=True,
            bottom=True,
            direction="in",
            labelsize=7,
        )

    plt.suptitle(f"$R_s = {smth_scale}~h^{-1}~Mpc$")
    fig.tight_layout()
    fig.savefig(save_path)


# -------------------------------------------------------------------------------------------------------------#


def overlay_all_envs(
    classification_matrix: np.ndarray,
    smoothed_rho: np.ndarray,
    slice_thickness: list[int, int],
    smoothing_scale: list[float],
    slice_index: int,
    projection: str,
    grid_size: int,
    box_size: int,
    save_path: str,
    lambda_th: float = None,
) -> None:
    """
    Overlay all environmental classifications on a density slice and save the resulting image.

    Parameters:
    - classification_matrix (np.ndarray): 3D array of environmental classifications.
    - smoothed_rho (np.ndarray): 3D array of smoothed density values.
    - slice_thickness (list[int, int]): List containing the start and end indices defining the slice thickness.
    - smoothing_scale (list[float]): List containing the smoothing scale values.
    - slice_index (int): Index of the slice to be taken along the specified projection axis.
    - projection (str): Axis along which the slice is taken ('x', 'y', or 'z').
    - grid_size (int): Size of the grid.
    - box_size (int): Size of the box in Mpc/h.
    - save_path (str): Path to save the resulting image.

    Returns:
    - None: This function saves the resulting image to the specified path.
    """

    labels = ["Void", "Sheet", "Filament", "Cluster"]

    env_pos = get_all_env_pos(
        classification_matrix, slice_index, projection, grid_size, box_size
    )

    density_slice = slice_density_field(smoothed_rho, slice_thickness, projection)

    alphas = [0.5, 0.5, 0.5, 0.5]
    sizes = [0.015, 0.015, 0.015, 0.015]

    plt.figure(figsize=(5, 7), dpi=300)

    for i in range(len(labels)):
        plt.imshow(
            density_slice,
            cmap="Greys",
            extent=[0, box_size, 0, box_size],
            origin="lower",
        )
        plt.scatter(
            env_pos[i][:, 0],
            env_pos[i][:, 1],
            s=sizes[i],
            alpha=alphas[i],
            label=labels[i],
        )
        plt.tick_params(
            axis="both",
            which="both",
            left=True,
            right=True,
            top=True,
            bottom=True,
            direction="in",
            labelsize=15,
        )
        plt.xlabel(r"$h^{-1}~Mpc$", fontsize=12)
        plt.ylabel(r"$h^{-1}~Mpc$", fontsize=12)

    plt.title(
        r"$R_s = {:.2f}~h^{{-1}}~Mpc$ (λ={})".format(smoothing_scale, lambda_th),
        fontsize=18,
    )

    plt.legend(bbox_to_anchor=(1.05, 0.5), fontsize=15, markerscale=70.0, ncol=1)

    plt.savefig(save_path)


# -------------------------------------------------------------------------------------------------------------#


def get_environment_change(
    clf1: np.ndarray,
    clf2: np.ndarray,
    init_env: int,
    final_env: int,
    slice_index: int,
    projection: str,
    grid_size: int,
    box_size: int,
) -> np.ndarray:
    """
    Extracts positions where the environment type changes between two classification arrays.

    Parameters:
    - clf1 (numpy.ndarray): First 3D classification array representing the initial structure classification.
    - clf2 (numpy.ndarray): Second 3D classificaiton array representing the final structure classification.
    - init_env (int): The initial environment type.
    - final_env (int): The final environment type.
    - slice_index (int): Index of the slice to be extracted.
    - projection (str): The projection type ('xy', 'yz', 'zx').
    - grid_size (int): Size of the grid in the 3D array.
    - box_size (int): Size of the simulation box in physical units [Mpc a/h].

    Returns:
    - numpy.ndarray: 2D array representing positions where the environment type changes.

    Notes:
    - Compares the classification arrays clf1 and clf2 to identify changes in environment types.
      Point to note:
        - Voids are represented by 0, Sheets by 1, Filaments by 2, and Nodes by 3.
        - Therefore, the difference (diff) indicates a corresponding change between the environment types.
          For example, diff = 1 indicates a V2S, S2F, F2N change.
    - The init_env parameter specifies the initial environment type.
    - Positions are returned in physical units, scaled based on the default grid size and box size.

    Example:
    >>> clf1 = np.random.randint(0, 4, size=(512, 512, 512))
    >>> clf2 = np.random.randint(0, 4, size=(512, 512, 512))
    >>> diff = 1
    >>> init_env = 2
    >>> slice_index = 256
    >>> projection = 'xy'
    >>> grid_size = 512
    >>> box_size = 100
    >>> result = get_env_change(clf1, clf2, diff, init_env)
    """

    diff = final_env - init_env

    if str(projection) in ["xy", "yx"]:

        # Calculate the difference between clf2 and clf1 for the specified slice
        dif_clf = clf2[:, :, slice_index] - clf1[:, :, slice_index]

        # Check where the difference equals the specified difference and the initial environment matches
        bool_clf = np.isclose(dif_clf, diff) & (clf1[:, :, slice_index] == init_env)

        # Get the row and column indices where the condition is met
        col, row = np.where(bool_clf)

        # Convert the indices to physical positions
        col = col * box_size / grid_size
        row = row * box_size / grid_size

        # Combine row and column indices into a 2D array
        result = np.column_stack((row, col))

        return result

    elif str(projection) in ["yz", "zy"]:
        dif_clf = clf2[slice_index, :, :] - clf1[slice_index, :, :]
        bool_clf = np.isclose(dif_clf, diff) & (clf1[slice_index, :, :] == init_env)
        col, row = np.where(bool_clf)
        col = col * box_size / grid_size
        row = row * box_size / grid_size
        result = np.column_stack((row, col))
        return result

    elif str(projection) in ["zx", "xz"]:
        dif_clf = clf2[:, slice_index, :] - clf1[:, slice_index, :]
        bool_clf = np.isclose(dif_clf, diff) & (clf1[:, slice_index, :] == init_env)
        col, row = np.where(bool_clf)
        col = col * box_size / grid_size
        row = row * box_size / grid_size
        result = np.column_stack((row, col))
        return result

    else:
        raise ValueError("Invalid projection type! Use one of: xy, yx, yz, zy, zx, xz")


# -------------------------------------------------------------------------------------------------------------#


def get_env_changes(
    clf1: np.ndarray,
    clf2: np.ndarray,
    slice_index: int,
    projection: str,
    grid_size: int,
    box_size: int,
) -> dict[str, np.ndarray]:
    """
    Extracts positions where different types of environment changes occur between two classification arrays.

    Parameters:
    - clf1 (numpy.ndarray): First 3D classification array representing the initial classification of points.
    - clf2 (numpy.ndarray): Second 3D classification array representing the final classification of points.
    - slice_index (int): Index of the slice to be extracted.
    - projection (str): The projection type ('xy', 'yz', 'zx').
    - grid_size (int): Size of the grid in the 3D array.
    - box_size (int): Size of the simulation box in physical units [Mpc a/h].

    Returns:
    - dict[str, numpy.ndarray]: A dictionary containing positions where different types of environment changes occur.

    Notes:
    - Compares the classification arrays clf1 and clf2 to identify changes in different types of environments.
    - Different types of environment changes are categorized based on the change in environment type (e.g., Void to Sheet, Sheet to Filament, etc.).
    - The slice_index parameter specifies the index of the slice to be extracted.
    - Positions are returned in physical units, scaled based on the default grid size (512) and box size (100).

    Example: (For change in environment from Void to Sheet)
    >>> clf1 = np.random.randint(0, 4, size=(512, 512, 512))
    >>> clf2 = np.random.randint(0, 4, size=(512, 512, 512))
    >>> slice_index = 256
    >>> projection = 'xy'
    >>> grid_size = 512
    >>> box_size = 100
    >>> result = get_env_changes(clf1, clf2, slice_index, projection, grid_size, box_size)
    """

    return {
        "Void to Sheet": get_environment_change(
            clf1, clf2, 0, 1, slice_index, projection, grid_size, box_size
        ),
        "Sheet to Filament": get_environment_change(
            clf1, clf2, 1, 2, slice_index, projection, grid_size, box_size
        ),
        "Filament to Node": get_environment_change(
            clf1, clf2, 2, 3, slice_index, projection, grid_size, box_size
        ),
        "Sheet to Void": get_environment_change(
            clf1, clf2, 1, 0, slice_index, projection, grid_size, box_size
        ),
        "Filament to Sheet": get_environment_change(
            clf1, clf2, 2, 1, slice_index, projection, grid_size, box_size
        ),
        "Node to Filament": get_environment_change(
            clf1, clf2, 3, 2, slice_index, projection, grid_size, box_size
        ),
        "Void to Filament": get_environment_change(
            clf1, clf2, 0, 2, slice_index, projection, grid_size, box_size
        ),
        "Void to Node": get_environment_change(
            clf1, clf2, 0, 3, slice_index, projection, grid_size, box_size
        ),
        "Sheet to Node": get_environment_change(
            clf1, clf2, 1, 3, slice_index, projection, grid_size, box_size
        ),
        "Filament to Void": get_environment_change(
            clf1, clf2, 2, 0, slice_index, projection, grid_size, box_size
        ),
        "Node to Void": get_environment_change(
            clf1, clf2, 3, 0, slice_index, projection, grid_size, box_size
        ),
        "Node to Sheet": get_environment_change(
            clf1, clf2, 3, 1, slice_index, projection, grid_size, box_size
        ),
    }


# -------------------------------------------------------------------------------------------------------------#


def plot_env_changes(
    transformations: dict[str, np.ndarray],
    density_slice: np.ndarray,
    box_size: int,
    title: str,
    save_path: str,
) -> None:
    """
    Plot changes in classification of halo environments.

    Parameters:
    - transformations (dict[str, numpy.ndarray]): A dictionary containing transformations of halo environments.
    - density_slice (numpy.ndarray): 2D array representing the density slice.
    - box_size (int): Size of the simulation box in physical units [Mpc a/h].
    - title (str): Title of the plot.
    - save_path (str): File path to save the plot.

    Returns:
    - None

    Notes:
    - The transformations parameter is a dictionary where keys represent the type of transformation and values represent the positions of transformations.
    - The density_slice parameter is a 2D array representing the density slice.
    - Positions of transformations are plotted on top of the density slice.
    - The box_size parameter specifies the size of the simulation box in physical units.
    - The title parameter is used as the title of the plot.
    - The save_path parameter specifies the file path to save the plot.

    Example:
    >>> transformations = {'Transformation 1': np.array([[x1, y1], [x2, y2], ...]), 'Transformation 2': np.array([[x1, y1], [x2, y2], ...]), ...}
    >>> density_slice = np.random.rand(512, 512)
    >>> box_size = 100
    >>> title = 'Changes in classification of Halo environments'
    >>> save_path = 'plot.png'
    >>> plot_env_changes(transformations, density_slice, box_size, title, save_path)
    """

    fig, ax = plt.subplots(2, 3, figsize=(12, 10), dpi=400, sharey=True)
    fig.subplots_adjust(wspace=0.1, hspace=0)
    ax = ax.flatten()

    for idx, (title, transformation) in enumerate(transformations.items()):
        ax[idx].imshow(
            density_slice,
            cmap="Greys",
            extent=[0, box_size, 0, box_size],
            origin="lower",
        )
        ax[idx].scatter(
            transformation[:, 0], transformation[:, 1], c="r", s=0.08, alpha=0.2
        )
        ax[idx].tick_params(
            axis="both",
            which="both",
            left=True,
            right=True,
            top=True,
            bottom=True,
            direction="in",
            labelsize=12,
        )
        ax[idx].set_title(title, fontsize=15)

    ax[1].set_xlabel(r"$h^{-1}~Mpc$")
    ax[0].set_ylabel(r"$h^{-1}~Mpc$")
    ax[3].set_ylabel(r"$h^{-1}~Mpc$")
    ax[4].set_xlabel(r"$h^{-1}~Mpc$")

    plt.suptitle("Changes in classification of Halo environments", fontsize=18)
    plt.savefig(save_path)


# ----------------------------------------------END-OF-THE-SCRIPT------------------------------------------------------------#
