"""
T-web Classification Interface

This module provides a clean interface to compute T-web classification
from external density fields using Dave's T-web implementation.

Author: Adapted from Asit Dave's T-web implementation
Date: September 2025
"""

import numpy as np
import logging
import os
from pathlib import Path
from .LSS_TWeb_BlackBox import (
    smooth_field,
    calculate_tidal_tensor,
    calculate_traceless_tidal_shear,
    calculate_eigenvalues_and_vectors,
    classify_structure,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def get_tweb_file_path(kernel: dict, variant: int, box: int, lambda_th: float, smoothing_scale: float) -> Path:
    """
    Generate the file path for saving/loading T-web classification results.
    Following the same pattern as COLAVERSEReader.get_path().

    Parameters:
    -----------
    kernel : dict
        Kernel configuration dictionary
    variant : int
        Variant/realization number
    box : int
        Box number
    lambda_th : float
        Lambda threshold value for classification
    smoothing_scale : float
        Gaussian smoothing scale in Mpc/h

    Returns:
    --------
    Path
        Path to the T-web classification file
    """
    # Import here to avoid circular imports
    import sys
    from pathlib import Path as PathLib

    # Add the parent directory to sys.path to find custom module
    src_dir = PathLib(__file__).parent.parent
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    from src.reader import COLAVERSEReader as cr

    reader = cr(kernel, variant, box)

    # Create lambda threshold and smoothing scale strings for filename compatibility
    lambda_th_str = str(lambda_th).replace(".", "p")
    smoothing_str = str(smoothing_scale).replace(".", "p")

    # Get output directory path
    output_directory = reader.output_directory
    if not output_directory.exists():
        output_directory.mkdir(parents=True, exist_ok=True)

    # Generate filename following COLAVERSE naming convention
    redshift_str = f"z{reader.redshift:.3f}".replace(".", "p")
    filename = f"{reader.simulation_name}_{box}_{redshift_str}_{reader.scheme}{reader.grid}_tweb_s{smoothing_str}_l{lambda_th_str}.npy"

    return output_directory / filename


def save_tweb_classification(classification: np.ndarray, file_path: Path) -> None:
    """
    Save T-web classification results to file.

    Parameters:
    -----------
    classification : np.ndarray
        T-web classification array
    file_path : Path
        Path where to save the classification
    """
    logging.info(f"Saving T-web classification to: {file_path}")
    np.save(str(file_path), classification)
    logging.info("T-web classification saved successfully.")


def load_tweb_classification(file_path: Path) -> np.ndarray:
    """
    Load T-web classification results from file.

    Parameters:
    -----------
    file_path : Path
        Path to the classification file

    Returns:
    --------
    np.ndarray
        Loaded T-web classification array
    """
    if not file_path.exists():
        raise FileNotFoundError(f"T-web classification file not found: {file_path}")

    logging.info(f"Loading T-web classification from: {file_path}")
    classification = np.load(str(file_path))
    logging.info("T-web classification loaded successfully.")
    return classification


def check_tweb_file_exists(kernel: dict, variant: int, box: int, lambda_th: float, smoothing_scale: float) -> bool:
    """
    Check if T-web classification file already exists.

    Parameters:
    -----------
    kernel : dict
        Kernel configuration dictionary
    variant : int
        Variant/realization number
    box : int
        Box number
    lambda_th : float
        Lambda threshold value for classification
    smoothing_scale : float
        Gaussian smoothing scale in Mpc/h

    Returns:
    --------
    bool
        True if file exists, False otherwise
    """
    file_path = get_tweb_file_path(kernel, variant, box, lambda_th, smoothing_scale)
    return file_path.exists()


def compute(field, kernel, variant, box, lambda_th=0.0, smoothing_scale=3.0, verbose=True, force_recalc=False):
    """
    Compute T-web classification from a density field using Dave's implementation.

    This function provides a clean interface to Dave's T-web classification code,
    taking a density field and returning the cosmic web classification.

    Parameters:
    -----------
    field : np.ndarray
        3D density contrast field (delta = rho/rho_mean - 1)
        Shape should be (N, N, N) for cubic grid
    kernel : dict
        Kernel configuration dictionary containing simulation parameters
    variant : int
        Variant/realization number
    box : int
        Box number
    lambda_th : float, optional
        Threshold for eigenvalue classification (default: 0.0)
    smoothing_scale : float, optional
        Gaussian smoothing scale in Mpc/h (default: 3.0)
    verbose : bool, optional
        Print progress information (default: True)
    force_recalc : bool, optional
        Force recalculation even if cached result exists (default: False)

    Returns:
    --------
    classification : np.ndarray
        T-web classification with same shape as input density field
        Values: 0=Void, 1=Sheet, 2=Filament, 3=Knot

    Examples:
    ---------
    >>> import numpy as np
    >>> from src.tweb.Tweb_Classification_Interface import compute
    >>>
    >>> # Example with COLAVERSE density field
    >>> from src.custom.reader import COLAVERSEReader as read
    >>> data = read(kernel_config)
    >>> density_field = data.load_density_field(1, 0.0, 128, "TSC")
    >>> classification = compute(density_field, kernel_config, variant=0)
    """
    params = kernel["realizations"][variant]

    # Check if cached result exists (unless forcing recalculation)
    if not force_recalc and check_tweb_file_exists(kernel, variant, box, lambda_th, smoothing_scale):
        if verbose:
            logging.info("T-web classification already computed and cached. Loading from file...")
        file_path = get_tweb_file_path(kernel, variant, box, lambda_th, smoothing_scale)
        try:
            return load_tweb_classification(file_path)
        except Exception as e:
            if verbose:
                logging.warning(f"Failed to load cached T-web classification: {e}")
                logging.info("Proceeding with fresh calculation...")

    if verbose:
        logging.info("Starting T-web classification using Dave's implementation...")
        logging.info(f"Density field shape: {field.shape}")
        logging.info(f"Box size: {params['physical_size']} Mpc/h")
        logging.info(f"Smoothing scale: {smoothing_scale} Mpc/h")
        logging.info(f"Lambda threshold: {lambda_th}")

    # Validate input
    if field.ndim != 3:
        raise ValueError(f"Density field must be 3D, got {field.ndim}D")

    if not (field.shape[0] == field.shape[1] == field.shape[2]):
        raise ValueError(f"Density field must be cubic, got shape {field.shape}")

    grid_size = field.shape[0]

    # Step 1: Apply smoothing if requested
    if smoothing_scale > 0:
        if verbose:
            logging.info("Smoothing density field...")
        smoothed_density = smooth_field(field, smoothing_scale, params["physical_size"], grid_size)
    else:
        smoothed_density = field.copy()

    # Step 2: Compute tidal tensor using Dave's function
    if verbose:
        logging.info("Computing tidal tensor...")

    tidal_tensor = calculate_tidal_tensor(smoothed_density, calculate_potential=False)

    # Step 3: Make tidal tensor traceless (convert to shear tensor)
    if verbose:
        logging.info("Converting to traceless tidal shear tensor...")

    tidal_shear_tensor = calculate_traceless_tidal_shear(tidal_tensor, grid_size)

    # Step 4: Compute eigenvalues and eigenvectors
    if verbose:
        logging.info("Computing eigenvalues and eigenvectors...")

    eigenvalues, eigenvectors = calculate_eigenvalues_and_vectors(tidal_shear_tensor, grid_size)

    # Step 5: Classify structures
    if verbose:
        logging.info("Classifying cosmic web structures...")

    classification = classify_structure(eigenvalues, lambda_th)

    if verbose:
        # Print classification statistics
        unique, counts = np.unique(classification, return_counts=True)
        total = classification.size
        structure_names = ["Void", "Sheet", "Filament", "Knot"]

        logging.info("Classification completed successfully!")
        logging.info("Structure statistics:")
        for structure_id, count in zip(unique, counts):
            if structure_id < len(structure_names):
                name = structure_names[structure_id]
                percentage = count / total * 100
                logging.info(f"  {name}: {count:8d} cells ({percentage:5.1f}%)")

    # Save the result to cache
    try:
        file_path = get_tweb_file_path(kernel, variant, box, lambda_th, smoothing_scale)
        save_tweb_classification(classification, file_path)
    except Exception as e:
        if verbose:
            logging.warning(f"Failed to save T-web classification to cache: {e}")

    return classification
