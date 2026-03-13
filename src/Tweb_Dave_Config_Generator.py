from pathlib import Path
import yaml
import argparse
from typing import Any

# path_to_snapshot: Path,
# path_for_results: Path,
# grid_size: int,
# density_field: bool,
# smooth_density_field: bool,
# calculate_potential_field: bool,
# calculate_traceless_tidal_tensor: bool,
# path_to_density_field: Path | None = None,
# smoothing_scales: list | float | None = None,


def assure_existence(path_to_file: Path):
    """Ensure the config directory exists and create the file if needed."""
    path_to_file.parent.mkdir(parents=True, exist_ok=True)
    if not path_to_file.exists():
        print("Creating a config file")
        path_to_file.touch()


def get_proper_args():
    parser = argparse.ArgumentParser(description="Generate TWEB DAVE configuration file")
    parser.add_argument("--snapshot", "-st", type=str, help="Path to the snapshot file")
    parser.add_argument("--results", "-rs", type=str, help="Path where to save the results")
    parser.add_argument("--grid-size", "-gs", type=int, default=256, help="Grid size (default: 256)")
    parser.add_argument("--density-field", "-df", action="store_true", help="Create density field")
    parser.add_argument("--smooth-density", "-sd", action="store_true", help="Smooth density field")
    parser.add_argument("--potential-field", "-pf", action="store_true", help="Calculate potential field")
    parser.add_argument("--tidal-tensor", "-tt", action="store_true", help="Calculate traceless tidal tensor")
    parser.add_argument("--density-path", "-dp", type=str, help="Path to load existing density field")
    parser.add_argument("--smoothing-scales", "-sc", nargs="+", type=float, help="Smoothing scales [h^-1 Mpc]")
    parser.add_argument(
        "--from-file",
        "-f",
        nargs="?",
        const="config/config.yaml",
        type=str,
        help="Load parameters from a YAML file (default: config/config.yaml)",
    )
    args = parser.parse_args()

    print("Running with args:")
    print(*vars(args).items())
    return args


def from_args_create_config(proper_args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    if proper_args.from_file:
        path_to_config_file = proper_args.from_file
        with open(path_to_config_file, "r") as file:
            config = yaml.safe_load(file)
    else:
        print("Try use command line arguments")
        config = {  # type: ignore
            "paths": {
                "snapshot_file": str(proper_args.snapshot),
                "results_directory": str(proper_args.results),
                "density_field_file": str(proper_args.density_path) if proper_args.density_path else None,
            },
            "grid": {"size": proper_args.grid_size},
            "density_field": {
                "create": proper_args.density_field,
                "smooth": proper_args.smooth_density,
                "smoothing_scales": proper_args.smoothing_scales if proper_args.smoothing_scales else None,
            },
            "calculations": {
                "potential_field": proper_args.potential_field,
                "traceless_tidal_tensor": proper_args.tidal_tensor,
            },
        }
    return config  # type: ignore


def create_tweb_dave_proper_config_file(config: dict[str, dict[str, Any]], path_to_config_file: Path) -> None:
    entries = {
        "Path to the snapshot file": str(config["paths"]["snapshot_file"]),
        "Path where you want to save the results": str(config["paths"]["results_directory"]),
        "Enter the grid size": config["grid"]["size"] if config["grid"]["size"] else 256,
        "Create the density field? (yes/no)": "yes" if config["density_field"]["create"] else "no",
        "Path to load the density field (optional)": (
            str(config["paths"]["density_field_file"]) if config["paths"]["density_field_file"] else ""
        ),
        "Smooth density field? (yes/no)": "yes" if config["density_field"]["smooth"] else "no",
        "Smoothing scales [h^-1 Mpc]": (
            config["density_field"]["smoothing_scales"] if config["density_field"]["smoothing_scales"] else ""
        ),
        "Calculate potential field? (yes/no)": "yes" if config["calculations"]["potential_field"] else "no",
        "Calculate traceless tidal tensor? (yes/no)": "yes" if config["calculations"]["traceless_tidal_tensor"] else "no",
    }

    with open(path_to_config_file, "w") as file:
        for ikey, ivalue in entries.items():
            if ivalue is not None:
                file.write(f"{ikey}: ")
                if isinstance(ivalue, (list, tuple)):
                    file.write(" ".join(map(str, ivalue)))  # type: ignore
                    file.write("\n")
                else:
                    file.write(str(ivalue))
                    file.write("\n")


def create_config_file(config: dict[str, dict[str, Any]], path_to_config_file: Path) -> None:
    with open(path_to_config_file, "w") as file:
        yaml.dump(config, file, default_flow_style=False, indent=4, sort_keys=False)


if __name__ == "__main__":
    path_to_tweb_dave_proper_config_file = Path("config/input_params.txt")
    path_to_config_file = Path("config/config.yaml")
    assure_existence(path_to_tweb_dave_proper_config_file)
    assure_existence(path_to_config_file)

    args = get_proper_args()
    try:
        config = from_args_create_config(args)
        print("Config created from arguments")
    except AttributeError as e:
        print("Printing error:")
        print(e)
        with open(path_to_config_file, "r") as file:
            config = yaml.safe_load(file)
        print("Config loaded from file")
    create_tweb_dave_proper_config_file(config, path_to_tweb_dave_proper_config_file)
    create_config_file(config, path_to_config_file)
