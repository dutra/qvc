"""Shared launcher configuration for the fixed-coverage completeness grid."""
import argparse
import math

MAG_WIDTH_ENV = "QVC_HUBBLE_COMPLETENESS_MAG_BIN_WIDTH"
Z_WIDTH_ENV = "QVC_HUBBLE_COMPLETENESS_Z_BIN_WIDTH"


def grid_bin_count(span, width):
    """Use the nearest integer bin count; retain the exact outer coverage."""
    width = float(width)
    if not math.isfinite(width) or width <= 0 or width > span:
        raise ValueError(f"Completeness bin width must be finite and in (0, {span}]; got {width}.")
    return max(1, int(math.floor(span / width + 0.5)))


def launcher_completeness_options(argv, environ):
    parser = argparse.ArgumentParser(description="Launch the configured Hubble campaign.")
    settings = (
        ("mag-bin-width", MAG_WIDTH_ENV, 0.1),
        ("z-bin-width", Z_WIDTH_ENV, 0.1),
        ("smooth-sigma-mag", "QVC_HUBBLE_COMPLETENESS_SMOOTH_SIGMA_MAG", 0.1),
        ("smooth-sigma-z", "QVC_HUBBLE_COMPLETENESS_SMOOTH_SIGMA_Z", 0.3),
    )
    for option, env, default in settings:
        parser.add_argument("--completeness-" + option, type=float,
                            default=environ.get(env, default),
                            help=f"Physical width (not pixels); default {default}, or {env}.")
    args = parser.parse_args(argv)
    resolved = {}
    for option, env, _ in settings:
        value = float(getattr(args, "completeness_" + option.replace("-", "_")))
        if not math.isfinite(value) or value < 0 or ("bin-width" in option and value == 0):
            parser.error(f"--completeness-{option} must be finite and {'positive' if 'bin-width' in option else 'nonnegative'}")
        resolved[env] = str(value)
    try:
        nm = grid_bin_count(8.0, resolved[MAG_WIDTH_ENV])
        nz = grid_bin_count(4.5, resolved[Z_WIDTH_ENV])
        if float(resolved[MAG_WIDTH_ENV]) > 1:
            raise ValueError("Magnitude bins must be at most 1 mag wide to preserve the padded science support.")
    except ValueError as exc:
        parser.error(str(exc))
    print(f"Completeness grid: {nm} magnitude bins (width {8/nm:g} mag), "
          f"{nz} redshift bins (width {4.5/nz:g}); "
          f"smoothing sigma_mag={resolved[settings[2][1]]}, sigma_z={resolved[settings[3][1]]}.")
    return resolved


def add_grid_arguments(parser):
    """Register grid options on both the early and full scientific CLI parsers."""
    parser.add_argument('--completeness-mag-bin-width', type=float, default=None,
                        help='Target completeness magnitude bin width (default 0.1 mag, or environment).')
    parser.add_argument('--completeness-z-bin-width', type=float, default=None,
                        help='Target completeness redshift bin width (default 0.1, or environment).')


def configure_grid_from_argv(argv, environ):
    """Resolve CLI widths before importing modules with grid-dependent constants."""
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    add_grid_arguments(parser)
    args, _ = parser.parse_known_args(argv)
    for value, env, span in (
        (args.completeness_mag_bin_width, MAG_WIDTH_ENV, 8.0),
        (args.completeness_z_bin_width, Z_WIDTH_ENV, 4.5),
    ):
        if value is None:
            continue
        try:
            n = grid_bin_count(span, value)
            if env == MAG_WIDTH_ENV and value > 1:
                raise ValueError('Magnitude bins must be at most 1 mag wide to preserve the padded science support.')
        except ValueError as exc:
            parser.error(str(exc))
        environ[env] = str(value)
        print(f'Completeness CLI grid: {env}={value:g}; {n} bins, actual width {span/n:g}.')
