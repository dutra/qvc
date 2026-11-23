"""Shared sampler settings for joint QVC fits and the Pantheon benchmark."""
import numpy as np

SPEED_CHOICES = ("fastest", "quicker", "quick", "medium", "standard", "production")


def normalize_speed(speed):
    normalized = str(speed).strip().lower()
    if normalized in SPEED_CHOICES:
        return normalized
    else:
        raise ValueError(
            f"Invalid speed={speed!r}. Expected one of {SPEED_CHOICES}. "
            "Ordered fastest to slowest: fastest, quicker, quick, medium, standard, production."
        )


def get_dynesty_speed_settings(speed, ndim, *, warm_start=False):
    speed = normalize_speed(speed)
    if speed == "fastest":
        settings = dict(dlogz_init=10, n_effective=50, nlive_init=20, nlive_batch=5)
    elif speed == "quicker":
        settings = dict(dlogz_init=0.01, n_effective=500, nlive_init=50, nlive_batch=25)
    elif speed == "quick":
        settings = dict(dlogz_init=0.01, n_effective=5000, nlive_init=100, nlive_batch=50)
    elif speed == "medium":
        settings = dict(dlogz_init=0.01, n_effective=7500, nlive_init=175, nlive_batch=75)
    elif speed == "standard":
        settings = dict(dlogz_init=0.01, n_effective=10000, nlive_init=250, nlive_batch=100)
    elif speed == "production":
        settings = dict(
            dlogz_init=0.01,
            n_effective=10000,
            nlive_init=max(1000, 50 * ndim),
            nlive_batch=max(500, 25 * ndim),
        )
    else:
        raise ValueError(f"Invalid speed={speed!r}. Expected one of {SPEED_CHOICES}.")

    if not warm_start:
        return settings

    return {
        "dlogz_init": settings["dlogz_init"],
        "n_effective": max(25, int(np.ceil(0.10 * settings["n_effective"]))),
        "nlive_init": max(2 * ndim + 1, int(np.ceil(0.10 * settings["nlive_init"]))),
        "nlive_batch": max(5, int(np.ceil(0.10 * settings["nlive_batch"]))),
    }


def prior_transform_dynesty(unit_cube, priors, model_labels):
    return [priors[key][0] + (priors[key][1] - priors[key][0]) * x
            for x, key in zip(unit_cube, model_labels)]
