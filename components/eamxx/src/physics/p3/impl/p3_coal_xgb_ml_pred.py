"""Python version of the P3 coalescence XGBoost emulator.

This module translates the supplied Cython implementation into ordinary
Python. The model-loading and prediction calculations follow the original
code. The Cython entry point returned a ``double *``; Python cannot expose
that C pointer directly, so ``trigger_coalescence_emulator`` returns a
NumPy array containing the same 3 * pcols values instead.

Call :func:`python_init` once before requesting predictions. It loads the
models and scalers from the absolute paths used by the original source.
"""

import numpy as np
import xgboost
from pickle import load


# Model/scaler globals are populated by python_init(), as in the Cython
# module where initialization made these names available to the prediction
# function.
model_Nr = None
model_Nc = None
model_Qc = None
scaler_X = None
scaler_y_Qc = None

RainInit_model_Nr = None
RainInit_model_Nc = None
RainInit_model_Qc = None
RainInit_scaler_X = None
RainInit_scaler_y_Qc = None

SelfColl_model_Nr = None


def python_init():
    """Load all models and scalers required by the emulator."""
    global model_Nr, model_Nc, model_Qc, scaler_X, scaler_y_Qc
    global RainInit_model_Nr, RainInit_model_Nc, RainInit_model_Qc
    global RainInit_scaler_X, RainInit_scaler_y_Qc
    global SelfColl_model_Nr

    data_dir = (
        "/global/cfs/cdirs/mp193/llin/github_code/YPS_CC/cloudemulator_20250523_v2"
    )

    cloud_dir = f"{data_dir}/E3SMstylecldfeature-5min25umCP2"
    rain_init_dir = f"{data_dir}/RainInitRelVar-5min25um"
    self_coll_dir = f"{data_dir}/RainSelfCollectionTest1-5min25um"

    model_Nr = xgboost.XGBRegressor(tree_method="hist", n_estimators=289)
    model_Nr.load_model(
        f"{cloud_dir}/trained_model_filtered2_Jan13_DeltaNr.ubj"
    )

    model_Nc = xgboost.XGBRegressor(tree_method="hist", n_estimators=150)
    model_Nc.load_model(
        f"{cloud_dir}/trained_model_filtered2_Jan13_DeltaNc.ubj"
    )

    model_Qc = xgboost.XGBRegressor(tree_method="hist", n_estimators=116)
    model_Qc.load_model(
        f"{cloud_dir}/trained_model_filtered2_Jan13_DeltaQc.ubj"
    )

    # Use context managers so the pickle files are closed after loading.
    with open(f"{cloud_dir}/scaler_X.pkl", "rb") as stream:
        scaler_X = load(stream)
    with open(f"{cloud_dir}/scaler_y_Qc.pkl", "rb") as stream:
        scaler_y_Qc = load(stream)


    RainInit_model_Nr = xgboost.XGBRegressor(
        tree_method="hist", n_estimators=289
    )
    RainInit_model_Nr.load_model(
        f"{rain_init_dir}/trained_model_filtered2_transformed_DeltaNr.ubj"
    )

    RainInit_model_Nc = xgboost.XGBRegressor(
        tree_method="hist", n_estimators=150
    )
    RainInit_model_Nc.load_model(
        f"{rain_init_dir}/trained_model_filtered2_transformed_DeltaNc.ubj"
    )

    RainInit_model_Qc = xgboost.XGBRegressor(
        tree_method="hist", n_estimators=116
    )
    RainInit_model_Qc.load_model(
        f"{rain_init_dir}/trained_model_filtered2_transformed_DeltaQc.ubj"
    )

    with open(f"{rain_init_dir}/scaler_X.pkl", "rb") as stream:
        RainInit_scaler_X = load(stream)
    with open(f"{rain_init_dir}/scaler_y_Qc.pkl", "rb") as stream:
        RainInit_scaler_y_Qc = load(stream)

    SelfColl_model_Nr = xgboost.XGBRegressor(
        tree_method="hist", n_estimators=289
    )
    SelfColl_model_Nr.load_model(
        f"{self_coll_dir}/trained_model_filtered2_transformed_DeltaNr.ubj"
    )


def trigger_coalescence_emulator(pcols, nvars, ml_input):
    """Convert flattened input and return the three prediction fields.

    The original Cython function accepted ``double **ml_input`` and read
    ``nvars`` consecutive blocks of ``pcols`` values. This Python interface
    accepts any array-like object with the same flattened layout, reshapes it
    to ``(nvars, pcols)``, transposes to ``(pcols, nvars)``, and returns a
    NumPy array of length ``3 * pcols`` rather than a C pointer.
    """

    python_init()

    flat_input = np.asarray(ml_input, dtype=np.float64).reshape(-1)
    expected_size = nvars * pcols
    if flat_input.size != expected_size:
        raise ValueError(
            f"Expected {expected_size} input values, got {flat_input.size}"
        )

    input_data = flat_input.reshape(nvars, pcols).T
    return p3_coalescence_emulator(pcols, nvars, input_data)


def p3_coalescence_emulator(pcols, nvars, input_data):
    """Run the matching emulator based on the number of input variables."""

    python_init()

    input_data = np.asarray(input_data, dtype=np.float64)
    expected_shape = (pcols, nvars)
    if input_data.shape != expected_shape:
        raise ValueError(
            f"Expected input shape {expected_shape}, got {input_data.shape}"
        )

    if nvars == 6:
        # Four positive-valued features are log10 transformed; the final two
        # are copied directly, matching the original cloud emulator path.
        input_data2 = np.empty((pcols, 6), dtype=np.float64)
        input_data2[:, 0:4] = np.log10(input_data[:, 0:4])
        input_data2[:, 4:6] = input_data[:, 4:6]

        ypred_Nc = model_Nc.predict(input_data2)
        ypred_Nr = model_Nr.predict(input_data2)
        input_data_scaled = scaler_X.transform(input_data2)
        ypred_Qc_scaled = model_Qc.predict(input_data_scaled)

        # The source scaler expects a 2-D input, hence reshape to one row and
        # squeeze its inverse-transformed result back to one value per column.
        ypred_Qc = np.squeeze(
            scaler_y_Qc.inverse_transform(ypred_Qc_scaled.reshape(1, -1))
        )

        predictions = np.zeros(3 * pcols, dtype=np.float64)
        predictions[0:pcols] = np.power(10.0, ypred_Nc)
        predictions[pcols:2 * pcols] = (
            np.sign(ypred_Nr)
            * (np.power(10.0, np.abs(ypred_Nr)) - 1)
        )
        ### YPS removed 10**12 Oct/7/2026.
        predictions[2 * pcols:3 * pcols] = (
            np.power(10.0, ypred_Qc) * 10**12
        )
        return predictions

    if nvars == 3:
        # Rain initialization path: log10-transform the first two features;
        # preserve the third feature as supplied.
        input_data2 = np.empty((pcols, 3), dtype=np.float64)
        input_data2[:, 0:2] = np.log10(input_data[:, 0:2])
        input_data2[:, 2] = input_data[:, 2]

        ypred_Nc = RainInit_model_Nc.predict(input_data2)
        ypred_Nr = RainInit_model_Nr.predict(input_data2)
        input_data_scaled = RainInit_scaler_X.transform(input_data2)
        ypred_Qc_scaled = RainInit_model_Qc.predict(input_data_scaled)

        ypred_Qc = np.squeeze(
            RainInit_scaler_y_Qc.inverse_transform(ypred_Qc_scaled.reshape(1, -1))
        )

        predictions = np.zeros(3 * pcols, dtype=np.float64)
        predictions[0:pcols] = np.power(10.0, ypred_Nc)
        predictions[pcols:2 * pcols] = (
            np.sign(ypred_Nr)
            * (np.power(10.0, np.abs(ypred_Nr)) - 1)
        )
        ### YPS removed 10**12 Oct/7/2026.
        predictions[2 * pcols:3 * pcols] = (
            np.power(10.0, ypred_Qc) * 10**12
        )
        return predictions

    if nvars == 2:
        # Self-collection path: the original model receives three features.
        # Its second feature is left at zero and its third feature is set to
        # zero explicitly, while the first input feature is log10 transformed.
        input_data2 = np.zeros((pcols, nvars + 1), dtype=np.float64)
        input_data2[:, 0] = np.log10(input_data[:, 0])
        input_data2[:, 2] = 1.0

        ypred_Nr = SelfColl_model_Nr.predict(input_data2)

        predictions = np.zeros(3 * pcols, dtype=np.float64)
        predictions[0:pcols] = 10.0
        predictions[pcols:2 * pcols] = (
            np.sign(ypred_Nr)
            * (np.power(10.0, np.abs(ypred_Nr)) - 1)
        )
        predictions[2 * pcols:3 * pcols] = 10.0
        return predictions

    raise ValueError(f"Unsupported nvars={nvars}; expected 2, 3, or 6.")
