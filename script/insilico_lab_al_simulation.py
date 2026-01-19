#############################################################################################
#
# Insilico Active Learning Simulation Script
#
# This script simulates an active learning experiment in a controlled environment.
# It allows for the configuration of various parameters and the execution of the experiment.
#
#############################################################################################

import yaml
import argparse
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple
from coacervopti.utils import save_to_json
from coacervopti.sampling import sample_landscape
from coacervopti.utils import create_strict_folder
from coacervopti.experiment import (sampling_block, 
                                  setup_ml_model, 
                                  validation_block, 
                                  setup_data_pool,
                                  get_gt_dataframes,
                                  remove_evidence_from_gt,
                                  setup_experiment_variables,
                                  AcquisitionParametersGenerator)

# TODO: implement extra features such as the ones in benchmark_experiment.py
# - multiple acquisition functions per cycle
# - protocol for acquisition functions per cycle

# FUNCTIONS

def create_insilico_al_experiment_paths(
        exp_name: str,
        pool_dataframe: pd.DataFrame,
        search_space_variables: List[str],
        evidence_dataframe: pd.DataFrame = None,
        dataset_path_name: str = 'dataset',
        overwrite: bool = False
    ) -> Tuple[Path, Path, Path]:
    """Initializes the experiment paths for the insilico active learning simulation.

    Args:
        exp_name (str): experiment name.
        pool_dataframe (pd.DataFrame): ground truth dataframe.
        search_space_variables (List[str]): search space variables.
        evidence_dataframe (pd.DataFrame, optional): starting evidence dataframe. Defaults to None.

    Returns:
        Tuple[Path, Path, Path]: Paths for the experiment, pool CSV, and candidates CSV.
    """
    from coacervopti.format import INSILICO_AL_REPO
    insilico_al_path = INSILICO_AL_REPO / exp_name

    # Create the experiment folder
    create_strict_folder(path_str=str(insilico_al_path), overwrite=overwrite)

    # Create the dataset folder
    dataset_path = insilico_al_path / dataset_path_name
    create_strict_folder(path_str=str(dataset_path))

    # Create the main experiment paths
    pool_csv_path = dataset_path / f'{exp_name}_POOL.csv'
    candidates_csv_path = dataset_path / f'{exp_name}_CANDIDATES.csv'
    train_csv_path = dataset_path / f'{exp_name}_TRAIN.csv'

    # Pool contains only the search space variables form the ground truth dataframe
    assert all(var in pool_dataframe.columns for var in search_space_variables), \
        f"Search space variables {search_space_variables} not found in ground truth dataframe columns."
    pool_df = pool_dataframe[search_space_variables]
    pool_df.to_csv(pool_csv_path, index=False)

    candidates_df = remove_evidence_from_gt(pool_df, evidence_df, search_space_variables)
    candidates_df.to_csv(candidates_csv_path, index=False)

    if evidence_dataframe is not None:
        evidence_dataframe.to_csv(dataset_path / f'{exp_name}_EVIDENCE.csv', index=False)

    return insilico_al_path, pool_csv_path, candidates_csv_path, train_csv_path


def data_scaler_setup(config: dict):
    """Sets up the data scaler based on the configuration.
    Args:
        config (dict): Configuration dictionary.
    Returns:
        BaseEstimator: The data scaler instance.
    """
    from sklearn.preprocessing import StandardScaler, MinMaxScaler

    data_scaler_type = config.get('data_scaler', 'StandardScaler')
    data_scaler_params = config.get('data_scaler_params', {})

    if data_scaler_type == 'StandardScaler':
        scaler = StandardScaler(**data_scaler_params)
    elif data_scaler_type == 'MinMaxScaler':
        scaler = MinMaxScaler(**data_scaler_params)
    else:
        raise ValueError(f"Unsupported data scaler type: {data_scaler_type}")
    
    return scaler

# MAIN

if __name__ == '__main__':
    # Parse the config.yaml
    parser = argparse.ArgumentParser(description="Read a YAML config file.")
    parser.add_argument("-c", "--config", required=True, help="Path to the YAML configuration file")
    parser.add_argument("--rerun", action='store_true', help="Rerun the experiment even if the benchmark folder exists")
    args = parser.parse_args()

    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    # --------------------------------------------------------------------------------
    # SETTING UP THE EXPERIMENT
    DATASET_PATH_NAME = 'dataset'

    (EXP_NAME,           # Experiment name
     ADDITIONAL_NOTES,   # Additional notes for the experiment
     N_CYCLES,           # Number of active learning cycles
     INIT_BATCH,         # Initial batch size
     INIT_SAMPLING,      # Initial sampling method
     ACQUI_PARAMS,       # Acquisition function parameters
     SEARCH_VAR,         # Search space variables
     TARGET_VAR) = setup_experiment_variables(config)

    # read pre-defined ground truth and evidence dataframes
    gt_file = config.get('ground_truth_file', None)
    evidence_file = config.get('experiment_evidence', None)
    if evidence_file is None:
        # Correction for cycle 0 sampling
        N_CYCLES += 1

    if gt_file is not None:
        gt_df, evidence_df = get_gt_dataframes(gt_file, evidence_file)
    
    # ---
    # Paths for the experiment
    INSILICO_AL_PATH, POOL_CSV_PATH, CANDIDATES_CSV_PATH, TRAIN_CSV_PATH = create_insilico_al_experiment_paths(
        exp_name=EXP_NAME,
        pool_dataframe=gt_df,
        search_space_variables=SEARCH_VAR,
        evidence_dataframe=evidence_df,
        dataset_path_name=DATASET_PATH_NAME,
        overwrite=True if args.rerun else False
    )
    
    # ---
    # Acquisition protocol
    ACQUI_PROTOCOL = config.get('acquisition_protocol', None)

    print(f"Experiment Name: {EXP_NAME}\n->\t{INSILICO_AL_PATH}")
    print(f"Search Variables: {SEARCH_VAR}")
    print(f"Target Variable: {TARGET_VAR}")
    print(f"Initial Sampling: {INIT_SAMPLING} with {INIT_BATCH} points")
    print(f"Acquisition Protocol: {ACQUI_PROTOCOL}")

    # ---
    # Set up the data scaler
    scaler_path = INSILICO_AL_PATH / DATASET_PATH_NAME / 'scaler.joblib'
    pool_df = pd.read_csv(POOL_CSV_PATH)

    data_scaler_type = config.get('data_scaler', None)
    if data_scaler_type is None:
        print("Data scaler type not specified in config file. Using StandardScaler as default.")
        data_scaler_type = "StandardScaler"

    data_scaler = data_scaler_setup(config)
    X_pool, scaler = setup_data_pool(df=pool_df, search_var=SEARCH_VAR, scaler=data_scaler)
    joblib.dump(data_scaler, scaler_path)

    # --- 
    # Batch selection strategy
    batch_selection_config = config.get('batch_selection', None)
    if batch_selection_config is not None:
        batch_selection_method = batch_selection_config.get('method', 'highest_landscape')
        batch_selection_params = batch_selection_config.get('method_params') or {}
        print(f"Batch selection strategy: {batch_selection_method} with params: {batch_selection_params}")
    else:
        raise ValueError("Batch selection configuration must be provided in the config file, with at least the method defined.")
    
    CYCLE_SAMPLING = batch_selection_params.get('sampling_method')

    # ---
    # Landscape penalization
    # Set up landscape penalization parameters
    landscape_penalization = config.get('landscape_penalization', None)
    if landscape_penalization is not None:
        pen_radius = landscape_penalization.get('radius', None)
        pen_strength = landscape_penalization.get('strength', None)
        print(f"Landscape penalization activated with radius {pen_radius} and strength {pen_strength}.")

    # ---
    # Set up the model
    ML_MODEL = setup_ml_model(config)

    # --------------------------------------------------------------------------------
    # RUN THE EXPERIMENT

    print('# ----------------------------------------------------------------------------\n'\
          f'# \tExperiment: {EXP_NAME} \t\n'\
          '# ----------------------------------------------------------------------------\n'\
          f'Additional notes: {ADDITIONAL_NOTES}\n'
          f'Stored at: {INSILICO_AL_PATH}'
          )

    # Init the acquisition parameters generator
    acqui_param_gen = AcquisitionParametersGenerator(
        acquisition_params=ACQUI_PARAMS,
        acquisition_protocol=ACQUI_PROTOCOL
    )

    # AL cycles    
    for cycle in range(N_CYCLES):

        # --- CYCLE 0
        # Candidates are the pool of points availalble for sampling at the current cycle
        # For cycle 0, candidates are the whole pool and for the next cycles, 
        # candidates are the points not yet sampled.
        if cycle == 0:
            print(f'\n# Cycle {cycle} - Initial sampling of new experiment')
            # Load the total pool dataframe
            candidates_df = pd.read_csv(CANDIDATES_CSV_PATH)

        elif cycle > 0:
            print(f'\n# Cycle {cycle} of {N_CYCLES-1}')
            # Load candidates dataframe
            if not CANDIDATES_CSV_PATH.exists():
                raise FileNotFoundError(f'Candidates CSV file not found at {CANDIDATES_CSV_PATH}. Please ensure it exists.')
            # Load candidates dataframe
            candidates_df = pd.read_csv(CANDIDATES_CSV_PATH)

        if not TRAIN_CSV_PATH.exists():
            print('Previous training CSV file not found.\nTraining from scratch.')
            train_df = pd.DataFrame(columns=candidates_df.columns)
        else:
            # Load training dataframe
            train_df = pd.read_csv(TRAIN_CSV_PATH)

        # Removing training dataframe from the candidates dataframe
        # This way we can start from a pre-defined set of training points
        # Given that the training dataframe is a subset of the candidates dataframe
        candidates_df = candidates_df[~candidates_df[SEARCH_VAR].apply(tuple, axis=1).isin(train_df[SEARCH_VAR].apply(tuple, axis=1))]

        # Create the cycle output folder and save the outputs
        cycle_output_path = INSILICO_AL_PATH / Path(f'cycle_{cycle}')
        create_strict_folder(path_str=str(cycle_output_path))

        # -------------------------------------- #
        # --- INIT of cycle 0
        if cycle == 0:

            X_candidates = scaler.transform(candidates_df[SEARCH_VAR].to_numpy())
            print(f'Candidates shape: {X_candidates.shape}, Sampling: {INIT_SAMPLING} with {INIT_BATCH} points')

            # Sample initial points from the candidates
            # using the sampling mode defined in the experiment setup
            screened_indexes = sample_landscape(
                X_landscape=X_candidates, 
                n_points=INIT_BATCH,
                sampling_mode=INIT_SAMPLING
            )

            cycle_0_log_dict = {
                'cycle' : cycle,
                'init_sampling_mode' : INIT_SAMPLING,
                'n_points' : INIT_BATCH,
                'screened_indexes' : np.array(screened_indexes).astype(int).tolist(),
                'candidates_df_shape' : candidates_df.shape,
                'train_df_shape' : train_df.shape,
            }
            save_to_json(
                dictionary=cycle_0_log_dict,
                fout_name=cycle_output_path / Path('cycle_0_log.json'),
                timestamp=False
            )
        # --- END of cycle 0
        # -------------------------------------- #

        # -------------------------------------- #
        # --- INIT of cycle > 0
        if cycle > 0:

            # Get the acquisition parameters for the current cycle
            # Predefined acquisition modes for tracking the source of acquisition of each point
            predefined_acquisition_modes = []
            cycle_acqui_params = acqui_param_gen.get_params_for_cycle(cycle-1)
            for acp in cycle_acqui_params:
                predefined_acquisition_modes.extend([acp['acquisition_mode']] * acp['n_points'])
            
            # Prepare candidates and training data
            candidates_df = candidates_df.reset_index(drop=True)
            train_df = train_df.reset_index(drop=True)
            X_candidates = scaler.transform(candidates_df[SEARCH_VAR].to_numpy())
            y_train = train_df[TARGET_VAR].to_numpy()
            X_train = scaler.transform(train_df[SEARCH_VAR].to_numpy())

            print(f'Candidates shape: {X_candidates.shape}, Training shape: {X_train.shape}\n'
                  f'Acquisition modes: {np.unique(predefined_acquisition_modes)}')

            # Compute the best target value from the training set
            y_best = max(y_train).item()

            # Train model on evidence and predict on pool to generate
            # the outputs per cycle
            ML_MODEL.train(X_train, y_train)
            _, y_pred, y_unc = ML_MODEL.predict(X_pool)

            # Sample new points from candidates based on the acquisition function
            screened_indexes, landscape = sampling_block(
                X_candidates=X_candidates, 
                X_train=X_train,
                y_train=y_train,
                ml_model=ML_MODEL,
                acquisition_params=cycle_acqui_params,
                batch_selection_method=batch_selection_method,
                batch_selection_params=batch_selection_params,
                penalization_params=(pen_radius, pen_strength) if landscape_penalization is not None else None
            )

            # get the total batch size for logging
            N_BATCH = sum(acq_param['n_points'] for acq_param in cycle_acqui_params)

            # Save the cycle log
            cycle_log_dict = {
                'cycle' : cycle,
                'sampling_mode' : CYCLE_SAMPLING,
                'n_points' : N_BATCH,
                'acquisition_params' : cycle_acqui_params,
                'screened_indexes' : np.array(screened_indexes).astype(int).tolist(),
                'candidates_df_shape' : candidates_df.shape,
                'train_df_shape' : train_df.shape,
                'y_best' : int(y_best),
                'nll' : ML_MODEL.model.log_marginal_likelihood().astype(float),
                'model_params' : ML_MODEL.__repr__()
            }
            save_to_json(
                dictionary=cycle_log_dict,
                fout_name=cycle_output_path / Path(f'cycle_{cycle}_log.json'),
                timestamp=False
            )
            
            # Save the landscape and predictions &
            # Add landscapes as columns with acquisition parameter names
            model_predictions_df = pd.DataFrame({
                **{col: pool_df[col] for col in SEARCH_VAR},
                'y_pred': y_pred,
                'y_uncertainty': y_unc
            })

            model_landscapes_df = pd.DataFrame({
                **{col: candidates_df[col] for col in SEARCH_VAR}
            })
            for i, acqui_param in enumerate(cycle_acqui_params):
                col_name = f"landscape_{acqui_param['acquisition_mode']}"
                model_landscapes_df[col_name] = landscape[i]

            model_predictions_df.to_csv(cycle_output_path / Path(f'cycle_{cycle}_predictions.csv'), index=False)
            model_landscapes_df.to_csv(cycle_output_path / Path(f'cycle_{cycle}_landscapes.csv'), index=False)

            train_df.to_csv(cycle_output_path / Path(f'X_train_cycle_{cycle}.csv'), index=False)

            # Save model snapshot
            model_snapshot_path = cycle_output_path / Path(f'model_snapshot_cycle_{cycle}_joblib.pkl')
            joblib.dump(ML_MODEL, model_snapshot_path)
        # --- END of cycle > 0
        # -------------------------------------- #

        # Update training set
        next_df = candidates_df.iloc[screened_indexes][SEARCH_VAR]
        next_df.to_csv(cycle_output_path / Path(f'cycle_{cycle}_output_sampled.csv'), index=False)

        # Validation block
        validated_df = validation_block(
            gt_df=gt_df, 
            sampled_df=next_df, 
            search_vars=SEARCH_VAR
        )
        validated_df.to_csv(cycle_output_path / Path(f'cycle_{cycle}_validated.csv'), index=False)

        # Merge the new validated data with the existing training data
        if cycle == 0:
            train_df = validated_df
        elif cycle > 0:
            train_df = pd.concat([train_df, validated_df], ignore_index=True)
        train_df.to_csv(TRAIN_CSV_PATH, index=False)

        # Update candidates dataframe removing the sampled points
        candidates_df = candidates_df.drop(index=screened_indexes)
        candidates_df.to_csv(CANDIDATES_CSV_PATH, index=False)

    # END of AL cycle
