import argparse
import logging
from dataclasses import dataclass
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
    precision_recall_fscore_support,
    accuracy_score,
    balanced_accuracy_score,
)
import sys

DEFAULT_TIME_BUDGET = 3600
DEFAULT_RANDOM_STATE = 57287
DEFAULT_N_SPLITS = 5

from datetime import datetime
import joblib
import lightgbm as lgb
import os
import sys
sys.path.append("/n/groups/patel/randy")
from sklearn.preprocessing import scale
import rfb.code.ukb_func.ml_utils as ml_utils
from proteomics_idp.code.ml.calculate_performance import save_labels_predictions
from flaml import AutoML
from flaml.automl.data import get_output_from_log
from typing import Dict, List, Tuple, Optional, Any


@dataclass
class ExperimentConfig:

    """Configuration for ML experiments."""
    path: str = "/n/groups/patel/randy/rfb/tidy_data/UKBiobank/all_outcomes"
    df = pd.read_parquet(
        f"{path}/proteomics_first_occurrences.parquet", engine='fastparquet')

    proteins = pd.read_csv(
        f"{path}/protein_colnames.txt", header=None)[0].tolist()

    outcomes = pd.read_csv(
        f"{path}/outcome_colnames.txt", header=None)[0].tolist()

    demographics = pd.read_parquet(
        "/n/groups/patel/randy/proj_idp/tidy_data/demographics/demographics_df.parquet"
    )

    results_base_dir: str = "/n/groups/patel/adithya/UKB_Ratios/ukb_ratios/results/"

    # AutoML settings
    default_metric: str = "log_loss"
    default_model: str = "lgbm"

    def get_results_dir(self, experiment: str,
                        outcome_index: int, fold_index: int) -> str:
        """Get the results directory path for a specific experiment."""
        return (f"{self.results_base_dir}/{experiment}/individual_results/"
                f"outcome_{outcome_index}/fold_{fold_index}/")


def parse_args():
    # Create the argument parser
    parser = argparse.ArgumentParser(
        description='''Comparing proteomics to demographics
        baselines and examining the impact of data leakage
        '''
    )

    # Add arguments
    # parser.add_argument("--experiment", type=str,
    #                     help="Experiment name"
    #                     "Options: proteins_alone, demographics_alone, "
    #                     "demographics_proteins, demographics_leakage, "
    #                     "proteins_leakage, "
    #                     "demographics_proteins_leakage, "
    #                     "both_leakage",
    #                     "protein_ratios_alone, protein_ratios_demographics")
    
    parser.add_argument("--experiment", type=str,
            choices=[
                "proteins_alone","demographics_alone","demographics_proteins",
                "demographics_leakage","proteins_leakage",
                "demographics_proteins_leakage","both_leakage",
                "protein_ratios_alone","protein_ratios_demographics"
            ],
            help="Experiment name"
        )

    parser.add_argument("--outcome", type=int, help="Outcome number")
    parser.add_argument("--fold", type=int, default=0,
                        help="Fold number for cross-validation")
                        

    # Parse the arguments
    args = parser.parse_args()

    return args


def pull_proteins(df, proteins):
    return df.loc[:, ['eid'] + proteins]


def scale_proteomics(proteomics):

    noeid_proteomics = proteomics.drop('eid', axis=1)

    # scale all columns except eid
    scaled_proteomics = pd.DataFrame(scale(noeid_proteomics),
                                     columns=noeid_proteomics.columns)

    # join back with eid column
    df = pd.concat([proteomics[['eid']], scaled_proteomics], axis=1)
    return df


def pull_demographics(df, demographics_df):
    demo = demographics_df.loc[:, ['eid', '21003-0.0', '31-0.0', '845-0.0']]
    demo = demo[demo.eid.isin(df.eid)]
    return demo


# Since the UKB alraedy log2 normalzies, we will not perform any further log normalization for now
def compute_log_protein_ratios(proteomics_df: pd.DataFrame, protein_names: List[str]) -> pd.DataFrame:
    """
    Build pairwise log2-ratio features (NPX_i - NPX_j) for i<j only.
    Olink NPX values are already log2-transformed, so subtraction is equivalent
    to computing log2 ratios in linear space.
    """
    missing = [p for p in protein_names if p not in proteomics_df.columns]
    if missing:
        raise ValueError(f"Missing protein columns in proteomics_df: {missing[:10]}{'...' if len(missing) > 10 else ''}")

    eid = proteomics_df[['eid']].copy()
    P = proteomics_df[protein_names].astype(float)

    n = len(protein_names)
    ratio_cols = {}
    for i in range(n):
        for j in range(i + 1, n):
            col_name = f"{protein_names[i]}__minus__{protein_names[j]}"
            # direct subtraction: NPX_i - NPX_j == log2(p_i / p_j)
            ratio_cols[col_name] = P.iloc[:, i] - P.iloc[:, j]

    ratios_df = pd.DataFrame(ratio_cols, index=proteomics_df.index)
    return pd.concat([eid, ratios_df], axis=1)

# ensure the accuracy of this function in the juypter notebook

# Function to pull predicitive featrues from training data
def get_selected_proteins(outcome_idx: int, base_dir: str) -> list[str]:
    """
    Read feature_importances.csv files from the proteins_alone experiment
    and select proteins that were non-zero in ≥2 folds.
    """
    from collections import defaultdict

    feature_counts = defaultdict(int)
    n_folds = 5
    base_path = os.path.join(base_dir, "outcome_" + str(outcome_idx))

    for fold in range(n_folds):
        csv_path = os.path.join(base_dir, f"outcome_{outcome_idx}", f"fold_{fold}", "feature_importances.csv")
        if not os.path.exists(csv_path):
            continue
        try:
            df = pd.read_csv(csv_path)
            protein_col, importance_col = df.columns[:2]
            for name, imp in zip(df[protein_col], df[importance_col]):
                if imp > 0:
                    feature_counts[name] += 1
        except Exception as e:
            print(f"Failed to read fold {fold} for outcome {outcome_idx}: {e}")

    # Proteins with nonzero importance in ≥ 2 folds
    selected = [p for p, count in feature_counts.items() if count >= 2]
    print(f" Outcome {outcome_idx}: {len(selected)} proteins selected.")
    return selected



def pull_predictors(df, proteins, demographics_df, experiment):
    if experiment not in [
        'proteins_alone', 'demographics_alone',
            'demographics_proteins', 'proteins_leakage',
            'demographics_leakage',
            'demographics_proteins_leakage', 'both_leakage',
            'protein_ratios_alone', 'protein_ratios_demographics'  # <-- ADDED
    ]:
        raise ValueError(f"Unknown experiment: {experiment}")

    if experiment == 'proteins_alone':
        data = pull_proteins(df, proteins)

    elif experiment == 'demographics_alone':
        data = pull_demographics(df, demographics_df)

    elif experiment == 'demographics_proteins':
        proteomics = pull_proteins(df, proteins)
        demographics = pull_demographics(df, demographics_df)
        data = proteomics.merge(demographics)

    elif experiment == 'demographics_leakage':
        demographics = pull_demographics(df, demographics_df)
        demographics = scale_proteomics(demographics)
        data = demographics

    elif experiment == 'proteins_leakage':
        proteomics = pull_proteins(df, proteins)
        data = scale_proteomics(proteomics)

    elif experiment == 'demographics_proteins_leakage':
        proteomics = pull_proteins(df, proteins)
        proteomics = scale_proteomics(proteomics)

        demographics = pull_demographics(df, demographics_df)
        data = proteomics.merge(demographics)

    elif experiment == 'both_leakage':
        demographics = pull_demographics(df, demographics_df)
        demographics = scale_proteomics(demographics)

        proteomics = pull_proteins(df, proteins)
        proteomics = scale_proteomics(proteomics)
        data = proteomics.merge(demographics)
    
    elif experiment == 'protein_ratios_alone':
        proteomics = pull_proteins(df, proteins)
        data = compute_log_protein_ratios(proteomics, proteins)

    elif experiment == 'protein_ratios_demographics':
        proteomics = pull_proteins(df, proteins)
        proteomics = compute_log_protein_ratios(proteomics, proteins)
        demographics = pull_demographics(df, demographics_df)
        data = proteomics.merge(demographics)

    # elif experiment == 'protein_ratios_alone':
    #     proteomics = pull_proteins(df, proteins)
    #     data = compute_protein_ratios(proteomics, proteins)

    # # --- NEW: protein ratios + demographics (no demographic ratios) ---
    # elif experiment == 'protein_ratios_demographics':
    #     proteomics = pull_proteins(df, proteins)
    #     proteomics = compute_protein_ratios(proteomics, proteins)
    #     demographics = pull_demographics(df, demographics_df)
    #     data = proteomics.merge(demographics)

    return data


def pull_outcome(df, outcomes, outcome_idx):
    y = df.loc[:, ['eid', outcomes[outcome_idx]]].copy()
    y['label'] = y[outcomes[outcome_idx]].notna().astype(int)
    return y


def check_x_and_y(x, y):
    common_eid = set(x.eid).intersection(y.eid)

    x = x[x.eid.isin(common_eid)]
    y = y[y.eid.isin(common_eid)]

    # make sure eid column is identical in x and y
    # Sort both dataframes by eid and reset index to ensure alignment
    x = x.sort_values('eid').reset_index(drop=True)
    y = y.sort_values('eid').reset_index(drop=True)

    x['eid'] = x['eid'].astype(int)
    y['eid'] = y['eid'].astype(int)

    # Verify that eid columns are identical
    assert x['eid'].equals(y['eid']), \
        "EID columns are not identical after sorting"

    return x, y


def settings_automl(time_budget: int,
                    metric: str, model: str) -> Dict[str, Any]:
    """
    Generate settings for an AutoML regression task.

    Parameters:
    -----------
    experiment : str
        The experiment type (used to determine if feature selection is
        being performed)
    time_budget : int
        The time budget for the AutoML process in seconds
    metric : str
        The evaluation metric to be used (e.g., 'mse', 'r2')
    model : str
        The model to be used in the AutoML process (e.g., 'lgbm', 'xgboost')

    Returns:
    --------
    Dict[str, Any]
        A dictionary containing the settings for the AutoML process
    """
    # if "fs_" in experiment:
    #     automl_settings: Dict[str, Any] = {
    #         "task": "regression",
    #         "train_full": True,
    #         "n_jobs": -1,
    #         "train_best": True,
    #     }
    # else:
    automl_settings: Dict[str, Any] = {
        "task": "classification",
        "time_budget": time_budget,
        "metric": metric,
        "n_jobs": -1,
        "eval_method": "cv",
        "n_splits": DEFAULT_N_SPLITS,
        "early_stop": True,
        "log_training_metric": True,
        # "model_history": True,
        "seed": DEFAULT_RANDOM_STATE,
        "estimator_list": [model],
    }

    return automl_settings




def main():

    config = ExperimentConfig()
    df = config.df
    proteins = config.proteins
    demographics_df = config.demographics
    outcomes = config.outcomes

    args = parse_args()
    experiment = args.experiment
    outcome_idx = args.outcome
    fold_idx = args.fold
    print(f"Running experiment {experiment}, "
          f"outcome number {outcome_idx}, "
          f"fold number {fold_idx}")
    
    # Automatically pull outcome-specific proteins based on prior results if we are using the ratio based models -- avoids enormous numbers of features
    if experiment in ["protein_ratios_alone", "protein_ratios_demographics"]:
        feature_base = "/n/groups/patel/randy/intl_congress_peer_review_2025/results/proteins_alone/individual_results"
        proteins = get_selected_proteins(outcome_idx, feature_base)
        if not proteins:
            raise ValueError(f"No selected proteins found for outcome {outcome_idx}")

    # Create results directory
    results_dir = config.get_results_dir(experiment, outcome_idx, fold_idx)
    os.makedirs(results_dir, exist_ok=True)

    # Set up logging
    logging.basicConfig(
        filename=f"{results_dir}/logging.txt",
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filemode='w'  # Overwrite log file for each run
    )

    x = pull_predictors(df, proteins, demographics_df, experiment)
    y = pull_outcome(df, outcomes, outcome_idx)
    x, y = check_x_and_y(x, y)

    x = x.drop(columns=['eid'])
    y = y.drop(columns=['eid'])

    skf = StratifiedKFold(n_splits=DEFAULT_N_SPLITS, shuffle=True,
                          random_state=DEFAULT_RANDOM_STATE)

    # Initialize lists to store results
    # train_res_list = []
    # test_res_list = []

    for j, (train_idx, val_idx) in enumerate(
                skf.split(x, y.label)
            ):

        if j == fold_idx:
            logging.info(f"Starting fold {j}")

            x_train, x_val = x.iloc[train_idx], x.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx].label.values, \
                y.iloc[val_idx].label.values

            logging.info(f"Training model: {datetime.now().time()}")

            automl = AutoML()
            automl_settings = settings_automl(
                DEFAULT_TIME_BUDGET, metric=config.default_metric,
                model=config.default_model
            )
            logging.info(f"AutoML settings: {automl_settings}")

            automl_settings["log_file_name"] = \
                f"{results_dir}/results_log.json"
            logging.info(f"Fitting model: {datetime.now().time()}")

            try:
                automl.fit(x_train, y_train, **automl_settings)
                logging.info(
                    f"Done fitting model: {datetime.now().time()}")
            except Exception as e:
                logging.error(f"Error fitting AutoML model: {e}")
                raise

            # Save model and predictions
            try:
                logging.info(f"Saving the model: {datetime.now().time()}")
                best_model = automl.model.estimator
                joblib.dump(best_model,
                            f"{results_dir}/flaml_best_model.joblib")

                # Save model metadata
                model_info = {
                    'best_estimator': str(type(best_model).__name__),
                    'best_config': automl.best_config,
                    'best_loss': automl.best_loss,
                    'training_time': DEFAULT_TIME_BUDGET,
                }

                import json
                with open(f"{results_dir}/model_info.json", 'w') as f:
                    json.dump(model_info, f, indent=2)

                train_preds = automl.predict_proba(x_train)[:, 1]
                test_preds = automl.predict_proba(x_val)[:, 1]

                save_labels_predictions(
                    y_val, test_preds,
                    f"{results_dir}/", "test")
                logging.info(
                    f"Test predictions saved: {datetime.now().time()}")

                # Train set predictions
                save_labels_predictions(
                    y_train, train_preds,
                    f"{results_dir}/", "train")
                logging.info(
                    f"Train predictions saved: {datetime.now().time()}")

                # save model
                joblib.dump(automl, f"{results_dir}/flaml_best_model.joblib")
                logging.info(f"Model saved: {datetime.now().time()}")

            except Exception as e:
                logging.error(f"Error saving model or predictions: {e}")
                raise

            # clf = lgb.LGBMClassifier(random_state=DEFAULT_RANDOM_SEED)
            # clf.fit(x_train, y_train)

            # train_preds = clf.predict_proba(x_train)
            # test_preds = clf.predict_proba(x_val)

            # train_res, threshold = ml_utils.calc_results(y_train,
            # train_preds)
            # test_res = ml_utils.calc_results(y_val, test_preds,
            #                                  threshold=threshold)

            # train_res_list.append(pd.concat(
            #     [outcomes[outcome_idx], fold_idx],
            #     train_res))

            # test_res_list.append(pd.concat(
            #     [outcomes[outcome_idx], fold_idx],
            #     test_res))

            # Test set predictions
            # save_labels_predictions(
            #     y_val.values, test_preds, f"{results_dir}/", "test")
            # logging.info(f"Test predictions saved: {datetime.now().time()}")

            # # Train set predictions
            # save_labels_predictions(
            #     y_train.values, train_preds, f"{results_dir}/", "train")
            # logging.info(f"Train predictions saved: {datetime.now().time()}")

            # # save model
            # joblib.dump(clf, f"{results_dir}/flaml_best_model.joblib")
            # logging.info(f"Model saved: {datetime.now().time()}")

        # # save results files
        # train_results_df = pd.concat(train_res_list)
        # test_results_df = pd.concat(test_res_list)

        # train_results_df.to_csv(f"{results_dir}/train_results.csv",
        #                         index=False)
        # test_results_df.to_csv(f"{results_dir}/test_results.csv",
        #                         index=False)


if __name__ == "__main__":
    main()