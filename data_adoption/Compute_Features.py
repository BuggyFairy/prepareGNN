"""This module is used for computing social and map features for motion forecasting baselines.
Example usage:
    $ python compute_features.py --data_dir ~/val/argodata
        --feature_dir ~/val/features --mode val
"""

import os
import shutil
import tempfile
import time
from typing import Any, Dict, List, Tuple

import argparse
from joblib import Parallel, delayed
import numpy as np
import pandas as pd

from data_adoption.Feature_Config import RAW_DATA_FORMAT, _FEATURES_SMALL_SIZE
from data_adoption.map.MapFeaturesUtils import MapFeaturesUtils
from data_adoption.social.SocialFeaturesUtils import SocialFeaturesUtils


def parse_arguments() -> Any:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        default="",
        type=str,
        help="Directory where the sequences (csv files) are saved",
    )
    parser.add_argument(
        "--feature_dir",
        default="",
        type=str,
        help="Directory where the computed features are to be saved",
    )
    parser.add_argument("--mode",
                        required=True,
                        type=str,
                        help="train/val/test")
    parser.add_argument(
        "--batch_size",
        default=100,
        type=int,
        help="Batch size for parallel computation",
    )
    parser.add_argument("--obs_len",
                        default=20,
                        type=int,
                        help="Observed length of the trajectory")
    parser.add_argument("--pred_len",
                        default=30,
                        type=int,
                        help="Prediction Horizon")
    parser.add_argument("--small",
                        action="store_true",
                        help="If true, a small subset of argodata is used.")
    return parser.parse_args()


def load_seq_save_features(
        start_idx: int,
        sequences: List[str],
        save_dir: str,
        map_features_utils_instance: MapFeaturesUtils,
        social_features_utils_instance: SocialFeaturesUtils,
) -> None:
    """Load sequences, compute features, and save them.

    Args:
        start_idx : Starting index of the current batch
        sequences : Sequence file names
        save_dir: Directory where features for the current batch are to be saved
        map_features_utils_instance: MapFeaturesUtils instance
        social_features_utils_instance: SocialFeaturesUtils instance
    """
    count = 0
    args = parse_arguments()
    data = []

    # Enumerate over the batch starting at start_idx
    for seq in sequences[start_idx:start_idx + args.batch_size]:

        if not seq.endswith(".csv"):
            continue

        file_path = f"{args.data_dir}/{seq}"
        seq_id = int(seq.split(".")[0])

        features, map_feature_helpers = compute_features(file_path, map_features_utils_instance,
                                                        social_features_utils_instance)

        list_oracle_centerlines = [{"ORACLE_CENTERLINE": oracle_centerlines["ORACLE_CENTERLINE"]} for oracle_centerlines
                                   in map_feature_helpers]
        list_candidate_centerlines = [{"CANDIDATE_CENTERLINES": candidate_centerlines["CANDIDATE_CENTERLINES"]} for
                                   candidate_centerlines in map_feature_helpers]
        list_oracle_candidate_nt_distances = [{"CANDIDATE_NT_DISTANCES": nt_distances["CANDIDATE_NT_DISTANCES"]} for
                                   nt_distances in map_feature_helpers]

        count += 1
        data.append([
            seq_id,
            features,
            list_candidate_centerlines,
            list_oracle_centerlines,
            list_oracle_candidate_nt_distances,
        ])

        print(
            f"{args.mode}:{count}/{args.batch_size} with start {start_idx} and end {start_idx + args.batch_size}"
        )

    data_df = pd.DataFrame(
        data,
        columns=[
            "SEQUENCE",
            "FEATURES",
            "CANDIDATE_CENTERLINES",
            "ORACLE_CENTERLINE",
            "CANDIDATE_NT_DISTANCES",
        ],
    )

    # Save the computed features for all the sequences in the batch as a single file
    os.makedirs(save_dir, exist_ok=True)
    data_df.to_pickle(
        f"{save_dir}/forecasting_features_{args.mode}_{start_idx}_{start_idx + args.batch_size}.pkl"
    )


def compute_features(
        seq_path: str,
        map_features_utils_instance: MapFeaturesUtils,
        social_features_utils_instance: SocialFeaturesUtils,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Compute social and map features for the sequence.
    Args:
        seq_path (str): file path for the sequence whose features are to be computed.
        map_features_utils_instance: MapFeaturesUtils instance.
        social_features_utils_instance: SocialFeaturesUtils instance.
    Returns:
        merged_features (numpy array): SEQ_LEN x NUM_FEATURES
        map_feature_helpers (dict): Dictionary containing helpers for map features
    """
    args = parse_arguments()
    df = pd.read_csv(seq_path, dtype={"TIMESTAMP": str})

    # Neue Spalte einfügen, damit TRACK_IDs eindeutige (einfachere) Nummern in der Adjazenz-Matrix bekommen
    unique_track_ids = np.unique(df['TRACK_ID'])
    df.loc[:, 'ADJACENCY_NUM'] = -1
    for idx, track_id in enumerate(unique_track_ids):
        df.loc[df['TRACK_ID'] == track_id, 'ADJACENCY_NUM'] = idx

    track_ids = np.unique(df["TRACK_ID"].values)

    reference_timestamps = np.unique(df["TIMESTAMP"].values)

    all_features = np.ndarray(shape=(0, 50, 11))
    all_map_feature_helpers = []


    for idx, track in enumerate(track_ids):

        # track = '00000000-0000-0000-0000-000000043117'
        print(f"Handle Track ID: {track} of Sequence: {seq_path}. {idx+1}/{len(track_ids)}")

        # agent_track = df[df["OBJECT_TYPE"] == "AGENT"].values
        current_track = df[df["TRACK_ID"] == track].values

        """ Interpoliere unvollständige Daten """
        if len(current_track) < len(reference_timestamps):
            filled_current_track = np.full( (len(reference_timestamps), 7), None)
            index_available_timestamp = []
            for timestamp in reference_timestamps:
                index = np.where(current_track[:, 0] == timestamp)[0]
                if index.size != 0:
                    index_available_timestamp.append(index.item(0))
            filled_current_track[index_available_timestamp] = current_track
            filled_current_track[:, 0] = reference_timestamps
            filled_current_track[:, 1] = filled_current_track[0, 1]
            filled_current_track[:, 2] = filled_current_track[0, 2]
            filled_current_track[:, 5] = filled_current_track[0, 5]
            filled_current_track[:, 6] = filled_current_track[0, 6]
            for row in filled_current_track:
                if row[3] is None:
                    value_x = None
                    value_y = None
                    idx = np.where(row[0] == reference_timestamps)[0].item(0)
                    look_up = idx
                    look_down = idx
                    while value_x is None:
                        if look_up > 0:
                            look_up -= 1
                        if look_down < len(reference_timestamps) - 1:
                            look_down += 1
                        value_x = filled_current_track[look_down, 3]
                        value_y = filled_current_track[look_down, 4]
                        if value_x is None:
                            value_x = filled_current_track[look_up, 3]
                            value_y = filled_current_track[look_up, 4]
                    filled_current_track[idx, 3] = value_x
                    filled_current_track[idx, 4] = value_y
            current_track = filled_current_track
        """ Interpolation zuende """

        # Social features are computed using only the observed trajectory
        # social_features, adjacenies, distances = social_features_utils_instance.compute_social_features(
        #     df, agent_track, args.obs_len, args.obs_len + args.pred_len,
        #     RAW_DATA_FORMAT)

        social_features = social_features_utils_instance.compute_social_features(
           df, current_track, args.obs_len, args.obs_len + args.pred_len,
           RAW_DATA_FORMAT)

        # agent_track will be used to compute n-t distances for future trajectory,
        # using centerlines obtained from observed trajectory
        '''
        Die map_features enthalten die tangential und normal-Werte. map_feature_helpers enthalten die Koordinaten der 
        Mittellinien, auf die sich die nt-Werte beziehen. Mit Hilfe der Stadt und den Koordinaten kann so die Strecke
        rekonstruiert werden. Die Dimension der map_feature_helpers können die von seq_len überschreiten, da sie die 
        Koordinaten von allen Mittellinien (einzeln) enthalten. Sie haben nichts mit Zeitschritten zu tun.
        '''
        map_features, map_feature_helpers = map_features_utils_instance.compute_map_features(
            current_track,
            args.obs_len,
            args.obs_len + args.pred_len,
            RAW_DATA_FORMAT,
            args.mode,
        )

        # Combine social and map features

        # If track is of OBS_LEN (i.e., if it's in test mode), use agent_track of full SEQ_LEN,
        # But keep (OBS_LEN+1) to (SEQ_LEN) indexes having None values
        if current_track.shape[0] == args.obs_len:
            agent_track_seq = np.full((args.obs_len + args.pred_len, current_track.shape[1]), None)
            agent_track_seq[:args.obs_len] = current_track
            merged_features = np.concatenate((agent_track_seq, social_features, map_features), axis=1)
            # merged_features = np.concatenate((agent_track_seq, map_features), axis=1)
        else:
            merged_features = np.concatenate((current_track, social_features, map_features), axis=1)
            # merged_features = np.concatenate((current_track, map_features), axis=1)

        merged_features = np.expand_dims(merged_features, axis=0)
        all_features = np.concatenate( (all_features, merged_features), axis=0)
        all_map_feature_helpers.append(map_feature_helpers)

    # return merged_features, adjacenies, distances, map_feature_helpers
    return all_features, all_map_feature_helpers


def merge_saved_features(batch_save_dir: str) -> None:
    """Merge features saved by parallel jobs.
    Args:
        batch_save_dir: Directory where features for all the batches are saved.
    """
    args = parse_arguments()
    feature_files = os.listdir(batch_save_dir)
    all_features = []
    for feature_file in feature_files:
        if not feature_file.endswith(".pkl") or args.mode not in feature_file:
            continue
        file_path = f"{batch_save_dir}/{feature_file}"
        df = pd.read_pickle(file_path)
        all_features.append(df)

        # Remove the batch file
        os.remove(file_path)

    all_features_df = pd.concat(all_features, ignore_index=True)

    # Save the features for all the sequences into a single file
    all_features_df.to_pickle(
        f"{args.feature_dir}/forecasting_features_{args.mode}.pkl")


if __name__ == "__main__":
    """Load sequences and save the computed features."""
    args = parse_arguments()

    start = time.time()

    map_features_utils_instance = MapFeaturesUtils()
    social_features_utils_instance = SocialFeaturesUtils()

    sequences = os.listdir(args.data_dir)
    temp_save_dir = tempfile.mkdtemp()

    num_sequences = _FEATURES_SMALL_SIZE if args.small else len(sequences)

    Parallel(n_jobs=2)(delayed(load_seq_save_features)(
        i,
        sequences,
        temp_save_dir,
        map_features_utils_instance,
        social_features_utils_instance,
    ) for i in range(0, num_sequences, args.batch_size))
    merge_saved_features(temp_save_dir)
    shutil.rmtree(temp_save_dir)

    print(
        f"Feature computation for {args.mode} set completed in {(time.time() - start) / 60.0} mins"
    )