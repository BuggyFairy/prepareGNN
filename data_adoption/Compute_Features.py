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

from data_adoption.Feature_Config import RAW_DATA_FORMAT, _FEATURES_SMALL_SIZE, FEATURE_FORMAT_MASTER_THESIS, \
    EXTENDED_RAW_DATA_FORMAT
from data_adoption.map.MapFeaturesUtils import MapFeaturesUtils
from data_adoption.social.SocialFeaturesUtils import SocialFeaturesUtils

FAIL = '\033[91m'
ENDC = '\033[0m'


def parse_arguments() -> Any:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cpu",
        default=-1,
        type=int,
        help="Directory where the sequences (csv files) are saved",
    )
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

        print(f"{FAIL}Will handle {start_idx}/{len(sequences)} Sequenes{ENDC}")

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


        # print(
        #     f"{args.mode}:{count}/{args.batch_size} with start {start_idx} and end {start_idx + args.batch_size}"
        # )

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

    all_track_ids = np.unique(df["TRACK_ID"].values)

    original_tracks = 0
    interpolated_tracks = 0
    discarded_tracks = 0

    filtered_relevant_tracks = np.ndarray((0, args.obs_len + args.pred_len, len(RAW_DATA_FORMAT)))

    reference_timestamps = np.unique(df["TIMESTAMP"])
    for track_id in all_track_ids:
        current_track = df[df["TRACK_ID"] == track_id].values
        if current_track.shape[0] == 50:
            current_track = np.expand_dims(current_track, axis=0)
            filtered_relevant_tracks = np.concatenate((filtered_relevant_tracks, current_track), axis=0)
            original_tracks += 1

        elif current_track.shape[0] >= 40 and current_track.shape[0] < 50:
            interpolated_track = interpolate_track(current_track, reference_timestamps)
            current_track = np.expand_dims(interpolated_track, axis=0)
            filtered_relevant_tracks = np.concatenate((filtered_relevant_tracks, current_track), axis=0)
            interpolated_tracks += 1
        else:
            discarded_tracks += 1

    filtered_df = pd.DataFrame(data=filtered_relevant_tracks.reshape(-1, len(RAW_DATA_FORMAT)), columns=RAW_DATA_FORMAT)
    # Neue Spalte einfügen, damit TRACK_IDs eindeutige (einfachere) Nummern in der Adjazenz-Matrix bekommen
    unique_track_ids = np.unique(filtered_df['TRACK_ID'])
    filtered_df.loc[:, 'ADJACENCY_NUM'] = -1
    for idx, track_id in enumerate(unique_track_ids):
        filtered_df.loc[filtered_df['TRACK_ID'] == track_id, 'ADJACENCY_NUM'] = idx

    social_tracks_obs, social_features = social_features_utils_instance.compute_social_features(
       filtered_df, args.obs_len, args.obs_len + args.pred_len,
       EXTENDED_RAW_DATA_FORMAT)

    # agent_track will be used to compute n-t distances for future trajectory,
    # using centerlines obtained from observed trajectory
    '''
    Die map_features enthalten die tangential und normal-Werte. map_feature_helpers enthalten die Koordinaten der 
    Mittellinien, auf die sich die nt-Werte beziehen. Mit Hilfe der Stadt und den Koordinaten kann so die Strecke
    rekonstruiert werden. Die Dimension der map_feature_helpers können die von seq_len überschreiten, da sie die 
    Koordinaten von allen Mittellinien (einzeln) enthalten. Sie haben nichts mit Zeitschritten zu tun.
    '''

    print(f"For sequence {seq_path} we kept {original_tracks} original, {interpolated_tracks} interpolated, and discarded {discarded_tracks} tracks."
          f" We kept {(original_tracks + interpolated_tracks)}/{len(all_track_ids)} tracks")

    final_filtered_track_ids = np.unique(filtered_df["TRACK_ID"])
    final_tracks = np.ndarray((0, args.obs_len + args.pred_len, len(FEATURE_FORMAT_MASTER_THESIS) - 2))
    counter = 1
    map_features_helpers_list = []
    for track_id in final_filtered_track_ids:
        print(f"Will handle track {track_id} of sequence {seq_path}. This is Track {counter}/{len(final_filtered_track_ids)}.")
        current_track = filtered_df[filtered_df["TRACK_ID"] == track_id].values

        map_features, map_feature_helpers = map_features_utils_instance.compute_map_features(
            current_track,
            args.obs_len,
            args.obs_len + args.pred_len,
            RAW_DATA_FORMAT,
            args.mode,
        )

        # auxiliary_track = np.full((args.obs_len + args.pred_len, current_track.shape[1] + map_features.shape[1]), None)
        auxiliary_track = np.concatenate((current_track, map_features), axis=1)
        auxiliary_track = np.expand_dims(auxiliary_track, axis=0)
        final_tracks = np.concatenate((final_tracks, auxiliary_track), axis=0)

        map_features_helpers_list.append(map_feature_helpers)
        counter += 1

    merged_features = np.concatenate((final_tracks, social_features), axis=2)
    return merged_features, map_features_helpers_list


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


def interpolate_track(track, reference_timestamps):
    """ Interpoliere unvollständige Daten """
    if len(track) < len(reference_timestamps):
        filled_current_track = np.full( (len(reference_timestamps), len(RAW_DATA_FORMAT)), None)
        index_available_timestamp = []
        for timestamp in reference_timestamps:
            index = np.where(track[:, 0] == timestamp)[0]
            if index.size != 0:
                index_available_timestamp.append(np.where(reference_timestamps == timestamp)[0].item(0))
        filled_current_track[index_available_timestamp] = track
        filled_current_track[:, 0] = reference_timestamps
        filled_current_track[:, 1] = track[0, 1]
        filled_current_track[:, 2] = track[0, 2]
        filled_current_track[:, 5] = track[0, 5]
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
        track = filled_current_track
    """ Interpolation zuende """
    return track


if __name__ == "__main__":
    """Load sequences and save the computed features."""
    args = parse_arguments()

    start = time.time()

    map_features_utils_instance = MapFeaturesUtils()
    social_features_utils_instance = SocialFeaturesUtils()

    sequences = os.listdir(args.data_dir)
    temp_save_dir = tempfile.mkdtemp()

    num_sequences = _FEATURES_SMALL_SIZE if args.small else len(sequences)

    Parallel(n_jobs=args.cpu)(delayed(load_seq_save_features)(
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