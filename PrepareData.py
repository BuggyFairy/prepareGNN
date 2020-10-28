import argparse
import os
from typing import Any


def parse_arguments() -> Any:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sub_folder_dir",
        default="./resources/argodata/val/",
        type=str,
        help="Directory where the sequences (csv files) are saved",
    )
    parser.add_argument(
        "--data_dir",
        default="./resources/argodata/val/",
        type=str,
        help="Directory where the sequences (csv files) are saved",
    )
    parser.add_argument(
        "--folder_num",
        default=10,
        type=int,
        help="Number of subfolders. Files will be equally distributed",
    )
    return parser.parse_args()


def split_sequence_files(args):
    files = [file for file in os.listdir(args.data_dir)]
    files = sorted(files, key=lambda x: int(os.path.splitext(x)[0]))
    dir_size = int(len(files) / args.folder_num)

    moved_files = 0
    for i in range(args.folder_num):
        dir_name = f"data_{i}"
        os.makedirs(os.path.join(args.sub_folder_dir, dir_name), exist_ok=True)

        files_in_dir = 0
        while moved_files < len(files) and files_in_dir <= dir_size:
            os.rename(os.path.join(args.data_dir, files[moved_files]), os.path.join(args.sub_folder_dir, dir_name, files[moved_files]))
            files_in_dir += 1
            moved_files += 1

    print(files)

if __name__ == "__main__":
    args = parse_arguments()

    split_sequence_files(args)