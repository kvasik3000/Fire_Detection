import argparse
import os

from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True)

    parser.add_argument(
        "--labels_dir",
        type=str,
        help="Директория с аннотациями.",
    )

    args = parser.parse_args()

    total_cnt = len(os.listdir(args.labels_dir))
    background_cnt = 0

    for label in tqdm(os.listdir(args.labels_dir), leave=False):
        with open(os.path.join(args.labels_dir, label)) as file:
            data = file.read()

            if len(data) == 0:
                background_cnt += 1

    print(f"background: {100 * background_cnt / total_cnt:.2f}%")
