"""Extract hit r values from TrackML test events and save to HDF5."""

from pathlib import Path

import h5py
import numpy as np
import pandas as pd

TEST_DIR = Path("/share/rcif2/pduckett/data/prepped/test/")
OUTPUT_PATH = Path("/share/rcif2/mmangat/hit_r_test.h5")
HIT_VOLUME_IDS = [7, 8, 9]


def main():
    event_files = sorted(TEST_DIR.glob("event*-hits.parquet"))
    print(f"Found {len(event_files)} events")

    with h5py.File(OUTPUT_PATH, "w") as f:
        for hits_path in event_files:
            event_name = hits_path.stem.replace("-hits", "")
            event_id = int(event_name.removeprefix("event"))

            hits = pd.read_parquet(hits_path)
            hits = hits[hits["volume_id"].isin(HIT_VOLUME_IDS)]

            for coord in ["x", "y", "z"]:
                hits[coord] *= 0.01

            r = np.sqrt(hits["x"].values ** 2 + hits["y"].values ** 2).astype(np.float32)

            f.create_dataset(str(event_id), data=r, compression="lzf")

    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
