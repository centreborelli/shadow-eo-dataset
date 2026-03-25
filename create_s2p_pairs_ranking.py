import os
from itertools import combinations

import numpy as np
import rasterio

import config


def create_s2p_pairs_ranking(output_dir: str):
    """
    Opens all images metadata and ranks all possible pairs from best to worst for use in S2P based on heuristics.
    """
    os.makedirs(output_dir, exist_ok=True)
    for city in config.SAT_FILES.keys():
        nadir_angles = []
        for image_file in config.SAT_FILES[city]:
            with rasterio.open(image_file) as src:
                nadir_angles.append(float(src.tags()["NITF_CSEXRA_OBLIQUITY_ANGLE"]))

        nadir_angles = np.array(nadir_angles)
        # Generate all unique pairs of indices
        index_pairs = list(combinations(range(len(nadir_angles)), 2))

        # Compute pair rankings based on criteria
        def ranking_criteria(pair):
            i, j = pair
            angle1, angle2 = nadir_angles[i], nadir_angles[j]
            # 1st priority: closeness to nadir
            min_angle = min(angle1, angle2)
            # 2nd priority: closeness to 20 degrees between angles
            diff_from_20 = abs(abs(angle1 - angle2) - 20)
            return (min_angle, diff_from_20)

        # Sort pairs based on the defined criteria
        sorted_pairs = sorted(index_pairs, key=ranking_criteria)
        # Write sorted pairs to output
        output_file = f"{output_dir}/{city}_s2p_pairs_ranking.txt"
        with open(output_file, "w") as f:
            for pair in sorted_pairs:
                f.write(f"{pair[0]}, {pair[1]}\n")


if __name__ == "__main__":
    import fire

    import config

    fire.Fire(create_s2p_pairs_ranking)
