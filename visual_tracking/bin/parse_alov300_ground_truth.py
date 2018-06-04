import pandas as pd
from os import path
import numpy as np
from tqdm import tqdm

PARSED_GROUND_TRUTH_DIR = '../../data/alov300++/ground_truth/'


def parse_ground_truth_line(l):
    vs = map(float, l['line'].split(' '))
    frame_id = int(vs[0])

    v1 = (vs[1], vs[2])
    v1 = np.flip(np.round(v1).astype(np.int), axis=0)

    v2 = (vs[3], vs[4])
    v2 = np.flip(np.round(v2).astype(np.int), axis=0)

    v3 = (vs[5], vs[6])
    v3 = np.flip(np.round(v3).astype(np.int), axis=0)

    v4 = (vs[7], vs[8])
    v4 = np.flip(np.round(v4).astype(np.int), axis=0)

    return pd.Series({'frame_id': frame_id, 'v1': v1, 'v2': v2, 'v3': v3, 'v4': v4})


if __name__ == '__main__':
    alov300_df = pd.read_csv('../../data/alov300++/alov300.csv')

    for ground_truth_path in tqdm(alov300_df['ground_truth_path']):
        ground_truth_path_rel = path.join('../../', ground_truth_path)
        ground_truth = pd.read_csv(ground_truth_path_rel, header=None, names=['line'])

        ground_truth_parsed = ground_truth.apply(parse_ground_truth_line, axis=1)
        ground_truth_parsed.set_index(keys='frame_id', inplace=True, drop=False)

        ground_truth_parsed_path = path.join(PARSED_GROUND_TRUTH_DIR, path.basename(ground_truth_path))

        ground_truth_parsed.to_json(ground_truth_parsed_path)


