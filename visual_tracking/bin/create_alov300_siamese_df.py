import pandas as pd
from os import path

from tqdm import tqdm

PARSED_GROUND_TRUTH_DIR = '../../data/alov300++/ground_truth'

if __name__ == '__main__':
    alov300_df = pd.read_csv('../../data/alov300++/alov300.csv')
    alov300_siamese_cols = ['f1_filepath', 'f2_filepath', 'f1_ann', 'f2_ann', 'video_path', 'video_name', 'category']
    alov300_siamese_df = pd.DataFrame(columns=alov300_siamese_cols)

    for _, row in tqdm(alov300_df.iterrows()):
        video_path = row['video_path']
        parsed_ground_truth_path = path.join(PARSED_GROUND_TRUTH_DIR, path.basename(row['ground_truth_path']))

        ground_truth_df = pd.read_json(parsed_ground_truth_path).sort_index()

        siamese_rows = []

        for i in range(0, len(ground_truth_df) - 1):
            f1_ground_truth = ground_truth_df.iloc[i]
            f2_ground_truth = ground_truth_df.iloc[i + 1]

            f1_filepath = path.join(video_path, '%08d.jpg' % f1_ground_truth['frame_id'])
            f2_filepath = path.join(video_path, '%08d.jpg' % f2_ground_truth['frame_id'])

            f1_ann = (f1_ground_truth['v1'], f1_ground_truth['v3'])
            f2_ann = (f2_ground_truth['v1'], f2_ground_truth['v3'])

            siamese_rows.append({'f1_filepath': f1_filepath,
                                 'f1_ann': f1_ann,
                                 'f2_filepath': f2_filepath,
                                 'f2_ann': f2_ann,
                                 'video_path': video_path,
                                 'video_name': path.basename(video_path),
                                 'category': row['category']})


        alov300_siamese_df = alov300_siamese_df.append(pd.DataFrame(siamese_rows))


    # alov300_siamese_df.reset_index(inplace=True)
    alov300_siamese_df.to_json('../../data/alov300++/siamese.json', orient='records')





