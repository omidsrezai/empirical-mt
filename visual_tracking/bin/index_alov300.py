import argparse
from os import listdir

import pandas as pd
from os.path import join, split, relpath

from skimage import io


def get_resol(p):
    f1 = io.imread(join(p, '00000001.jpg'))
    f_num = len(listdir(join(p)))
    return f1.shape[0:2] + (f_num, )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='index ALOV300++ dataset')
    parser.add_argument('--input-folderpath', action='store', type=str, default='../../input')
    parser.add_argument('--output-folderpath', action='store', type=str, default='../../data/alov300++')

    args = parser.parse_args()

    video_paths = []

    for cat_dir_path in listdir(join(args.input_folderpath, 'ALOV300++/imagedata++')):
        for video_path in listdir(join(args.input_folderpath, 'ALOV300++/imagedata++', cat_dir_path)):
            video_path = join('ALOV300++/imagedata++', cat_dir_path, video_path)
            video_paths.append(video_path)

    alov300_df = pd.DataFrame({'video_path': video_paths})
    alov300_df['category'] = alov300_df['video_path'].map(lambda p: split(p)[1].split('-')[1].split('_')[0].lower())

    alov300_df['ground_truth_path'] = alov300_df['video_path'].map(
        lambda p: p.replace('imagedata++', 'alov300++_rectangleAnnotation_full') + '.ann')

    print(alov300_df.head())

    alov300_df['size'] = alov300_df['video_path'].map(lambda p: join(args.input_folderpath, p)).map(get_resol)
    alov300_df['size_x'] = alov300_df['size'].map(lambda s: s[0])
    alov300_df['size_y'] = alov300_df['size'].map(lambda s: s[1])
    alov300_df['size_z'] = alov300_df['size'].map(lambda s: s[2])

    alov300_df = alov300_df.sort_values(by='video_path')

    alov300_df.to_csv(join(args.output_folderpath, 'alov300.csv'), index=False,
                      columns=['video_path', 'category', 'ground_truth_path', 'size_x', 'size_y', 'size_z'])



