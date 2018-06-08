from itertools import chain
from multiprocessing import Pool, cpu_count
from shutil import rmtree

from skimage import transform, util, draw, io, img_as_ubyte, color
import numpy as np
from os import path, mkdir, makedirs
import pandas as pd
from tqdm import tqdm

PAIRWISE_DF_DIR = '../../data/alov300++/'
PAIRWISE_DF_NAME = 'pairwise_train'

FIXED_SIZE = 76
OBJ_PATCH_RATIO = 0.3
SAMPLE_FRAC = 1



dataset_name = '%s_frac%s_size%s_ratio%s_grayscale' % (PAIRWISE_DF_NAME, SAMPLE_FRAC, FIXED_SIZE, OBJ_PATCH_RATIO)
save_dir = path.join('../../data/alov300++', dataset_name)


def _rescale_on_bounding_box(f1_ann, f2_ann):
    v1, v3 = _bounding_box_map_to_coord(f1_ann[:, :, 1])

    box_size = np.max(np.abs(v1 - v3))

    scale = FIXED_SIZE * OBJ_PATCH_RATIO / box_size

    f1_ann_rescaled = transform.rescale(f1_ann, scale, order=3)
    f2_ann_resaled = transform.rescale(f2_ann, scale, order=3)

    return f1_ann_rescaled, f2_ann_resaled


def _bounding_box_map_to_coord(box):
    ann_coords = tuple(zip(*np.nonzero(box[:, :] >= 0.5)))
    return np.array((ann_coords[0], ann_coords[-1]))


def _center_on_bounding_box_and_crop(f1_ann, f2_ann):
    f1_ann_padded = util.pad(f1_ann, ((FIXED_SIZE, FIXED_SIZE), (FIXED_SIZE, FIXED_SIZE), (0, 0)), mode='constant')
    f2_ann_padded = util.pad(f2_ann, ((FIXED_SIZE, FIXED_SIZE), (FIXED_SIZE, FIXED_SIZE), (0, 0)), mode='constant')

    f1_box_v1, f1_box_v3 = _bounding_box_map_to_coord(f1_ann_padded[:, :, 1])
    f1_center = (f1_box_v1 + f1_box_v3) / 2

    f1_cropped_start = f1_center - (FIXED_SIZE / 2)
    f1_cropped_end = f1_center + (FIXED_SIZE / 2) - 1

    rr, cc = draw.rectangle(f1_cropped_start, f1_cropped_end)

    f1_cropped = f1_ann_padded[rr, cc, :]
    f1_cropped = np.fliplr(f1_cropped)
    f1_cropped = transform.rotate(f1_cropped, 90)

    f2_cropped = f2_ann_padded[rr, cc, :]
    f2_cropped = np.fliplr(f2_cropped)
    f2_cropped = transform.rotate(f2_cropped, 90)

    # make sure that the dimension is FIXED_SIZE x FIXED_SIZE
    assert (f1_cropped.shape[0:2] == (FIXED_SIZE, FIXED_SIZE)) and (f2_cropped.shape[0:2] == (FIXED_SIZE, FIXED_SIZE))

    return f1_cropped, f2_cropped


def _annotate_frame(frame, box):
    annotated = np.zeros(frame.shape + (2,))
    annotated[:, :, 0] = frame
    rr, cc = draw.rectangle(box[0], box[1])
    annotated[rr, cc, 1] = 1
    return annotated


def _parse(f1, f2, box1, box2):
    f1_ann = _annotate_frame(f1, box1 - 1)
    f2_ann = _annotate_frame(f2, box2 - 1)

    f1_ann_rescaled, f2_ann_rescaled = _rescale_on_bounding_box(f1_ann, f2_ann)
    f1_ann_centered, f2_ann_centered = _center_on_bounding_box_and_crop(f1_ann_rescaled, f2_ann_rescaled)

    f1_cropped = f1_ann_centered[:, :, 0]
    box1_cropped = _bounding_box_map_to_coord(f1_ann_centered[:, :, 1])

    f2_cropped = f2_ann_centered[:, :, 0]
    box2_cropped = _bounding_box_map_to_coord(f2_ann_centered[:, :, 1])

    return f1_cropped, f2_cropped, box1_cropped, box2_cropped



def _process_df_chunk(df):
    pairwise_standardized_df_rows = []

    for _, r in tqdm(df.iterrows()):
        try:
            frame_id, video_name, box1, box2 = _process_record(r)

        except:
            print('error processing %s' % r['f2_filepath'])
            continue

        pairwise_standardized_df_rows.append({'video_category': r['category'],
                                              'feature_dir': path.join(video_name, frame_id),
                                              'video_id': video_name,
                                              'frame_id': int(frame_id),
                                              'box1': box1,
                                              'box2': box2})

    return pairwise_standardized_df_rows

def _read_img_as_gray(p):
    img = io.imread(p, as_gray=True)
    return img


def _process_record(r):
    f1 = _read_img_as_gray(path.join('../../', r['f1_filepath']))
    f2 = _read_img_as_gray(path.join('../../', r['f2_filepath']))

    f1_ann = np.array(r['f1_ann'])
    f2_ann = np.array(r['f2_ann'])
    # preprocess frames and annotations
    f1, f2, box1, box2 = _parse(f1, f2, f1_ann, f2_ann)
    video_name = path.basename(path.dirname(r['f2_filepath']))
    frame_id = path.basename(r['f2_filepath']).split('.')[0]
    f1_pathname = path.join(save_dir, video_name, frame_id, 'prev_frame.png')
    f2_pathname = path.join(save_dir, video_name, frame_id, 'frame.png')
    # save frames
    makedirs(path.join(save_dir, video_name, frame_id))
    try:
        f1 = img_as_ubyte(f1)
        f2 = img_as_ubyte(f2)
        io.imsave(f1_pathname, f1)
        io.imsave(f2_pathname, f2)
    except:
        rmtree(path.join(save_dir, video_name, frame_id))
        raise

    return frame_id, video_name, box1, box2


if __name__ == '__main__':
    pairwise_df = pd.read_json(path.join(PAIRWISE_DF_DIR, '%s.json' % PAIRWISE_DF_NAME))

    if path.exists(save_dir):
        print('removing %s' % save_dir)
        rmtree(save_dir)

    mkdir(save_dir)

    pool = Pool(cpu_count() - 1)
    results = pool.map(_process_df_chunk,
                       np.array_split(pairwise_df.sample(frac=SAMPLE_FRAC), cpu_count() - 1))

    pairwise_std_df = pd.DataFrame(list(chain.from_iterable(results)))
    pairwise_std_df.to_json(path.join(save_dir, 'index.json'), orient='records')







