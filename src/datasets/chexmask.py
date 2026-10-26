import numpy as np
import pandas as pd


def decode_rle(rle, height, width):
    mask = np.zeros(height * width, dtype=np.uint8)

    if pd.isna(rle) or not str(rle).strip():
        return mask.reshape((height, width), order="F")

    values = np.fromstring(rle, sep=" ", dtype=np.int64)

    starts = values[0::2] - 1
    lengths = values[1::2]

    for start, length in zip(starts, lengths):
        mask[start:start + length] = 1

    return mask.reshape((height, width), order="C")

def decode_lung_mask(row):
    height = int(row["Height"])
    width = int(row["Width"])

    left_lung = decode_rle(row["Left Lung"], height, width)
    right_lung = decode_rle(row["Right Lung"], height, width)

    return np.logical_or(left_lung, right_lung).astype(np.uint8)