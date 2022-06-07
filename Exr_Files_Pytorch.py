import numpy as np
import Imath
import OpenEXR
import torch


def import_exr(file_path: str):
    # source: https://excamera.com/articles/26/doc/intro.html
    PIXEL_TYPE = Imath.PixelType(Imath.PixelType.FLOAT)
    exr_file = OpenEXR.InputFile(file_path)
    data_window = exr_file.header()['dataWindow']
    width = data_window.max.x - data_window.min.x + 1
    height = data_window.max.y - data_window.min.y + 1
    try:
        channel = exr_file.channel('R', PIXEL_TYPE)
    except TypeError:
        channel = exr_file.channel('Y', PIXEL_TYPE)

    depth = np.frombuffer(channel, dtype=np.float32)
    depth.shape = (height, width)
    return depth.copy()


def export_exr(map: np.ndarray, file_path: str):
    # source: https://excamera.com/articles/26/doc/intro.html
    PIXEL_TYPE = Imath.PixelType(Imath.PixelType.FLOAT)

    map_tmp = map.astype(np.float32)
    if len(map.shape) == 3:
        map_tmp = map_tmp[:, :, 1]

    pixels = map_tmp.tostring()
    header = OpenEXR.Header(map.shape[1], map.shape[0])
    channel = Imath.Channel(PIXEL_TYPE)
    header['channels'] = dict([(c, channel) for c in "Y"])
    exr = OpenEXR.OutputFile(file_path, header)
    exr.writePixels({'Y': pixels})
    exr.close()


input_dir = ''