from typing import AnyStr, Tuple
import numpy as np
import cv2
import argparse
import os
import random

parse = argparse.ArgumentParser(description='Convert to cartesian and save')
parse.add_argument('in_path', type=str, help = 'the path to the input folder')
parse.add_argument('out_path', type=str, help = 'the path to the output folder')

args = parse.parse_args()

directory = args.in_path
save_path = args.out_path

def load_radar(example_path: AnyStr) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Decode a single Oxford Radar RobotCar Dataset radar example
    Args:
        example_path (AnyStr): Oxford Radar RobotCar Dataset Example png
    Returns:
        timestamps (np.ndarray): Timestamp for each azimuth in int64 (UNIX time)
        azimuths (np.ndarray): Rotation for each polar radar azimuth (radians)
        valid (np.ndarray) Mask of whether azimuth data is an original sensor reading or interpolated from adjacent
            azimuths
        fft_data (np.ndarray): Radar power readings along each azimuth
        radar_resolution (float): Resolution of the polar radar data (metres per pixel)
    """
    # Hard coded configuration to simplify parsing code
    radar_resolution = np.array([0.0432], np.float32)
    encoder_size = 5600

    raw_example_data = cv2.imread(example_path, cv2.IMREAD_GRAYSCALE)
    timestamps = raw_example_data[:, :8].copy().view(np.int64)
    azimuths = (raw_example_data[:, 8:10].copy().view(np.uint16) / float(encoder_size) * 2 * np.pi).astype(np.float32)
    valid = raw_example_data[:, 10:11] == 255
    fft_data = raw_example_data[:, 11:].astype(np.float32)[:, :, np.newaxis] / 255.

    return timestamps, azimuths, valid, fft_data, radar_resolution


def radar_polar_to_cartesian(azimuths: np.ndarray, fft_data: np.ndarray, radar_resolution: float,
                             cart_resolution: float, cart_pixel_width: int, interpolate_crossover=True) -> np.ndarray:
    if (cart_pixel_width % 2) == 0:
        cart_min_range = (cart_pixel_width / 2 - 0.5) * cart_resolution
    else:
        cart_min_range = cart_pixel_width // 2 * cart_resolution
    coords = np.linspace(-cart_min_range, cart_min_range, cart_pixel_width, dtype=np.float32)
    Y, X = np.meshgrid(coords, -coords)
    sample_range = np.sqrt(Y * Y + X * X)
    sample_angle = np.arctan2(Y, X)
    sample_angle += (sample_angle < 0).astype(np.float32) * 2. * np.pi

    # Interpolate Radar Data Coordinates
    azimuth_step = azimuths[1] - azimuths[0]
    sample_u = (sample_range - radar_resolution / 2) / radar_resolution
    sample_v = (sample_angle - azimuths[0]) / azimuth_step

    # We clip the sample points to the minimum sensor reading range so that we
    # do not have undefined results in the centre of the image. In practice
    # this region is simply undefined.
    sample_u[sample_u < 0] = 0

    if interpolate_crossover:
        fft_data = np.concatenate((fft_data[-1:], fft_data, fft_data[:1]), 0)
        sample_v = sample_v + 1

    polar_to_cart_warp = np.stack((sample_u, sample_v), -1)
    cart_img = np.expand_dims(cv2.remap(fft_data, polar_to_cart_warp, None, cv2.INTER_LINEAR), -1)
    return cart_img


radar_path = directory + '/radar'

timestamps_path = directory + 'radar.timestamps'

# Cartesian Visualsation Setup
# Resolution of the cartesian form of the radar scan in metres per pixel
cart_resolution = .25
# Cartesian visualisation size (used for both height and width)
cart_pixel_width = 501  # pixels
interpolate_crossover = True

title = "Radar Visualisation Example"

radar_timestamps = np.loadtxt(timestamps_path, delimiter=' ', usecols=[0], dtype=np.int64)

for radar_timestamp in radar_timestamps:
    filename = os.path.join(radar_path, str(radar_timestamp) + '.png')
    save_filename = os.path.join(save_path, str(radar_timestamp) + '.png')
    orig_save_filename = os.path.join(save_path, str(radar_timestamp)+"_orig" + '.png')

    if not os.path.isfile(filename):
        raise FileNotFoundError("Could not find radar example: {}".format(filename))

    timestamps, azimuths, valid, fft_data, radar_resolution = load_radar(filename)
    cart_img = radar_polar_to_cartesian(azimuths, fft_data, radar_resolution, cart_resolution, cart_pixel_width,
                                        interpolate_crossover)
    original_img = (cart_img*255).astype('uint8')
    rows,cols,dims = cart_img.shape

    rand_num = random.randrange(45,50,1)/100

    for i in range(rows):
        for j in range(cols):
            if (random.random()>0.75):
                cart_img[i,j] = [0]

    cart_img = cv2.resize(cart_img, (256,256), interpolation = cv2.INTER_AREA)
    orig_img = cv2.resize(original_img, (256,256), interpolation = cv2.INTER_AREA)
    img = (cart_img*255).astype('uint8')
    cv2.imwrite(orig_save_filename, orig_img)
    cv2.imwrite(save_filename, img)
