# coding=utf-8
import cv2
import math
import numpy as np
import os

from typing import Callable

"""
Author:

    hu.leying@columbia.edu

Maintainer:

    Mikołaj Pokora, pokora.mikolaj@gmail.com

Usage:

    edi_predict(img, m, s)

    # Universal method that assures an output image, no matter the value of scale (s)
    # If s < 2 linear interpolation is used, if s > 2 NEDI is used

    # img is the input single channel image (e.g. grayscale)
    # m is the sampling window size, not scaling factor! The larger the m, more blurry the image. Ideal m >= 4
    # m should be the multiple of 2. If m is odd, it will be incremented by 1
    # s is the scaling factor, support any s > 0 (e.g. use s=2 to upscale by 2, use s=0.5 to downscale by 2)


    edi_predict_multichannel(img, m, s)

    # Universal method that assures an output image, no matter the value of scale (s)
    # If s < 2 linear interpolation is used, if s > 2 NEDI is used

    # img is the input multi-channel image
    # m is the sampling window size, not scaling factor! The larger the m, more blurry the image. Ideal m >= 4
    # m should be the multiple of 2. If m is odd, it will be incremented by 1
    # s is the scaling factor, support any s > 0 (e.g. use s=2 to upscale by 2, use s=0.5 to downscale by 2)


    edi_upscale(img, m)

    # Actual NEDI upscaling function. Scales the image by the factor of 2

    # img is the input single channel image (e.g. grayscale)
    # m is the sampling window size, not scaling factor! The larger the m, more blurry the image. Ideal m >= 4
    # m should be the multiple of 2. If m is odd, it will be incremented by 1

"""


def _multichannel_multiprocess(
        function: Callable[[np.ndarray, ...], np.ndarray],
        img: np.ndarray,
        s: float,
        multiprocess: bool = True,
        mode_override: str | None = "spawn",
        *args
) -> np.ndarray:
    import multiprocessing

    w, h, channels = img.shape

    if multiprocess:
        # Use multiprocessing with specified mode
        if mode_override is None:
            mode_override = multiprocessing.get_start_method()
        ctx = multiprocessing.get_context(mode_override)

        # Run function in parallel
        with ctx.Pool(processes=channels) as pool:
            mp_args = [(img[:, :, i], *args) for i in range(channels)]
            print(f"{len(args)=}; {len(mp_args[0])=}")
            results = pool.map(function, mp_args)

        # Parse results
        new_img = np.stack(results, axis=2)

    else:
        # No multiprocessing
        new_img = np.stack([function(img[:, :, i], *args) for i in range(channels)], axis=2)

    return new_img


def edi_upscale(img: np.ndarray, m: int) -> np.ndarray:
    """
        Actual NEDI upscaling function. Scales the image by the factor of 2

        :param img: Single channel image
        :param m: Sampling window size. The larger the m, more blurry the image. Ideal m >= 4.
            m should be the multiple of 2. If m is odd, it will be incremented by 1
        :return: Upscaled single channel image
    """
    # m should be equal to a multiple of 2
    # assert (m % 2 == 0)
    if m % 2 == 1:
        m += 1

    # initializing image to be predicted
    w, h = img.shape
    imgo = np.zeros((w * 2, h * 2))

    # Place low-resolution pixels
    for i in range(w):
        for j in range(h):
            imgo[2 * i][2 * j] = img[i][j]

    y = np.zeros((m ** 2, 1))  # pixels in the window
    c = np.zeros((m ** 2, 4))  # interpolation neighbours of each pixel in the window

    # Reconstruct the points with the form of (2*i+1,2*j+1)
    for i in range(math.floor(m / 2), w - math.floor(m / 2)):
        for j in range(math.floor(m / 2), h - math.floor(m / 2)):
            tmp = 0
            for ii in range(i - math.floor(m / 2), i + math.floor(m / 2)):
                for jj in range(j - math.floor(m / 2), j + math.floor(m / 2)):
                    y[tmp][0] = imgo[2 * ii][2 * jj]
                    c[tmp][0] = imgo[2 * ii - 2][2 * jj - 2]
                    c[tmp][1] = imgo[2 * ii + 2][2 * jj - 2]
                    c[tmp][2] = imgo[2 * ii + 2][2 * jj + 2]
                    c[tmp][3] = imgo[2 * ii - 2][2 * jj + 2]
                    tmp += 1

            # calculating weights
            # a = (c^T * c)^(-1) * (c^T * y) = (c^T * c) \ (c^T * y)
            a = np.matmul(np.matmul(np.linalg.pinv(np.matmul(np.transpose(c), c)), np.transpose(c)), y)
            imgo[2 * i + 1][2 * j + 1] = np.matmul(
                [imgo[2 * i][2 * j], imgo[2 * i + 2][2 * j], imgo[2 * i + 2][2 * j + 2], imgo[2 * i][2 * j + 2]], a)

    # Reconstructed the points with the forms of (2*i+1,2*j) and (2*i,2*j+1)
    for i in range(math.floor(m / 2), w - math.floor(m / 2)):
        for j in range(math.floor(m / 2), h - math.floor(m / 2)):
            tmp = 0
            for ii in range(i - math.floor(m / 2), i + math.floor(m / 2)):
                for jj in range(j - math.floor(m / 2), j + math.floor(m / 2)):
                    y[tmp][0] = imgo[2 * ii + 1][2 * jj - 1]
                    c[tmp][0] = imgo[2 * ii - 1][2 * jj - 1]
                    c[tmp][1] = imgo[2 * ii + 1][2 * jj - 3]
                    c[tmp][2] = imgo[2 * ii + 3][2 * jj - 1]
                    c[tmp][3] = imgo[2 * ii + 1][2 * jj + 1]
                    tmp += 1

            # calculating weights
            # a = (c^T * c)^(-1) * (c^T * y) = (c^T * c) \ (c^T * y)
            a = np.matmul(np.matmul(np.linalg.pinv(np.matmul(np.transpose(c), c)), np.transpose(c)), y)
            imgo[2 * i + 1][2 * j] = np.matmul(
                [imgo[2 * i][2 * j], imgo[2 * i + 1][2 * j - 1], imgo[2 * i + 2][2 * j], imgo[2 * i + 1][2 * j + 1]], a)
            imgo[2 * i][2 * j + 1] = np.matmul(
                [imgo[2 * i - 1][2 * j + 1], imgo[2 * i][2 * j], imgo[2 * i + 1][2 * j + 1], imgo[2 * i][2 * j + 2]], a)

    # Fill the rest with bilinear interpolation
    np.clip(imgo, 0, 2 ** (8 * img.itemsize), out=imgo)
    imgo_bilinear = cv2.resize(img, dsize=(h * 2, w * 2), interpolation=cv2.INTER_LINEAR)
    imgo[imgo == 0] = imgo_bilinear[imgo == 0]

    return imgo.astype(img.dtype)


def _edi_upscale_channel(*args) -> np.ndarray:
    img, m = args[0]
    return edi_upscale(img, m)


def edi_upscale_multichannel(
        img: np.ndarray,
        m: int,
        multiprocess: bool = True,
        mode_override: str | None = None
) -> np.ndarray:
    return _multichannel_multiprocess(_edi_upscale_channel, img, 2, multiprocess, mode_override, m)


def edi_predict(img: np.ndarray, m: int, s: float) -> np.ndarray:
    try:
        w, h = img.shape
    except ValueError as e:
        print(e)
        raise ValueError("Error: Invalid input; Please input a valid single channel image!")

    output_type = img.dtype

    if s <= 0:
        # sys.exit("Error input: Please input s > 0!")
        raise ValueError("Error input: Please input s > 0!")

    elif s == 1:
        print("No need to rescale since s = 1")
        return img

    elif s < 2:
        # Linear Interpolation is enough for upscaling not over 2
        return cv2.resize(img, dsize=(int(h * s), int(w * s)), interpolation=cv2.INTER_LINEAR).astype(output_type)

    else:
        # Calculate how many times to do the EDI upscaling
        n = math.floor(math.log(s, 2))
        for i in range(n):
            img = edi_upscale(img, m)

        # Upscale to the expected size with linear interpolation
        linear_factor = s / math.pow(2, n)
        if linear_factor == 1:
            return img.astype(output_type)

        # Update new shape
        w, h = img.shape
        return cv2.resize(
            img,
            dsize=(round(h * linear_factor), round(w * linear_factor)),
            interpolation=cv2.INTER_LINEAR
        ).astype(output_type)


def _edi_predict_channel(*args) -> np.ndarray:
    img_channel, m, s = args[0]
    return edi_predict(img_channel, m, s)


def edi_predict_multichannel(
        img: np.ndarray,
        m: int, s: float,
        multiprocess: bool = True,
        mode_override: str | None = None
) -> np.ndarray:
    return _multichannel_multiprocess(_edi_predict_channel, img, s, multiprocess, mode_override, m, s)


def _edi_cli():
    import argparse

    # Create an argument (flag) parser
    parser = argparse.ArgumentParser(
        prog="Python NEDI",
        description="Python NEDI implementation",
        epilog="Please consider acknowledging this small project for research use. Thank you!",
    )

    # Add arguments
    parser.add_argument('-s', "--scale", type=float, default=2)
    parser.add_argument('-m', "--m", type=int, default=4)
    parser.add_argument('-i', "--input", type=str, default=None, help="Path to the input file or directory")
    parser.add_argument('-o', "--output", type=str, default=None, help="Path to the output file or directory")
    parser.add_argument('-r', "--recursive", action="store_true", default=False)

    # Parse arguments
    args = parser.parse_args()

    scale: float = args.scale
    m: int = args.m

    input_path: str | None = args.input
    if input_path is None:
        # Set the input directory to be the current working directory
        input_path = os.getcwd()

    output_path: str | None = args.output
    if output_path is None:
        # Check if the input path is a directory
        if os.path.isdir(input_path):
            output_path = os.path.join(input_path, "output")
            os.makedirs(output_path, exist_ok=True)
        else:
            output_path = f"scaled_{input_path}"

    recursive: bool = args.recursive

    print("Loaded all arguments/flags")

    # Process files
    # Check if the input path is a directory
    if os.path.isdir(input_path):
        cv2_supported_extensions = {"png", "jpg", "jpeg", "tiff", "tif"}

        if not recursive:
            # Check all files in the input directory
            for filename in os.listdir(os.getcwd()):
                # Check if the file has supported file extension
                extension = filename.split('.')[-1]
                if extension in cv2_supported_extensions:
                    input_file = os.path.join(input_path, filename)

                    if os.path.isdir(output_path):
                        output_file = os.path.join(output_path, f"scaled_{filename}")
                    else:
                        output_file = os.path.join(input_path, f"scaled_{filename}", output_path)

                    print("Found an image to process; Processing...")
                    _cli_process_file(input_file, scale, m, output_file)

        else:
            for root, _, files in os.walk(input_path):
                for filename in files:
                    # Check if the file has supported file extension
                    extension = filename.split('.')[-1]
                    if extension in cv2_supported_extensions:
                        input_file = os.path.join(input_path, filename)

                        if os.path.isdir(output_path):
                            output_file = os.path.join(root, output_path, f"scaled_{filename}")
                        else:
                            output_file = os.path.join(root, input_path, f"scaled_{filename}", output_path)

                        print("Found an image to process; Processing...")
                        _cli_process_file(input_file, scale, m, output_file)
    else:
        # The path is a file
        print("Found an image to process; Processing...")
        _cli_process_file(input_path, scale, m, output_path)

    print("Finished")


# Initialize universal CV2 saving options
_cv2_saving_options: list[int] = [
    cv2.IMWRITE_PNG_COMPRESSION, 9,  # Highest
    cv2.IMWRITE_TIFF_COMPRESSION, 32946,  # Deflate
    cv2.IMWRITE_JPEG_QUALITY, 95,
    cv2.IMWRITE_WEBP_QUALITY, 100  # Lossless
]


def _cli_process_file(input_path: str, scale: float, m: int, output_path: str):
    img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)

    # Check if the image has multiple channels
    if len(img.shape) == 2:
        img = edi_predict(img, m, scale)
    else:
        img = edi_predict_multichannel(img, m, scale)

    cv2.imwrite(output_path, img, _cv2_saving_options)


# Internal testing
if __name__ == "__main__":
    img = cv2.imread("images/math_psnr.png", cv2.IMREAD_UNCHANGED)

    try:
        edi_upscale(img, 4)
    except ValueError as e:
        assert str(e) == "too many values to unpack (expected 2)"
    else:
        raise Exception("Test failed")

    try:
        edi_predict(img, 4, 2)
    except ValueError as e:
        assert str(e) == "Error: Invalid input; Please input a valid single channel image!"
    else:
        raise Exception("Test failed")
