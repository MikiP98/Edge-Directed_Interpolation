# coding=utf-8
import cv2
import math
import numpy as np

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
        multiprocess: bool = True,
        mode_override: str | None = "spawn",
        *args
) -> np.ndarray:
    """
    This function helps to process individual channels of a multi-channel image in parallel.

    :param function: The function to be processed in parallel. Should take a single channel image and any additional arguments
    :param img: The input multi-channel image
    :param multiprocess: Whether to use multiprocessing or not. Defaults to True
    :param mode_override: The multiprocessing mode to use. Defaults to None, which uses the mode set by multiprocessing.set_start_method()
    :param args: Any additional arguments to pass to the function
    :return: The processed image
    """

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
        Implementation of New Edge-Directed Interpolation (NEDI) for upscaling an image by a factor of 2.
        NEDI is an edge-directed technique that is based on the idea of interpolating the missing high-frequency components.

        :param img: Single channel image
        :param m: Sampling window size. The larger the `m`, the more blurry the image, but better the edges. Ideal m >= 4. `m` should be the multiple of 2. If m is odd, it will be incremented by 1
        :return: Upscaled single channel image
    """
    # `m` should be equal to a multiple of 2
    # assert (m % 2 == 0)
    m += m % 2  # Increment `m` by 1 if it is odd
    # m &= ~1  # More optimized, but decrements `m` by 1 if it is odd instead of incrementing by 1

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
    return _multichannel_multiprocess(_edi_upscale_channel, img, multiprocess, mode_override, m)


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
    return _multichannel_multiprocess(_edi_predict_channel, img, multiprocess, mode_override, m, s)
