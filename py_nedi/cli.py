# coding=utf-8
import argparse
import cv2
import os

from py_nedi import (
    edi_predict,
    edi_predict_multichannel
)


def _edi_cli():
    """
    This function is the main entry point for the NEDI command-line interface.
    It parses the command-line arguments and then processes the files accordingly.
    """

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


def _cli_process_file(input_path: str, scale: float, m: int, output_path: str) -> None:
    """
    Process an image file using the NEDI algorithm.

    :param input_path: The path to the input image file.
    :param scale: The scale factor to use for the NEDI algorithm.
    :param m: The sampling window size for the NEDI algorithm.
    :param output_path: The path to the output image file.
    """
    # Read the image from the input path using OpenCV
    img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)

    # Check if the image has multiple channels
    if len(img.shape) == 2:
        # If the image is grayscale, use the raw edi_predict single-channel function
        img = edi_predict(img, m, scale)
    else:
        # If the image is multi-channel, use the edi_predict_multichannel function
        img = edi_predict_multichannel(img, m, scale)

    # Save the processed image to the output path
    cv2.imwrite(output_path, img, _cv2_saving_options)
