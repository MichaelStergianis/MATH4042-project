"""The program project file, collects all libraries and puts together the program logic"""
import argparse
import cv2
# from filters import
# from denoising import
from interpolation import interpolate, bilinear
from filters import gaussian, median, improved_median, convolve

def main():
    ap = argparse.ArgumentParser()

    # command line arguments
    ap.add_argument('img', type=str, help='The input image')
    ap.add_argument(
        '-i',
        '--interpolation',
        type=str,
        choices=['bilinear', 'None'],
        default='None',
        help='The interpolation scheme to use')
    ap.add_argument('--width', type=int, help='The output image width')
    ap.add_argument('--height', type=int, help='The output image height')
    ap.add_argument(
        '-k',
        '--kernel',
        type=str,
        choices=['gaussian', 'median', 'improved_median', 'None'],
        default='None',
        help='The kernel to convolve over the image'
    )
    ap.add_argument('--patch-width', type=int, default=5, help='The kernel patch width')
    ap.add_argument('-o', '--output', type=str, help='The output file name')
    ap.add_argument('-g', '--grayscale', action='store_true', help='Convert the image to grayscale when loading')

    args = ap.parse_args()

    # check to make sure something is in the pipeline
    if args.kernel == 'None' and args.interpolation == 'None':
        raise ValueError("Must provide arguments to do something")

    interpolations = {'bilinear': bilinear, 'None': None}
    args.interpolation = interpolations[args.interpolation]

    patch_width = args.patch_width

    kernels = {'gaussian': gaussian(patch_width),
               'median': median,
               'improved_median': improved_median,
               'None': None}
    args.kernel = kernels[args.kernel]

    if args.grayscale:
        args.img = cv2.imread(args.img, cv2.IMREAD_GRAYSCALE)
    else:
        args.img = cv2.imread(args.img)

    img = args.img


    # convolution
    if args.kernel is not None:
        img = convolve(img, args.kernel, patch_width)

    if args.interpolation is not None:
        if args.width is None or args.height is None:
            raise ValueError("For interpolation you must provide width and height arguments")
        img = interpolate(img, args.interpolation, [args.width, args.height])

    # do final io ops
    if args.output is not None:
        # write the image to disk
        print("Now outputting image")
        cv2.imwrite(args.output, img)
    else:
        # display the image
        print("Press q to quit")
        cv2.imshow('Output', img)
        while True:
            key = cv2.waitKey(0)
            if key == 113: # q
                break
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
