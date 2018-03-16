"""The program project file, collects all libraries and puts together the program logic"""
import argparse
import cv2
# from filters import
# from denoising import
from interpolation import interpolate, bilinear

def main(args):
    out = interpolate(args.img, args.func, [args.width, args.height])
    print("Now outputting image")
    cv2.imwrite(args.output, out)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('img', type=str, help='The input image')
    ap.add_argument('width', type=int, help='The output image width')
    ap.add_argument('height', type=int, help='The output image height')
    ap.add_argument(
        '-f',
        '--func',
        type=str,
        choices=['bilinear'],
        default='bilinear',
        help='The interpolation scheme to use')
    ap.add_argument('-o', '--output', type=str, help='The output file name')
    ARGS = ap.parse_args()
    FUNCS = {'bilinear': bilinear}
    ARGS.func = FUNCS[ARGS.func]
    ARGS.img = cv2.imread(ARGS.img)
    main(ARGS)
