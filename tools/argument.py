import argparse

def parseArgse():
    parser = argparse.ArgumentParser()

    parser.add_argument('--videoPath', type = str, default = None)
    parser.add_argument('--target', type = str, default = 'vihecle', 
                help="please use - without whitespace to separate multiple targets")
    parser.add_argument('--outputSize', type = int, nargs = 2, default = [None, None])
    parser.add_argument('--resume', default='cp/SiamMask_DAVIS.pth', type = str,
                metavar='PATH', help='path to latest checkpoint (default: none)')
    parser.add_argument('--mode', type = str, default = 'O', help='N for no weight, O for one look, G for Gaussian')

    args = parser.parse_args()

    return args


# parser.add_argument('--imgSize', type = int, nargs = 2, default = [None, None])
# parser.add_argument('--frameNum', type = int, default = None)
# parser.add_argument('--videoPath', type = str, default = None)
# parser.add_argument('--videoFolder', type=str, default = None)
# parser.add_argument('--maskPath', type = str, default = None)
# parser.add_argument('--framesFolder', type = str, default = None)
# parser.add_argument('--masksFolder', type = str, default = None)
# parser.add_argument('--mixMask', default = False, action = 'store_true')
# parser.add_argument('--outputSize', type = int, nargs = 2, default = [None, None])
# parser.add_argument('--videoName', type = str, default = None)
# parser.add_argument('--outputFolder', type = str, default = None)
# parser.add_argument('--resume', default='cp/SiamMask_DAVIS.pth', type = str,
#                 metavar='PATH', help='path to latest checkpoint (default: none)')
# parser.add_argument('--retarget_interval', type = int, default = 1)
# parser.add_argument('--useSiamMask', default = False, action = 'store_true')
# parser.add_argument('--txtFolder', type = str, default = None)
# parser.add_argument('--outputFramePath', type = str, default = None)
# parser.add_argument('--outputMaskPath', type = str, default = None)
# parser.add_argument('--outputInpaintPath', type = str, default = None)
# parser.add_argument('--fileName', type = str, default = None)
# parser.add_argument('--mode', type = str, default = 'O', help='N for no weight, O for one look, G for Gaussian')
# parser.add_argument('--resize', default = False, action = 'store_true')
# parser.add_argument('--target', type = str, default = 'vihecle', help="please use - without whitespace to separate multiple targets")