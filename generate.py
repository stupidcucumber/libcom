from libcom import ControlComModel
from libcom.utils.process_image import make_image_grid, draw_bbox_on_image
import cv2, argparse
from ast import literal_eval

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--background-image', type=str, required=True,
                        help='Path to the background image.')
    parser.add_argument('--foreground-image', type=str, required=True,
                        help='Path to the foreground image.')
    parser.add_argument('--bboxes', type=literal_eval, required=True,
                        help='Bounding boxes. They are represented as "[[80, 20, 30, 40], ...]". \
                            Bounding boxes are in the format of [x1, y1, x2, y2] -- [upper-left, lower-right] points.')
    
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_arguments()

    input_bg = args.background_image
    input_fg = args.foreground_image
    bboxes = args.bboxes
    print('Bboxes: ', bboxes)
    print('Background: ', input_bg)
    print('Foreground: ', input_fg)

    net = ControlComModel(device=0)
    computed = net(background_image=input_bg,
                   foreground_image=input_fg,
                   bbox=bboxes[0],
                   task=['blending', 'harmonization'])
    
    print(len(computed))
    print(computed[0])
    cv2.imwrite('test.jpg', computed[0])
    result = draw_bbox_on_image(computed[0], bboxes[0])
    cv2.imshow('Result', result)


    print('Done.')
