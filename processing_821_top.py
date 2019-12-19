import math
import os
import time
from functools import reduce

import cv2 as cv
import numpy as np

import param_top
import utils


def hough_method(image_path):
    image = utils.cv_imread(image_path)

    w, h, c = image.shape

    circles = utils.get_hough_circle(image)
    circles = utils.get_top_valid_circles(circles[0])

    all_rate = []

    for k, v in param_top.cross_points.items():
        location_coord = get_location_coord(circles, k, w, h)

        if k == 'top':
            quad = utils.cross_points2quad(location_coord)
        else:
            from itertools import chain
            quad = [list(chain.from_iterable(location_coord))]

        for idx_quad, _quad in enumerate(quad):
            line = [[_quad[i], _quad[i + 1]] for i in range(0, len(_quad), 2)]
            line = np.array([line]).astype(np.int64)

            crop_res = utils.crop_rectangle(image, line)
            crop_gray = cv.cvtColor(crop_res, cv.COLOR_BGR2GRAY)
            crop_w, crop_h = crop_gray.shape
            crop_bin = cv.inRange(crop_gray, param_top.segment[k]['minVal'],
                                  param_top.segment[k]['maxVal'])

            rate = np.sum(crop_bin) / 255 / (crop_w * crop_h)
            rate = round(rate, 2)
            all_rate.append(rate)

    return all_rate


def get_location_coord(circles, point_type, w, h):
    '''
    用hough定位圆再根据3个圆心位置，angle，dist去找cross_points
    '''
    location_point = []

    test_circles_dist_angle_list = utils.get_3_circles_dist_angle(circles, w, h)
    dist_01, angle_01, dist_02, angle_02, dist_12, angle_12 = test_circles_dist_angle_list
    test_circles_dist_list = [dist_01, dist_02, dist_12]
    test_circles_angle_list = [angle_01, angle_02, angle_12]
    test_x, test_y = utils.circles_angle_dist_2_x_y(test_circles_angle_list, test_circles_dist_list)

    for idx_point in range(len(param_top.template_angle[point_type])):
        per_circle_point = []

        for idx_circle, circle in enumerate(circles):
            x, y, r = [int(round(c)) for c in circle]
            x1, y1 = utils.encode_coord(x, y, w, h)

            angle = param_top.template_angle[point_type][f'point_{idx_point}'][idx_circle]
            dist = param_top.template_dist[point_type][f'point_{idx_point}'][idx_circle]

            template_circles_angle_list = param_top.circles_angle_list
            template_circles_dist_list = param_top.circles_dist_list
            template_x, template_y = utils.circles_angle_dist_2_x_y(template_circles_angle_list,
                                                                    template_circles_dist_list)

            dx_list, dy_list = [], []
            for i in range(1, 3):
                dx_list.append(template_x[i] / test_x[i])
                dy_list.append(template_y[i] / test_y[i])

            dx = utils.mean(dx_list)
            dy = utils.mean(dy_list)

            # caution: sin & cos 输入为弧度制
            x_0 = int(round(math.cos(math.radians(angle)) * dist / dx + x1))
            y_0 = int(round(math.sin(math.radians(angle)) * dist / dy + y1))

            x_0, y_0 = utils.decode_coord(x_0, y_0, w, h)
            per_circle_point.append((x_0, y_0))

        dst_x = reduce(lambda accumulator, item: accumulator + item[0], per_circle_point, 0)
        dst_y = reduce(lambda accumulator, item: accumulator + item[1], per_circle_point, 0)
        length = len(per_circle_point)
        dst_x, dst_y = dst_x / length, dst_y / length
        location_point.append([dst_x, dst_y])

    return location_point


def run(image_path):
    rate = hough_method(image_path)
    result = [rate[i] >= param_top.threshold[i] for i in range(len(rate))]
    return result, rate


def vis_hough(test_path):

    for file in os.listdir(test_path):
        image_path = test_path + file
        image = utils.cv_imread(image_path)
        cv.namedWindow('vis', 0)
        cv.imshow('vis', image)
        cv.waitKey()

        start = time.time()
        result, rate = run(image_path)
        end = time.time()

        print(f"executive time: {end - start: .2f}s, result: {result} in {[f'{r:.2f}' for r in rate]}")


if __name__ == '__main__':
    top_path = 'D:/dataset/豹小秘/样本_1113/821_MB_1113/top_valid/'
    top_invert_path = 'D:/dataset/豹小秘/样本_1113/821_MB_1113/top_invert/'
    all_top = 'D:/dataset/821/all_top/'

    vis_hough(all_top)
