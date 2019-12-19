import math

import cv2 as cv
import numpy as np


def cv_imread(img_path):
    image = cv.imdecode(np.fromfile(img_path, dtype=np.int8), -1)
    assert image is not None, f'can`t found image in {img_path}'
    return image


def cross_point(line1, line2):  # 计算交点函数
    x1 = line1[0]  # 取四点坐标
    y1 = line1[1]
    x2 = line1[2]
    y2 = line1[3]

    x3 = line2[0]
    y3 = line2[1]
    x4 = line2[2]
    y4 = line2[3]

    k1 = (y2 - y1) * 1.0 / (x2 - x1)  # 计算k1,由于点均为整数，需要进行浮点数转化
    b1 = y1 * 1.0 - x1 * k1 * 1.0  # 整型转浮点型是关键
    if (x4 - x3) == 0:  # L2直线斜率不存在操作
        k2 = None
        b2 = 0
    else:
        k2 = (y4 - y3) * 1.0 / (x4 - x3)  # 斜率存在操作
        b2 = y3 * 1.0 - x3 * k2 * 1.0
    if k2 == None:
        x = x3
    else:
        x = (b2 - b1) * 1.0 / (k1 - k2)
    y = k1 * x * 1.0 + b1 * 1.0
    return [x, y]


def cross_point_remake(line1, line2):  # 计算交点函数
    x1 = line1[1]  # 取四点坐标
    y1 = line1[0]
    x2 = line1[3]
    y2 = line1[2]

    x3 = line2[1]
    y3 = line2[0]
    x4 = line2[3]
    y4 = line2[2]

    k1 = (y2 - y1) * 1.0 / (x2 - x1)  # 计算k1,由于点均为整数，需要进行浮点数转化
    b1 = y1 * 1.0 - x1 * k1 * 1.0  # 整型转浮点型是关键
    if (x4 - x3) == 0:  # L2直线斜率不存在操作
        k2 = None
        b2 = 0
    else:
        k2 = (y4 - y3) * 1.0 / (x4 - x3)  # 斜率存在操作
        b2 = y3 * 1.0 - x3 * k2 * 1.0
    if k2 == None:
        x = x3
    else:
        x = (b2 - b1) * 1.0 / (k1 - k2)
    y = k1 * x * 1.0 + b1 * 1.0
    return (x, y)


def get_angle(x1, y1, x2, y2):
    dy = y2 - y1
    dx = x2 - x1
    angle = math.degrees(math.atan2(dy, dx))
    if dy < 0:
        angle += 360
    return angle


def get_dist(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def get_circles_dist(circles):
    # dist_01 mean: dist between circle_0 and circle_1
    if len(circles) == 2:
        dist = get_dist(circles[0][0], circles[0][1], circles[1][0], circles[1][1])
        return [dist]
    elif len(circles) == 3:
        dist_01 = get_dist(circles[0][0], circles[0][1], circles[1][0], circles[1][1])
        dist_12 = get_dist(circles[2][0], circles[2][1], circles[1][0], circles[1][1])
        dist_20 = get_dist(circles[0][0], circles[0][1], circles[2][0], circles[2][1])
        # return [dist_01, dist_12, dist_20
        return [dist_01]
    else:
        return None


def get_3_circles_dist_angle(circles, w, h):
    # 计算三个定位圆之间的距离和角度
    # return: 注意顺序
    assert len(circles) == 3, f'the length is not equal 3 in \n{circles}'

    c0, c1, c2 = [(encode_coord(c[0], c[1], w, h)) for c in circles]

    dist_01 = get_dist(c0[0], c0[1], c1[0], c1[1])
    angle_01 = get_angle(c0[0], c0[1], c1[0], c1[1])

    dist_02 = get_dist(c0[0], c0[1], c2[0], c2[1])
    angle_02 = get_angle(c0[0], c0[1], c2[0], c2[1])

    dist_12 = get_dist(c1[0], c1[1], c2[0], c2[1])
    angle_12 = get_angle(c1[0], c1[1], c2[0], c2[1])

    return dist_01, angle_01, dist_02, angle_02, dist_12, angle_12


def get_hough_circle(image):
    image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    circles = cv.HoughCircles(image_gray, cv.HOUGH_GRADIENT, 1, 50, param1=80, param2=30, minRadius=15, maxRadius=19)
    return circles


def get_cross_points(line_dict):
    '''
    line2cross_point
    :param line_dict: dict = {'index': [line1, line2]}
    :return: cross_points -> list # len(list) = len(dict)
    '''
    cross_points = []
    for k, v in line_dict.items():
        line1, line2 = v
        line1 = [round(_) for _ in line1]
        line2 = [round(_) for _ in line2]

        point = cross_point_remake(line1, line2)
        cross_points.append(point)

    return cross_points


def get_top_angle_dist(cross_points):
    '''
    使用get_cross_points的ret作为输入，得到每个点对三个hough圆的相对位置关系（angle，dist）
    :param cross_points:
    :return:
    '''
    image = cv_imread('img_file/top/original_template.png')
    assert image is not None, 'not found image'

    roi = [865.248, 1607.33, 2037.67, 2747.62]  # x1, y1, x2, y2
    roi = [round(i) for i in roi]
    image_roi = image[roi[0]:roi[2], roi[1]:roi[3], :]
    image = image_roi
    w, h, c = image.shape

    circles = get_hough_circle(image)  # ROI的circles

    template_angle = {}
    template_dist = {}

    for point_index, points in enumerate(cross_points):

        template_angle[f'point_{point_index}'] = []
        template_dist[f'point_{point_index}'] = []

        for circle_index, circle in enumerate(circles[0]):
            x = round(circle[0])
            y = round(circle[1])

            x1, y1 = encode_coord(x, y, w, h)
            x2, y2 = encode_coord(points[0], points[1], w, h)

            angle = get_angle(x1, y1, x2, y2)
            dist = get_dist(x1, y1, x2, y2)

            template_angle[f'point_{point_index}'].append(angle)
            template_dist[f'point_{point_index}'].append(dist)

    return template_angle, template_dist


def get_bottom_angle_dist(cross_points):
    '''
    使用get_cross_points的ret作为输入，得到每个点对三个hough圆的相对位置关系（angle，dist）
    混合了circles之间的angle和dist
    '''
    image = cv_imread('img_file/bottom/template_original.png')
    assert image is not None, 'not found image'

    w, h, c = image.shape

    circles = get_hough_circle(image)  # ROI的circles

    circles = get_bottom_valid_circles(circles[0])

    dist_01, angle_01, dist_02, angle_02, dist_12, angle_12 = get_3_circles_dist_angle(circles, w, h)

    template_circles_dist_list = [dist_01, dist_02, dist_12]
    template_circles_angle_list = [angle_01, angle_02, angle_12]

    template_angle = {}
    template_dist = {}

    for point_index, points in enumerate(cross_points):

        template_angle[f'point_{point_index}'] = []
        template_dist[f'point_{point_index}'] = []

        for circle_index, circle in enumerate(circles):
            x = round(circle[0])
            y = round(circle[1])

            x1, y1 = encode_coord(x, y, w, h)
            x2, y2 = encode_coord(points[0], points[1], w, h)

            angle = get_angle(x1, y1, x2, y2)
            dist = get_dist(x1, y1, x2, y2)

            template_angle[f'point_{point_index}'].append(angle)
            template_dist[f'point_{point_index}'].append(dist)

    return template_angle, template_dist, template_circles_dist_list, template_circles_angle_list


def circles_angle_dist_2_x_y(circles_angle_list, circles_dist_list):
    # 根据3个定位圆的angle和dist数据计算其x轴和y轴上的距离 get_3_circles_dist_angle()

    x_list, y_list = [], []
    for i in range(len(circles_angle_list)):
        x = math.cos(math.radians(circles_angle_list[i])) * circles_dist_list[i]
        y = math.sin(math.radians(circles_angle_list[i])) * circles_dist_list[i]

        x_list.append(abs(x))
        y_list.append(abs(y))

    return x_list, y_list


def halcon2cv_top(line_dict):
    # top 板的案例
    # 从halcon得到line以及手工找出点作为输入
    # 再得到交点
    # 再获取交点与定位圆的angle和dist
    cross_point = get_cross_points(line_dict)  # line_dict 是基于ROI的
    angle, dist = get_top_angle_dist(cross_point)  # angle 和 dist 是 ROI无关的
    print(cross_point)
    print(angle)
    print(dist)


def encode_coord(x, y, w, h):
    # 为方便计算角度，将图片坐标系转为真实坐标系，即图片左下角为坐标原点，而非左上角
    return x, h - y


def decode_coord(x, y, w, h):
    # same func content, different meaning
    return x, h - y


def mean(_list):
    return sum(_list) / len(_list)


def crop_rectangle(img, geo):
    rect = cv.minAreaRect(geo.astype(int))
    center, size, angle = rect[0], rect[1], rect[2]
    if (angle > -45):
        center = tuple(map(int, center))
        size = tuple([int(rect[1][0] + 10), int(rect[1][1] + 10)])
        height, width = img.shape[0], img.shape[1]
        M = cv.getRotationMatrix2D(center, angle, 1)
        img_rot = cv.warpAffine(img, M, (width, height))
        img_crop = cv.getRectSubPix(img_rot, size, center)
    else:
        center = tuple(map(int, center))
        size = tuple([int(rect[1][1] + 10), int(rect[1][0]) + 10])
        angle -= 270
        height, width = img.shape[0], img.shape[1]
        M = cv.getRotationMatrix2D(center, angle, 1)
        img_rot = cv.warpAffine(img, M, (width, height))
        img_crop = cv.getRectSubPix(img_rot, size, center)
    return img_crop


def get_top_valid_circles(circles):
    circles = circles.tolist()
    circles_x = sorted(circles, key=lambda item: item[0])
    left_2 = circles_x[:2]
    ys = np.array(left_2)[:, 1]
    assert len(ys) == 2, f'{ys} length {left_2}'
    y_min, y_max = sorted(ys)
    right_circle = [circle for circle in circles if y_min < circle[1] < y_max]
    assert len(right_circle) != 0, 'not found right circle'
    left_2 = sorted(left_2, key=lambda item: item[1])

    valid_circles = left_2 + right_circle
    return valid_circles


def get_bottom_valid_circles(circles):
    circles = circles.tolist()
    circles_x = sorted(circles, key=lambda item: item[0], reverse=True)
    right_2 = circles_x[:2]
    ys = np.array(right_2)[:, 1]
    assert len(ys) == 2, f'{ys} length {right_2}'
    y_min, y_max = sorted(ys)
    left_circle = [circle for circle in circles if y_min < circle[1] < y_max]
    if len(left_circle) == 2:
        left_circle = sorted(left_circle, key=lambda item: item[1])
        left_1 = [left_circle[-1]]
    elif len(left_circle) == 1:
        left_1 = left_circle
    else:
        left_1 = None

    assert left_1 is not None, f'bottom left circle can`t found in {circles}'

    right_2 = sorted(right_2, key=lambda item: item[1])
    valid_circles = left_1 + right_2
    return valid_circles

def get_color():
    color = np.random.randint(0, 256, 3)
    color = tuple([int(x) for x in color])
    return color

def cross_points2quad(cross_points):
    quad = []
    # top_left, top_right, bottom_right, bottom_left -> 1, 2, 3, 4
    quad.append([  # →
        cross_points[0][0], cross_points[0][1],
        cross_points[0][0] + 40, cross_points[0][1],
        cross_points[1][0] + 40, cross_points[1][1],
        cross_points[1][0], cross_points[1][1],
    ])
    quad.append([  # ↓
        cross_points[2][0], cross_points[2][1],
        cross_points[1][0], cross_points[1][1],
        cross_points[1][0], cross_points[1][1] + 40,
        cross_points[2][0], cross_points[2][1] + 40,
    ])
    quad.append([  # ←
        cross_points[3][0] - 40, cross_points[3][1],
        cross_points[3][0], cross_points[3][1],
        cross_points[2][0], cross_points[2][1],
        cross_points[2][0] - 40, cross_points[2][1],
    ])
    quad.append([  # ↓
        cross_points[4][0], cross_points[4][1],
        cross_points[3][0], cross_points[3][1],
        cross_points[3][0], cross_points[3][1] + 40,
        cross_points[4][0], cross_points[4][1] + 40,
    ])
    quad.append([  # ←
        cross_points[5][0] - 40, cross_points[5][1],
        cross_points[5][0], cross_points[5][1],
        cross_points[4][0], cross_points[4][1],
        cross_points[4][0] - 40, cross_points[4][1],
    ])
    quad.append([  # ↖
        cross_points[5][0] - 40 * math.sqrt(2) / 2, cross_points[5][1] - 40 * math.sqrt(2) / 2,
        cross_points[6][0] - 40 * math.sqrt(2) / 2, cross_points[6][1] - 40 * math.sqrt(2) / 2,
        cross_points[6][0], cross_points[6][1],
        cross_points[5][0], cross_points[5][1],
    ])

    quad.append([  # ↑
        cross_points[6][0], cross_points[6][1] - 40,
        cross_points[0][0], cross_points[0][1] - 40,
        cross_points[0][0], cross_points[0][1],
        cross_points[6][0], cross_points[6][1],
    ])

    return quad


def bottom_cross_points2quad(cross_points):
    quad = []
    # top_left, top_right, bottom_right, bottom_left -> 1, 2, 3, 4
    quad.append([  # →
        cross_points[0][0], cross_points[0][1],
        cross_points[0][0] + 40, cross_points[0][1],
        cross_points[1][0] + 40, cross_points[1][1],
        cross_points[1][0], cross_points[1][1],
    ])

    quad.append([  # ←
        cross_points[3][0] - 40, cross_points[3][1],
        cross_points[3][0], cross_points[3][1],
        cross_points[2][0], cross_points[2][1],
        cross_points[2][0] - 40, cross_points[2][1],
    ])
    quad.append([  # ↑
        cross_points[4][0], cross_points[4][1] - 40,
        (cross_points[3][0] + cross_points[4][0]) / 2, (cross_points[3][1] + cross_points[4][1]) / 2 - 40,
        (cross_points[3][0] + cross_points[4][0]) / 2, (cross_points[3][1] + cross_points[4][1]) / 2,
        cross_points[4][0], cross_points[4][1],
    ])
    # quad.append([  # ↑
    #     cross_points[4][0], cross_points[4][1] - 40,
    #     cross_points[3][0], cross_points[3][1] - 40,
    #     cross_points[3][0], cross_points[3][1],
    #     cross_points[4][0], cross_points[4][1],
    # ])
    quad.append([  # ←
        cross_points[5][0] - 40, cross_points[5][1],
        cross_points[5][0], cross_points[5][1],
        cross_points[4][0], cross_points[4][1],
        cross_points[4][0] - 40, cross_points[4][1],
    ])

    return quad


