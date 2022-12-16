import cv2
import math
import numpy as np

def get_shortest_lengths(pre_img, post_img):
    height, width = pre_img.shape[:2]
    shortest_lengths = []
    for y in range(height):
        for x in range(width):
            if post_img[y, x] == 0:
                distances = []
                lvl = 1
                while not distances:
                    min_distance = 2 * lvl
                    for iter_y in range(y - lvl, y + lvl + 1):
                        for iter_x in range(x - lvl, x + lvl + 1):
                            if (height > iter_y >= 0 and width > iter_x >= 0) and (iter_y in (y - lvl, y + lvl) or iter_x in (x - lvl, x + lvl)):
                                if pre_img[iter_y, iter_x] == 0:
                                    point = (y, x)
                                    distance = int(math.sqrt((abs(y - iter_y)) ** 2 + (abs(x - iter_x)) ** 2))
                                    if distance < min_distance:
                                        min_distance = distance
                                        distances.append((distance, point))
                    lvl += 1
                shortest_lengths.append(distances[-1])
    return shortest_lengths

def img_check(pre_img, post_img):
    shortest_lengths = get_shortest_lengths(pre_img, post_img)
    #length_mode = max([shortest_lengths.count(length) for length in set(shortest_lengths)])
    shortest_lengths.sort()
    trim_pct = 0.5
    lst_len = len(shortest_lengths)
    trimmed_lengths = shortest_lengths[int((lst_len/2) - (lst_len/2) * trim_pct) : (lst_len//2)] + shortest_lengths[(lst_len//2) : int((lst_len/2) * trim_pct)]
    trimmed_mean = sum(length[0] for length in trimmed_lengths)/len(trimmed_lengths)
    outlier_thresh = 5
    outliers = [length for length in shortest_lengths if abs(length[0] - trimmed_mean) > outlier_thresh]
    outlier_ct_thresh = 0.05 * lst_len
    return len(outliers) < outlier_ct_thresh

def reverse_point(point):
    return (point[1], point[0])

def construct_img(img, points, segments): # img is immediately after thresholding
    height, width = img.shape[:2]
    res_img = np.zeros((height, width, 3), np.uint8)
    res_img = ~res_img
    for i1, i2 in segments:
        cv2.line(res_img, reverse_point(points[i1]), reverse_point(points[i2]), (0, 0, 0), 1)
    res_img = cv2.cvtColor(res_img, cv2.COLOR_BGR2GRAY)
    return res_img

def is_valid_img(img, points, segments):
    res_img = construct_img(img, points, segments)
    return img_check(img, res_img)

def main():
    pts = [(13, 7), (14, 291), (202, 7), (202, 289), (58, 7), (174, 7), (209, 41), (42, 7), (9, 40), (68, 284), (113, 294), (26, 266), (182, 272), (216, 145), (200, 158), (24, 7), (26, 30), (0, 150), (20, 150)]
    segs = [[0, 1], [2, 3], [1, 3], [4, 2], [5, 6], [7, 8], [9, 10], [11, 1], [12, 3], [13, 14], [15, 16], [17, 18]]
    img = cv2.imread("atest.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = img.shape[:2]
    check = is_valid_img(img, pts, segs)
    print(check)

if __name__ == "__main__":
    main()