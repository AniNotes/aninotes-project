import requests
from bs4 import BeautifulSoup
import urllib.request
from PIL import Image
import os
import cv2 # Slow
import numpy as np
import math
from anytree import Node, RenderTree
from sentence_transformers import SentenceTransformer, util # SLOOOOOOOOOW
from random import sample
import sys

STATS = {}

def update_stats(phrase):
    STATS[phrase] = 0 if phrase not in STATS else STATS[phrase] + 1

#from multiprocessing import Pool
#from itertools import repeat



'''             |
    ARCHIVED    |
                v   '''

def point_slope_form(line):
    m = 0
    if line[2] - line[0] == 0:
        m = 10000
    else:
        m = (line[3] - line[1])/(line[2] - line[0])
    return "y - " + str(line[1]) + " = " + str(m) + "(x - " + str(line[0]) + ")"

def dot_product(vector1, vector2):
    if vector1 == None or vector2 == None:
        return "none"
    else:
        sum = 0
        for index in range(0, len(vector1)):
            sum += vector1[index] * vector2[index]
        return sum

def vector_length(vector):
    if vector == None:
        return "none"
    else:
        return math.sqrt(dot_product(vector, vector))

def calculate_angle_between(vectors):
    if vectors == "none":
        return "none"
    vector1, vector2 = vectors
    return math.acos(dot_product(vector1, vector2)/(vector_length(vector1) * vector_length(vector2)))

def get_bidirectional_vectors(line, start_point):# *
    vector1 = vector2 = (0, 0)
    if line[2] - line[0] == 0:
        vector1 = (0, 1)
        vector2 = (0, -1)
    else:
        vector1 = (1, calculate_y_val(line, start_point[0] + 1) - start_point[1])
        vector2 = (-1, calculate_y_val(line, start_point[0] - 1) - start_point[1])
    return (vector1, vector2)

def lines_to_vectors(line1, line2):
    intersection = calculate_intersection(line1, line2)
    if intersection == "none":
        return "none"
    else:
        line1_vectors = get_bidirectional_vectors(line1, intersection)
        line2_vectors = get_bidirectional_vectors(line2, intersection)
        first_vector = line1_vectors[0]
        second_vector = line2_vectors[0]
        if calculate_angle_between((first_vector, line2_vectors[1])) < calculate_angle_between((first_vector, second_vector)):
            second_vector = line2_vectors[1]
        return (first_vector, second_vector)

def swap_point_values(point):
    if point != "none":
        return (point[1], point[0])
    else:
        return "none"

def calculate_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

def get_perpendicular_distance(line1, line2):
    x1 = line1[0]
    y1 = line1[1]
    x2 = line1[2]
    y2 = line1[3]
    m = 0
    if x2 - x1 == 0:
        m = "UND"
    else:
        m = (y2 - y1)/(x2 - x1)
    m_perpedicular = -1 * (1/m)
    b1 = b2 = 0
    b1 = calculate_y_val(line1, 0)
    b2 = calculate_y_val(line2, 0)
    b = b1
    bottom_line = line1
    top_line = line2
    if b2 < b1:
        b = b2
        bottom_line = line2
        top_line = line1
    perpendicular_line = (-1 * b/m_perpedicular, 0, 0, b)
    intersection = calculate_intersection(top_line, perpendicular_line)
    side_a = calculate_distance((0, b), (intersection[0], b))
    side_b = calculate_distance((intersection[0], b), intersection)
    side_c = math.sqrt(side_a ** 2 + side_b ** 2)
    return side_c

def standard_area(img, angle):
    dimensions = img.shape
    width = dimensions[1]
    adjacent = width
    opposite = adjacent * math.sin(math.radians(angle))
    return (adjacent * opposite)/2

def calculate_triangle_area(point_A, point_B, point_C):
    side_AB = calculate_distance(point_A, point_B)
    side_BC = calculate_distance(point_B, point_C)
    side_AC = calculate_distance(point_A, point_C)
    semiperimeter = (side_AB + side_BC + side_AC)/2
    return math.sqrt(semiperimeter * (semiperimeter - side_AB) * (semiperimeter - side_BC) * (semiperimeter - side_AC))

def round_point(point):
    return (round(point[0]), round(point[1]))

def calculate_inside_area(img, line1, line2):
    dimensions = img.shape
    height = dimensions[0]
    width = dimensions[1]
    top_x_axis = (0, height, width, height)
    bottom_x_axis = (0, 0, width, 0)
    left_y_axis = (0, 0, 0, height)
    right_y_axis = (width, 0, width, height)
    axes = (left_y_axis, top_x_axis, right_y_axis, bottom_x_axis)
    x_axes = (top_x_axis, bottom_x_axis)
    y_axes = (left_y_axis, right_y_axis)
    axes_intersections = {top_x_axis: [], bottom_x_axis: [], left_y_axis: [], right_y_axis: []}
    intersection_points = []
    for axis in axes:
        line1_intersection = calculate_intersection(line1, axis)
        line2_intersection = calculate_intersection(line2, axis)
        if line1_intersection != "none":
            if (line1_intersection[0] >= 0 and line1_intersection[0] <= width) and (line1_intersection[1] >= 0 and line1_intersection[1] <= height):
                axes_intersections[axis].append(line1)
                intersection_point = round_point(line1_intersection)
                if intersection_point not in intersection_points:
                    intersection_points.append(intersection_point)
        if line2_intersection != "none":
            if (line2_intersection[0] >= 0 and line2_intersection[0] <= width) and (line2_intersection[1] >= 0 and line2_intersection[1] <= height):
                axes_intersections[axis].append(line2)
                intersection_point = round_point(line2_intersection)
                if intersection_point not in intersection_points:
                    intersection_points.append(intersection_point)
    if len(intersection_points) == 3:
        return calculate_triangle_area(intersection_points[0], intersection_points[1], intersection_points[2])
    line1_balance_point = line2_balance_point = (0, 0)
    line1_nonbalance_point = line2_nonbalance_point = (0, 0)
    two_intersections = False
    for axis in axes_intersections:
        if len(axes_intersections[axis]) == 2:
            two_intersections = True
            line1_balance_point = calculate_intersection(line1, axis)
            line2_nonbalance_point = calculate_intersection(line2, axis)
            for second_axis in axes_intersections:
                if second_axis != axis:
                    if line2 in axes_intersections[second_axis]:
                        line2_balance_point = calculate_intersection(line2, second_axis)
                        line1_nonbalance_point = calculate_intersection(line1, second_axis)
                        break
            break
    if not(two_intersections):
        for axis in axes_intersections:
            if line1 in axes_intersections[axis]:
                line2_intersect_axis = None
                if line2 in axes_intersections[axes[(axes.index(axis) + 1 + 4) % 4]]:
                    line2_intersect_axis = axes[(axes.index(axis) + 1 + 4) % 4]
                elif line2 in axes_intersections[axes[(axes.index(axis) - 1 + 4) % 4]]:
                    line2_intersect_axis = axes[(axes.index(axis) - 1 + 4) % 4]
                if line2_intersect_axis != None:
                    line1_balance_point = calculate_intersection(line1, axis)
                    line2_nonbalance_point = calculate_intersection(line2, line2_intersect_axis)
                    for second_axis in axes_intersections:
                        if second_axis != line2_intersect_axis and line2 in axes_intersections[second_axis]:
                            line2_balance_point = calculate_intersection(line2, second_axis)
                            break
                    for second_axis in axes_intersections:
                        if second_axis != axis and line1 in axes_intersections[second_axis]:
                            line1_nonbalance_point = calculate_intersection(line1, second_axis)
                            break
                break
    return calculate_triangle_area(line1_balance_point, line2_balance_point, line1_nonbalance_point) + calculate_triangle_area(line1_balance_point, line2_balance_point, line2_nonbalance_point)

def are_distinct(img, line1, line2):
    l1x1 = line1[0]
    l1y1 = line1[1]
    l1x2 = line1[2]
    l1y2 = line1[3]
    l2x1 = line2[0]
    l2y1 = line2[1]
    l2x2 = line2[2]
    l2y2 = line2[3]
    m1 = m2 = 0
    if l1x2 - l1x1 == 0:
        m1 = 100
    else:
        m1 = (l1y2 - l1y1)/(l1x2 - l1x1)
    if l2x2 - l2x1 == 0:
        m2 = 100
    else:
        m2 = (l2y2 - l2y1)/(l2x2 - l2x1)
    dimensions = img.shape
    intersection = swap_point_values(calculate_intersection(line1, line2))
    if intersection == "none":
        max_angle = 8.5
        if calculate_inside_area(img, line1, line2) <= standard_area(img, max_angle):
            return False
        else:
            return True
    else:
        intersection_x = intersection[0]
        intersection_y = intersection[1]
        height = dimensions[0]
        width = dimensions[1]
        x_val = width/2
        y_val = height/2
        y_val1 = calculate_y_val(line1, x_val)
        y_val2 = calculate_y_val(line2, x_val)
        x_val1 = calculate_x_val(line1, y_val)
        x_val2 = calculate_x_val(line2, y_val)
        slope_threshold = 1
        are_close = False
        max_angle = 8.5
        if calculate_inside_area(img, line1, line2) <= standard_area(img, max_angle):
            are_close = True
        if (((intersection_y >= 0 and intersection_y <= height) and (intersection_x >= 0 and intersection_x <= width)) or are_close == True) and calculate_angle_between(lines_to_vectors(line1, line2)) <= max_angle * (np.pi/180):
            return False
        else:
            return True

def eliminate_border_lines(img, lines):
    dimensions = img.shape
    height = dimensions[0]
    width = dimensions[1]
    top_x_axis = (0, height, width, height)
    bottom_x_axis = (0, 0, width, 0)
    left_y_axis = (0, 0, 0, height)
    right_y_axis = (width, 0, width, height)
    axes = (top_x_axis, bottom_x_axis, left_y_axis, right_y_axis)
    slope_bound = math.radians(1)
    closeness_bound = 3
    result_lines = []
    for line in lines:
        not_border = True
        for axis in axes:
            m_line = calculate_slope(line, True)
            m_axis = calculate_slope(axis, True)
            angle_between = calculate_angle_between(lines_to_vectors(line, axis))
            if angle_between == "none":
                if m_line > 1 and m_axis > 1:
                    x_val_line = calculate_x_val(line, height/2)
                    x_val_axis = calculate_x_val(axis, height/2)
                    if abs(x_val_line - x_val_axis) <= closeness_bound:
                        not_border = False
                elif m_line < 1 and m_axis < 1:
                    y_val_line = calculate_y_val(line, width/2)
                    y_val_axis = calculate_y_val(axis, width/2)
                    if abs(y_val_line - y_val_axis) <= closeness_bound:
                        not_border = False
            elif angle_between <= slope_bound:
                if m_line > 1 and m_axis > 1:
                    x_val_line = calculate_x_val(line, height/2)
                    x_val_axis = calculate_x_val(axis, height/2)
                    if abs(x_val_line - x_val_axis) <= closeness_bound:
                        not_border = False
                elif m_line < 1 and m_axis < 1:
                    y_val_line = calculate_y_val(line, width/2)
                    y_val_axis = calculate_y_val(axis, width/2)
                    if abs(y_val_line - y_val_axis) <= closeness_bound:
                        not_border = False
        if not_border:
            result_lines.append(line)
    return result_lines

def calculate_transformed_points(img, line):
    dimensions = img.shape
    height = dimensions[0]
    width = dimensions[1]
    x1 = line[0]
    y1 = line[1]
    x2 = line[2]
    y2 = line[3]
    m = calculate_slope(line, True)
    new_y1 = height - y1
    p1 = (x1, new_y1)
    p2 = (0, m * x1 + new_y1)
    return p1 + p2

def remove_holes(img):
    thick = thicken_img(img, 1)
    thick = cv2.cvtColor(thick, cv2.COLOR_BGR2GRAY)
    _, prethin = cv2.threshold(thick, 128, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)
    first_thin = cv2.ximgproc.thinning(prethin)
    first_thin = ~first_thin
    final = thin_img(first_thin)
    return final

'''             ^
    ARCHIVED    |
                |   '''

# -------------------------------

'''                            |
    CENTRAL IMAGE MANAGEMENT    |
                               v   '''

# Manages the lookup table of my main image that contains all necessary information about the image.
'''class img_data:
    def __init__(self, height, width):
        self.height = height
        self.width = width

        # 2D array with pixel data at indices. The first element of the data is a 2-element list with the first
        # value as a boolean value that is True if the image at that point is black (img is black and white), and
        # the second value as a list of the information about that point. The second element of the main data is
        # a list of the neighbors to that point. The neighbors are stored as the first items in their main data.
        arr = [[[[False, []], []] for y in range(height) for x in range(width)]]
        white_points = []
        for y in range(height):
            for x in range(width):
                white_points.append((y, x))
                px_data = arr[y, x]
                px, nbrs = px_data
                mark, info = px

                # The neighbors are put into two separate lists: one for adjacent neighbors, and the other for
                # diagonal neighbors.
                adj_nbrs = []
                dia_nbrs = []

                # The adjcent neighbors are stored in the order of: "UP", "LEFT", "RIGHT", "DOWN". The diagonal
                # neighbors are stored in the order of: "TOP LEFT", "TOP RIGHT", "BOTTOM LEFT", "BOTTOM RIGHT".
                for iy in range(y - 1, y + 2):
                    for ix in range(x - 1, x + 2):
                        if not(iy == y and ix == x):
                            if 0 <= iy < height and 0 <= ix < width:
                                nbr = arr[iy, ix][0]
                                if iy == y or ix == x:
                                    adj_nbrs.append(nbr)
                                else:
                                    dia_nbrs.append(nbr)
                nbrs.append(adj_nbrs)
                nbrs.append(dia_nbrs)
        self.img = np.array(arr)
        self.white_points = white_points
        self.black_points = []

    # Marks or unmarks all the points in point_set (depends on the boolean value of change).
    def change_points(point_set, change):
        for y, x in point_set:
            self.img[y, x][0][0] = change'''

# HEIGHT = WIDTH = 0
#IMG_DCT = {}
#WHITE_POINTS = set()
#BLACK_POINTS = set()
#WHITE_POINTS_WHITE_NBRS = {}
#WHITE_POINTS_BLACK_NBRS = {}
#BLACK_POINTS_WHITE_NBRS = {}
#BLACK_POINTS_BLACK_NBRS = {}
IMG_INFO = []#[(DIMENSIONS, IMG_DCT, WHITE_POINTS, BLACK_POINTS, WHITE_POINTS_WHITE_NBRS, WHITE_POINTS_BLACK_NBRS, BLACK_POINTS_WHITE_NBRS, BLACK_POINTS_BLACK_NBRS)]

def img_dct_idx(WIDTH, y, x):
    #WIDTH = IMG_INFO[img_idx][0][1]
    return x + WIDTH * y

def img_arr_idx(WIDTH, i):
    #WIDTH = IMG_INFO[img_idx][0][1]
    return (i // WIDTH, i % WIDTH)

def initialize_img(DIMENSIONS):
    HEIGHT, WIDTH = DIMENSIONS
    IMG_DCT = {i : [[i, False], []] for i in range(HEIGHT * WIDTH)}
    WHITE_POINTS = set()
    BLACK_POINTS = set()
    WHITE_POINTS_WHITE_NBRS = {}
    WHITE_POINTS_BLACK_NBRS = {}
    BLACK_POINTS_WHITE_NBRS = {}
    BLACK_POINTS_BLACK_NBRS = {}
    for y in range(HEIGHT):
        for x in range(WIDTH):
            i_pt = img_dct_idx(WIDTH, y, x)
            WHITE_POINTS.add(i_pt)
            px_data = IMG_DCT[i_pt]
            px, nbrs = px_data
            for iy in range(y - 1, y + 2):                                          #   1 2 3
                for ix in range(x - 1, x + 2):                                      #   4 * 5
                    if not(iy == y and ix == x):                                    #   6 7 8
                        if 0 <= iy < HEIGHT and 0 <= ix < WIDTH:
                            nbr = IMG_DCT[img_dct_idx(WIDTH, iy, ix)][0]
                            nbrs.append(nbr)
    img_idx = len(IMG_INFO)
    IMG_INFO.append((DIMENSIONS, IMG_DCT, WHITE_POINTS, BLACK_POINTS, WHITE_POINTS_WHITE_NBRS, WHITE_POINTS_BLACK_NBRS, BLACK_POINTS_WHITE_NBRS, BLACK_POINTS_BLACK_NBRS))
    return img_idx

def change_points(point_set, change, img_idx, point_type): # point_type: 0 = i_pt ; 1 = yx_pt
    DIMENSIONS, IMG_DCT, WHITE_POINTS, BLACK_POINTS, WHITE_POINTS_WHITE_NBRS, WHITE_POINTS_BLACK_NBRS, BLACK_POINTS_WHITE_NBRS, BLACK_POINTS_BLACK_NBRS = IMG_INFO[img_idx]
    HEIGHT, WIDTH = DIMENSIONS
    for pt in point_set:
        i_pt = pt if point_type == 0 else img_dct_idx(WIDTH, pt[0], pt[1])
        pre_change = IMG_DCT[i_pt][0][1]
        IMG_DCT[i_pt][0][1] = change
        if pre_change != change:

            # If I change to white...
            if change == False:
                if i_pt not in WHITE_POINTS:
                    BLACK_POINTS.remove(i_pt)
                    if i_pt in BLACK_POINTS_WHITE_NBRS:
                        BLACK_POINTS_WHITE_NBRS.pop(i_pt)
                    if i_pt in BLACK_POINTS_BLACK_NBRS:
                        BLACK_POINTS_BLACK_NBRS.pop(i_pt)
                    WHITE_POINTS.add(i_pt)
                    white_nbrs = set()
                    black_nbrs = set()
                    for nbr in IMG_DCT[i_pt][1]:
                        nbr_idx, mark = nbr
                        if mark == False:
                            white_nbrs.add(nbr_idx)
                            if nbr_idx not in WHITE_POINTS_WHITE_NBRS:
                                WHITE_POINTS_WHITE_NBRS[nbr_idx] = set()
                            WHITE_POINTS_WHITE_NBRS[nbr_idx].add(i_pt)
                        else:
                            black_nbrs.add(nbr_idx)
                            if nbr_idx not in BLACK_POINTS_WHITE_NBRS:
                                BLACK_POINTS_WHITE_NBRS[nbr_idx] = set()
                            BLACK_POINTS_WHITE_NBRS[nbr_idx].add(i_pt)
                            BLACK_POINTS_BLACK_NBRS[nbr_idx].remove(i_pt)
                    if white_nbrs:
                        WHITE_POINTS_WHITE_NBRS[i_pt] = white_nbrs
                    if black_nbrs:
                        WHITE_POINTS_BLACK_NBRS[i_pt] = black_nbrs

            # If I change to black...
            else:
                if i_pt not in BLACK_POINTS:
                    WHITE_POINTS.remove(i_pt)
                    if i_pt in WHITE_POINTS_BLACK_NBRS:
                        WHITE_POINTS_BLACK_NBRS.pop(i_pt)
                    BLACK_POINTS.add(i_pt)
                    white_nbrs = set()
                    black_nbrs = set()
                    for nbr in IMG_DCT[i_pt][1]:
                        nbr_idx, mark = nbr
                        if mark == False:
                            white_nbrs.add(nbr_idx)
                            if nbr_idx not in WHITE_POINTS_BLACK_NBRS:
                                WHITE_POINTS_BLACK_NBRS[nbr_idx] = set()
                            WHITE_POINTS_BLACK_NBRS[nbr_idx].add(i_pt)
                        else:
                            black_nbrs.add(nbr_idx)
                            if nbr_idx not in BLACK_POINTS_BLACK_NBRS:
                                BLACK_POINTS_BLACK_NBRS[nbr_idx] = set()
                            BLACK_POINTS_BLACK_NBRS[nbr_idx].add(i_pt)
                            BLACK_POINTS_WHITE_NBRS[nbr_idx].remove(i_pt)
                    if white_nbrs:
                        BLACK_POINTS_WHITE_NBRS[i_pt] = white_nbrs
                    if black_nbrs:
                        BLACK_POINTS_BLACK_NBRS[i_pt] = black_nbrs

def get_neighbor_segments(img_idx, i_pt):
    nbrs = IMG_INFO[img_idx][1][i_pt][1]
    nbr_order = (0, 1, 2, 4, 7, 6, 5, 3)
    nbr_order_inds = {0 : 0, 1 : 1, 2 : 2, 4 : 3, 7 : 4, 6 : 5, 5 : 6, 3 : 7}
    nbr_marks = [nbrs[i][1] for i in nbr_order]
    segments = [[nbrs[nbr_order_inds[i]][0] for i in segment_str.split(".")[: -1]] for segment_str in list(filter(None, "".join([" ", f"{i}."][nbr_marks[i]] for i in range(8)).split(" ")))]
    if nbr_marks[0] == nbr_marks[7] == True:
        segments[0] += segments.pop(7)
    return segments

def restart_img(white_points, black_points, img_idx):
    change_points(white_points, False, img_idx, 1)
    change_points(black_points, True, img_idx, 1)

def reconstruct_img(img_idx):
    res_img_info = IMG_INFO[img_idx]
    HEIGHT, WIDTH = res_img_info[0]
    WHITE_POINTS = res_img_info[2]
    res_img = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    for i_pt in WHITE_POINTS:
        y, x = img_arr_idx(WIDTH, i_pt)
        res_img[y, x] = (255, 255, 255)
    return res_img

def to_cv_img(img_idx):
    HEIGHT, WIDTH = IMG_INFO[img_idx][0]
    WHITE_POINTS = IMG_INFO[img_idx][2]
    cv_img = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    for i_pt in WHITE_POINTS:
        y, x = img_arr_idx(WIDTH, i_pt)
        cv_img[y, x] = (255, 255, 255)
    return cv_img

'''if __name__ == "__main__":
    HEIGHT = 300
    WIDTH = 260
    DIMENSIONS = (HEIGHT, WIDTH)
    initialize_img(DIMENSIONS)
    change_points({(0, 0), (1, 0)}, True)
    print(len(WHITE_POINTS_BLACK_NBRS))
    print(len(BLACK_POINTS_WHITE_NBRS))
    print(len(BLACK_POINTS_BLACK_NBRS))
    print(WHITE_POINTS_BLACK_NBRS)
    print(BLACK_POINTS_WHITE_NBRS)
    print(BLACK_POINTS_BLACK_NBRS)'''

'''                            ^
    CENTRAL IMAGE MANAGEMENT    |
                               |   '''

# -------------------------------

'''                      |
    IMAGE PROCUREMENT    |
                         v   '''

# Takes in a term and finds the (best) wikipedia page on that term.
def get_img_urls(text, sem_check):
    wiki = "https://en.wikipedia.org"
    formatted_text = text.replace(" ", "_")
    initial_href = "/wiki/" + formatted_text
    initial_url = wiki + initial_href
    page = requests.get(initial_url)
    initial_soup = BeautifulSoup(page.content, 'html.parser')
    initial_p_tags = initial_soup.find_all('p')
    is_redirection_page = False

    # Finds if the initial page is a redirection page, in which case I need another step to get to the correct page.
    for tag in initial_p_tags:
        tag = str(tag)
        if "may refer to:" in tag:
            is_redirection_page = True
            break
    explanatory_page_link = initial_url

    # The next step.
    if is_redirection_page:
        title_tags = initial_soup.find_all('span', class_ = "mw-headline")

        # Titles of the possible next pages.
        titles = []
        for tag in title_tags:
            titles.append(tag.get("id"))
        if "See_also" in titles:
            titles.remove("See_also")
        li_tags = initial_soup.find_all('li')
        tag_count = 0
        for tag in li_tags:
            bad_title = False
            ul_tags = tag.find_all("ul")
            if ul_tags != None:
                for ul_tag in ul_tags:
                    ul_li_tags = ul_tag.find_all("li")
                    if ul_li_tags != False:
                        for ul_li_tag in ul_li_tags:
                            ul_li_tags_classes = ul_li_tag.get("class")
                            if ul_li_tags_classes != None:

                                # 'toclevel' is only in the tags of titles that don't directly link to new pages
                                # (I think 'Science, technology, and mathematics' in https://en.wikipedia.org/wiki/Set
                                # should be an example).
                                if "toclevel" in ul_li_tags_classes[0]:
                                    bad_title = True
                                    break
                    if bad_title:
                        break
            if bad_title:
                titles.pop(tag_count)
                tag_count -= 1
            tag_count += 1

        # The title whose links I use is determined by semantic similarity to the word 'mathematics'.
        model = SentenceTransformer('all-MiniLM-L6-v2')
        scores = {}
        for title in titles:
            words = [title]
            if "_" in title:
                words = title.split("_")
            math_words = []
            for _ in words:
                math_words.append(sem_check)
            embeddings1 = model.encode(math_words, convert_to_tensor = True)
            embeddings2 = model.encode(words, convert_to_tensor = True)
            cosine_scores = util.cos_sim(embeddings1, embeddings2)
            total_score = 0
            for i in range(len(math_words)):
                total_score += cosine_scores[i][i].item()

            # If there are multiple words in the title, I take the average of their similarity score.
            average_score = total_score/len(math_words)
            scores[title] = average_score

        # The final title whose links I will use.
        semantically_closest_title = titles[0]
        for title in titles:
            if scores[title] > scores[semantically_closest_title]:
                semantically_closest_title = title
        ul_tags = initial_soup.find_all("ul")
        next_page_hrefs = []

        # The position of the selected title among all of the possible titles on the page.
        link_index = titles.index(semantically_closest_title)
        for tag in ul_tags:
            next_page_link_tag = False
            for subtag in tag:
                subtag = str(subtag)
                subtag_chunks = subtag.split(" ")
                for chunk in subtag_chunks:
                    if "href=" in chunk:
                        href = chunk[6:len(chunk) - 1]

                        # A check to see if the current tag is representative of a title that links to other pages.
                        if "/wiki/" in href:

                            # If the current tag is representative of the selected title...
                            if link_index == 0:
                                next_page_hrefs.append(href)
                            next_page_link_tag = True

            # Decrease link_index if the tag is of a linking title to ultimately arrive at the selected title.
            if next_page_link_tag:
                link_index -= 1
        next_page_href = next_page_hrefs[0]
        explanatory_page_link = wiki + next_page_href

    # Where the image links of the page will be stored
    file = text.replace(" ", "_") + "_IMAGELINKS.txt"
    path = "math_obj_image_links/" + file
    if not os.path.isfile(path):
        with open(path, "a") as f:
            page = requests.get(explanatory_page_link).text
            img_soup = BeautifulSoup(page, 'html.parser')

            # Finding the images.
            for raw_img in img_soup.find_all('img'):
                link = raw_img.get('src')
                if link:

                    # Some wikipedia images are used across different pages, and they are for the site instead of the
                    # topic(s) described on the page.
                    ubiquitous_links = [
                    "//upload.wikimedia.org/wikipedia/en/thumb/1/1b/Semi-protection-shackle.svg/20px-Semi-protection-shackle.svg.png",
                    "//upload.wikimedia.org/wikipedia/commons/thumb/9/99/Wiktionary-logo-en-v2.svg/40px-Wiktionary-logo-en-v2.svg.png",
                    "//upload.wikimedia.org/wikipedia/en/thumb/9/99/Question_book-new.svg/50px-Question_book-new.svg.png",
                    "//upload.wikimedia.org/wikipedia/commons/thumb/0/0f/Mergefrom.svg/50px-Mergefrom.svg.png",
                    "//upload.wikimedia.org/wikipedia/commons/thumb/a/a4/Text_document_with_red_question_mark.svg/80px-Text_document_with_red_question_mark.svg.png",
                    "//upload.wikimedia.org/wikipedia/commons/thumb/a/a4/Text_document_with_red_question_mark.svg/40px-Text_document_with_red_question_mark.svg.png",
                    "//upload.wikimedia.org/wikipedia/commons/thumb/3/3e/Nuvola_apps_edu_mathematics_blue-p.svg/16px-Nuvola_apps_edu_mathematics_blue-p.svg.png",
                    "//en.wikipedia.org/wiki/Special:CentralAutoLogin/start?type=1x1"
                    ]
                    if link not in ubiquitous_links:
                        if "https:" not in link:
                            link = "https:" + link
                        f.write(link + "\n")
            f.close()

    # Return the path of the image link file.
    return file

# Takes in the image link file and downloads all of the images linked to within it to a folder.
def download_images(file):

    # object is the term that was inputted into the image links function.
    object = file[0:file.index("IMAGELINKS") - 1]

    # Where the images will be downloaded.
    folder_name = object + "_images"
    dir = os.path.join("math_obj_images/original_images/", folder_name)
    if not os.path.exists(dir):
        os.mkdir(dir)
        with open("math_obj_image_links/" + file, "r") as f:
            count = 1
            for line in f:
                try:
                    line = line.strip()
                    if line[0] != "/":
                        content = requests.get(line).content

                        # Determine the ending of the image (ext).
                        ext = -1
                        if content.startswith(b'<svg'):
                            ext = '.svg'
                        elif line.endswith('.png'):
                            ext = '.png'
                        elif line.endswith('.jpg'):
                            ext = '.jpg'
                        if ext == -1 or ext == ".svg":
                            continue

                        # Write the image ending to a certain file.
                        file = object + "_IMAGEENDINGS.txt"
                        path = "math_obj_image_endings/" + file
                        with open(path, "a") as f2:
                            f2.write(ext + "\n")
                        f2.close()
                        urllib.request.urlretrieve(line, "math_obj_images/original_images/" + folder_name + "/" + object + str(count) + ext)
                        count += 1
                except:
                    continue
            f.close()

    # Create two new folders for image preparation in the future.
    dir2 = os.path.join("math_obj_images/opaque_images/", "opaque_" + folder_name)
    if not os.path.exists(dir2):
        os.mkdir(dir2)
    dir3 = os.path.join("math_obj_images/resized_images/", "resized_" + folder_name)
    if not os.path.exists(dir3):
        os.mkdir(dir3)

# Checks if the text is the name of a symbol that Manim can display through LaTex.
def is_symbol(text):
    symbol_dict = {"mu": "µ", "pi": "π"} # WILL FILL LATER
    return symbol_dict[text] if text in symbol_dict else False

# Creates a tuple with the text representing the object and the object's type.
def create_object(text):

    # Type 1 = text ; Type 2 = image.
    obj_text = obj_type = -1
    symbol = is_symbol(text)
    if symbol != False:
        obj_text = symbol
        obj_type = 1
    else:
        obj_text = text
        file = get_img_urls(text, "mathematics")
        download_images(file)
        obj_type = 2
    return (obj_text, obj_type)

'''                      ^
    IMAGE PROCUREMENT    |
                         |   '''

# -------------------------------

'''                      |
    IMAGE PREPARATION    |
                         v   '''

#                                                     .  .  .
# Neighborhood of range 1 around a point (*) --->     .  *  .
#                                                     .  .  .

def get_sets_of_colored_points_binary(img_idx, value):
    i_point_sets = []
    POINTS = IMG_INFO[img_idx][2] if value == False else IMG_INFO[img_idx][3]
    for i_pt in POINTS:
        in_new_object = True
        for i_point_set in i_point_sets:
            if i_pt in i_point_set:
                in_new_object = False
                break
        if in_new_object:
            i_point_sets.append(get_set_of_points_from_point_binary(img_idx, i_pt, value))
    return i_point_sets

# Returns a list of all continuous groups of points in img with the same characteristic as denoted by color.
#
# For color: 0 = black ; 1 = white ; 2 = nonwhite ; 3 = pseudowhite ; 4 = transparent
def get_sets_of_colored_points(img, color, pil_image = -1):
    height, width = img.shape[0:2]
    object_point_lists = []
    for y in range(0, height):
        for x in range(0, width):
            is_transparent = -1
            if color not in (0, 1):
                is_white = True
                for val in img[y, x]:
                    if val < 248:
                        is_white = False
                if color == 4:
                    is_transparent = pil_image.getpixel((x, y))[3] == 0
            if (color == 0 and img[y, x] == 0) or (color == 1 and img[y, x] == 255) or (color == 2 and not(is_white)) or (color == 3 and is_white) or is_transparent == True:
                point = (y, x)
                in_new_object = True
                for list in object_point_lists:
                    if point in list:
                        in_new_object = False
                        break
                if in_new_object:
                    set_of_points = get_set_of_points_from_point(img, point, color, pil_img = pil_image)
                    object_point_lists.append(set_of_points)
    return object_point_lists

# Returns a black and white image where pixels are colored in if they are distinctly different from background_color
# in img.
def background_threshold(img, background_color, points, edge_point):
    height, width = img.shape[0:2]
    thresh = np.zeros((height, width, 3), np.uint8)
    thresh = ~thresh
    gradient_points = get_set_of_points_from_point(img, edge_point, 5)
    if len(points) - 5 <= len(gradient_points) <= len(points):
        for y, x in gradient_points:
            thresh[y, x] = 0
    else:
        for y, x in points:
            is_background = True
            for val in range(0, 3):
                if not(img[y, x][val] >= background_color[val] - 8 and img[y, x][val] <= background_color[val] + 8):
                    is_background = False
            if not is_background:
                thresh[y, x] = 0
    return thresh

# PRECONDITION: img is thresholded and contains one object.
#
# Returns whether or not the img is "hollow" (i.e. the object within it has an inside.)
def is_hollow(img):
    height, width = img.shape[0:2]
    edges = np.zeros((height, width, 3), np.uint8)
    edges = ~edges
    drawn_points = []
    for y in range(0, height):
        for x in range(0, width):
            if img[y, x] == 255:
                is_edge = False

                # Checks if this (white) point is on the edge. i.e. at least one point that neighbors it in a range
                # of 1 is black.
                for iter_y in range(y - 1, y + 2):
                    for iter_x in range(x - 1, x + 2):
                        if not(iter_y == y and iter_x == x) and iter_y >= 0 and iter_y < height and iter_x >= 0 and iter_x < width:
                            if img[iter_y, iter_x] == 0:
                                is_edge = True

                # If this point is on the edge, draw it to my edges img and record it in drawn_points.
                if is_edge:
                    drawn_points.append((y, x))
                    edges[y, x] = (0, 0, 0)
            else:

                # The only time a black point is considered on the edge is when it touches the edge of the img.
                if y == 0 or y == height - 1 or x == 0 or x == width - 1:
                    drawn_points.append((y, x))
                    edges[y, x] = (0, 0, 0)

    # Find all of the groups of points in edges that are black.
    point_sets = get_sets_of_colored_points(edges, 2)

    # Returns whether or not img is hollow, which occurs when the edges img has more than one group of black points,
    # and the points that were drawn to edges (drawn_points).
    return len(point_sets) != 1, drawn_points

# PRECONDITION: img is black and white.
#
# Returns a list of points of the edge of the object given by obj_points on img.
def get_edge_points(img, obj_points, removed_points, boundary_layers):
    height, width = img.shape[0:2]

    # The points on the very edge of obj_points.
    edge_points = []

    # The points not on the very edge of obj_points.
    new_obj_points = []
    for y, x in obj_points:
        point = (y, x)

        # Points on the edge of the bounds of img are automatically considered edge points.
        is_edge = y == 0 or y == height - 1 or x == 0 or x == width - 1
        if is_edge:
            edge_points.append(point)
        else:

            # Marks the point as an edge point if a white point is a neighbor to it in a range of 1.
            for iter_y in range(y - 1, y + 2):
                for iter_x in range(x - 1, x + 2):
                    if img[iter_y, iter_x] == 255:
                        is_edge = True
            if is_edge:
                edge_points.append(point)
            else:
                new_obj_points.append(point)

    # If I am done removing the outer layers of the object, return relevant information.
    if boundary_layers == 0:
        return edge_points, removed_points

    # Otherwise, remove the edge points, add them to the removed points, and recursively call this function with
    # boundary layers minus one.
    else:
        for y, x in edge_points:
            removed_points.append((y, x))
            img[y, x] = 255
        return get_edge_points(img, new_obj_points, removed_points, boundary_layers - 1)

# PRECONDITION: img is black and white.
#
# Fills in all small holes (black or white) with sizes less than max_size or a smart size if max_size is equal to -1.
def remove_small_objects(img_idx, max_size = -1):

    # List of black point groups in img.
    black_point_sets = get_sets_of_colored_points_binary(img_idx, True)
    black_point_count = sum(len(point_list) for point_list in black_object_point_lists)

    # If max_size is -1, make it a "smart" size equal to roughly log base 2 of the number of black points in img. The
    # idea behind this choice is that the black points are the main points I care about for this function.
    if max_size == -1:
        max_size = black_point_count.bit_length() - 1

    # Fill in all the small black groups.
    for point_set in black_point_sets:
        if len(point_set) < max_size:
            change_points(point_set, False, img_idx, 0)

# PRECONDITION: img has no transparent points.
#
# Returns a black and white image containing the objects in img.
def get_thresholded_img(img, img_idx, current_points, edge_points):

    # Don't continue if the current points are small in number (extraneous).
    if len(current_points) > 5:
        height, width = img.shape[0:2]

        edge_colors = []
        edge_colors_set = set()
        for y, x in edge_points:
            v1, v2, v3 = img[y, x]
            edge_colors.append((v1, v2, v3))
            edge_colors_set.add((v1, v2, v3))
        color_mode = 1
        for color in edge_colors_set:
            if edge_colors.count(color) > color_mode:
                color_mode = edge_colors.count(color)
        background_color = (255, 255, 255)
        for color in edge_colors_set:
            if edge_colors.count(color) == color_mode:
                background_color = color
                break
        edge_point = edge_points[0]

        # current_thresh is an image with points in current_points colored black if they have significantly
        # different colors in img than the background color.
        current_thresh = background_threshold(img, background_color, current_points, edge_point)
        current_thresh = cv2.cvtColor(current_thresh, cv2.COLOR_BGR2GRAY)

        # Do a smart removal of small objects from current_thresh.
        black_point_sets = get_sets_of_colored_points(current_thresh, 0)
        black_point_count = sum(len(point_list) for point_list in black_point_sets)
        white_point_sets = get_sets_of_colored_points(current_thresh, 1)
        max_size = black_point_count.bit_length() - 1
        for point_set in black_point_sets:
            if len(point_set) < max_size:
                for y, x in point_set:
                    current_thresh[y, x] = 255
                black_point_sets.remove(point_set)
        for point_set in white_point_sets:
            if len(point_set) < max_size:
                for y, x in point_set:
                    current_thresh[y, x] = 0
                white_point_sets.remove(point_set)

        # All of the black groups of points in current_thresh.
        for point_set in black_point_sets:
            current_object = np.zeros((height, width, 3), np.uint8)
            current_object = ~current_object
            current_object = cv2.cvtColor(current_object, cv2.COLOR_BGR2GRAY)

            # Draw object to current_object.
            for y, x in point_set:
                current_object[y, x] = 0
            hollow, drawn_points = is_hollow(current_object)
            draw_points = set()

            # If object is hollow, draw it exactly to thresh.
            if hollow:
                for point in point_set:
                    draw_points.add(point)

            # Otherwise, draw its edge to thresh.
            else:
                for point in drawn_points:
                    draw_points.add(point)
            print(f"GET THRESH: {len(draw_points)}")
            change_points(draw_points, True, img_idx, 1)

            # removed_points are the first five outer layers of object, and current_edge_points is the sixth.
            current_edge_points, removed_points = get_edge_points(current_object, point_set, [], 6)

            # Remove the first five layers from object.
            for point in removed_points:
                point_set.remove(point)

            # Draw onto thresh the threshold of all objects present within img. What this does is it gets objects
            # inside of other objects that have distinctly different colors. For an example, look at "set" on
            # wikipedia.
            get_thresholded_img(img, img_idx, point_set, current_edge_points)

def white_to_black_transitions(yx_pt, black_nbrs, WIDTH):
    y, x = yx_pt
    nbr_positions = ((y - 1, x), (y - 1, x + 1), (y, x + 1), (y + 1, x + 1), (y + 1, x), (y + 1, x - 1), (y, x - 1), (y - 1, x - 1))
    transitions = sum(1 for i, (i_y, i_x) in enumerate(nbr_positions) if img_dct_idx(WIDTH, i_y, i_x) in black_nbrs and img_dct_idx(WIDTH, nbr_positions[i - 1][0], nbr_positions[i - 1][1]) not in black_nbrs)

    return transitions

# PRECONDITION: img is black and white.
#
# Zhang-Suen Skeletonization.
def thin_img(img_idx):
    '''info = IMG_INFO[img_idx]
    HEIGHT, WIDTH = info[0]
    WHITE_POINTS = info[2]
    BLACK_POINTS = info[3]
    BLACK_POINTS_BLACK_NBRS = info[7]
    done = False
    print("start while")
    c = 0
    while not done:
        c += 1
        print(c)
        done = True
        erase_points = set()
        for i_pt in BLACK_POINTS:
            yx_pt = img_arr_idx(WIDTH, i_pt)
            y, x = yx_pt
            P2 = img_dct_idx(WIDTH, y - 1, x)
            P4 = img_dct_idx(WIDTH, y, x + 1)
            P6 = img_dct_idx(WIDTH, y + 1, x)
            P8 = img_dct_idx(WIDTH, y, x - 1)
            black_nbrs = BLACK_POINTS_BLACK_NBRS[i_pt]
            cond_0 = 0 < y < HEIGHT - 1 and 0 < x < WIDTH - 1
            cond_1 = 2 <= len(black_nbrs) <= 6
            cond_2 = white_to_black_transitions(yx_pt, black_nbrs, WIDTH) == 1
            cond_3 = P2 in WHITE_POINTS or P4 in WHITE_POINTS or P6 in WHITE_POINTS
            cond_4 = P4 in WHITE_POINTS or P6 in WHITE_POINTS or P8 in WHITE_POINTS
            if cond_0 and cond_1 and cond_2 and cond_3 and cond_4:
                erase_points = erase_points | {i_pt, P2, P4, P6, P8}
        if erase_points:
            change_points(erase_points, False, img_idx, 0)
            print(f"change length 1: {len(erase_points)}")
            done = False
        erase_points = set()
        for i_pt in BLACK_POINTS:
            yx_pt = img_arr_idx(WIDTH, i_pt)
            y, x = yx_pt
            P2 = img_dct_idx(WIDTH, y - 1, x)
            P4 = img_dct_idx(WIDTH, y, x + 1)
            P6 = img_dct_idx(WIDTH, y + 1, x)
            P8 = img_dct_idx(WIDTH, y, x - 1)
            black_nbrs = BLACK_POINTS_BLACK_NBRS[i_pt]
            cond_0 = 0 < y < HEIGHT - 1 and 0 < x < WIDTH - 1
            cond_1 = 2 <= len(black_nbrs) <= 6
            cond_2 = white_to_black_transitions(yx_pt, black_nbrs, WIDTH) == 1
            cond_3 = P2 in WHITE_POINTS or P4 in WHITE_POINTS or P8 in WHITE_POINTS
            cond_4 = P2 in WHITE_POINTS or P6 in WHITE_POINTS or P8 in WHITE_POINTS
            if cond_0 and cond_1 and cond_2 and cond_3 and cond_4:
                erase_points = erase_points | {i_pt, P2, P4, P6, P8}
        if erase_points:
            change_points(erase_points, False, img_idx, 0)
            print(f"change length 2: {len(erase_points)}")
            done = False'''

    cv_img = to_cv_img(img_idx)
    cv_img = ~cv_img
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    thin_img = cv2.ximgproc.thinning(cv_img)
    thin_img = ~thin_img
    cv2.imwrite("stupid_thin.jpg", thin_img)
    white_points = set()
    black_points = set()
    HEIGHT, WIDTH = IMG_INFO[img_idx][0]
    for y in range(HEIGHT):
        for x in range(WIDTH):
            val = thin_img[y, x]
            if val == 255:
                white_points.add((y, x))
            elif val == 0:
                black_points.add((y, x))
    restart_img(white_points, black_points, img_idx)


# PRECONDITION: img is black and white.
#
# Thickens img by an amount designatied by thick_factor.
def thicken_img(img_idx, thick_factor):
    info = IMG_INFO[img_idx]
    HEIGHT, WIDTH = info[0]
    BLACK_POINTS = info[3]
    draw_points = set()
    for i_pt in BLACK_POINTS:
        y, x = img_arr_idx(WIDTH, i_pt)
        for i in range(1, thick_factor + 1):
            if y - i >= 0 and y - i < HEIGHT and x >= 0 and x < WIDTH:
                draw_points.add((y - i, x))
            if y >= 0 and y < HEIGHT and x - i >= 0 and x - i < WIDTH:
                draw_points.add((y, x - i))
            if y + i >= 0 and y + i < HEIGHT and x >= 0 and x < WIDTH:
                draw_points.add((y + i, x))
            if y >= 0 and y < HEIGHT and x + i >= 0 and x + i < WIDTH:
                draw_points.add((y, x + 1))
    change_points(draw_points, True, img_idx, 1)

# One-stop-shop for image preparation according to everything required for the analysis to work.
def prep_for_img_analysis(img, img_idx):
    cv2.imwrite("CIM_pre-prep.jpg", img)
    HEIGHT, WIDTH = IMG_INFO[img_idx][0]
    thresh = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    thresh = ~thresh
    points = []

    # The initial edge points (on the borders of img.)
    edge_points = []
    for y in range(HEIGHT):
        for x in range(WIDTH):
            points.append((y, x))
            if y == 0 or y == HEIGHT - 1 or x == 0 or x == WIDTH - 1:
                edge_points.append((y, x))
    get_thresholded_img(img, img_idx, points, edge_points)

    print(done)
    cv2.imwrite("CIM_init_prep_pre-thick.jpg", to_cv_img(img_idx))

    thicken_img(img_idx, 1)

    cv2.imwrite("CIM_init_prep_pre-thin.jpg", to_cv_img(img_idx))

    thin_img(img_idx)

def number_of_objects(img):
    dimensions = img.shape
    height = dimensions[0]
    width = dimensions[1]
    objects = []
    object_point_lists = []
    for y in range(0, height):
        for x in range(0, width):
            if img[y][x] <= 5:
                point = (y, x)
                in_new_object = True
                for list in object_point_lists:
                    if point in list:
                        in_new_object = False
                        break
                if in_new_object:
                    set_of_points = get_set_of_points_from_point(img, point, 0)
                    if len(set_of_points) == 0:
                        empty_points.append(point)
                    object_point_lists.append(set_of_points)
    return len(object_point_lists)

def resize_img(img_path, img_name, output_path, max_dim):
    img = Image.open(img_path + img_name)
    width, height = img.size
    bigger_dimension = width
    if height > width:
        bigger_dimension = height
    factor = bigger_dimension/max_dim
    resized_img = img.resize((math.floor(width/factor), math.floor(height/factor)), Image.ANTIALIAS)
    resized_img.save(output_path + "R" + str(max_dim) + "-" + img_name, optimize = True, quality = 95)
    cv_img = cv2.imread(output_path + "R" + str(max_dim) + "-" + img_name)
    return cv_img

def get_transparent_boundary_points(cv_img, pil_img):
    height, width = cv_img.shape[:2]
    boundary_points = set()
    point_neighbors = {}
    for y in range(height):
        for x in range(width):
            if pil_img.getpixel((x, y))[3] == 0:
                neighbors = set()
                is_boundary = False
                neighbors_to_check = []
                for iter_y in range(y - 1, y + 2):
                    for iter_x in range(x - 1, x + 2):
                        if pil_img.getpixel((iter_x, iter_y))[3] == 0:
                            neighbors_to_check.append((iter_y, iter_x))
                            neighbors.add((iter_y, iter_x))
                        else:
                            is_boundary = True
                if is_boundary:
                    for i in range(3):
                        next_neighbors_to_check = []
                        if neighbors_to_check:
                            for neighbor_y, neighbor_x in neighbors_to_check:
                                possible_next_neighbors = []
                                iter_is_boundary = False
                                for iter_y in range(neighbor_y - 1, neighbor_y + 2):
                                    for iter_x in range(neighbor_x - 1, neighbor_x + 2):
                                        if pil_img.getpixel((iter_x, iter_y))[3] == 0:
                                            possible_next_neighbors.append((iter_y, iter_x))
                                        else:
                                            iter_is_boundary = True
                                if iter_is_boundary:
                                    next_neighbors_to_check += possible_next_neighbors
                            neighbors_to_check = []
                            for neighbor in next_neighbors_to_check:
                                neighbors.add(neighbor)
                                neighbors_to_check.append(neighbor)
                if is_boundary:
                    boundary_points.add((y, x))
                    point_neighbors[(y, x)] = neighbors
    return boundary_points, point_neighbors

def remove_transparent_border(cv_img, pil_img):
    height, width = cv_img.shape[0:2]
    transparent_point_sets = get_sets_of_colored_points(cv_img, 4, pil_image = pil_img)
    transparent_points = []
    background_transparent_points = []
    for point_set in transparent_point_sets:
        transparent_points += point_set
        is_background_set = False
        for point in point_set:
            y, x = point
            if y == 0 or y == height - 1 or x == 0 or x == width - 1:
                is_background_set = True
                break
        if is_background_set:
            background_transparent_points += point_set
    if len(background_transparent_points) == len(transparent_points):
        for y in range(0, height):
            for x in range(0, width):
                if y >= 0 and y < height and x >= 0 and x < width:
                    pixel = pil_img.getpixel((x, y))
                    if pixel[3] != 0:
                        is_transparent_border = False
                        range_length = 4
                        for iter_y in range(y - range_length, y + range_length + 1):
                            for iter_x in range(x - range_length, x + range_length + 1):
                                if not(iter_y == y and iter_y == x) and iter_y >= 0 and iter_y < height and iter_x >= 0 and iter_x < width:
                                    if pil_img.getpixel((iter_x, iter_y))[3] == 0:
                                        is_transparent_border = True
                            if is_transparent_border:
                                break
                        if is_transparent_border:
                            cv_img[y, x] = (255, 255, 255)
    return cv_img
    '''boundary_points, point_neighbors = get_transparent_boundary_points(cv_img, pil_img)
    line_border_points = 0
    for point in point_neighbors:'''


def opaque_img(img_path, img_name, output_path):
    cv_img = cv2.imread(img_path + img_name)
    pil_img = Image.open(img_path + img_name)
    pil_img = pil_img.convert("RGBA")
    height, width = cv_img.shape[0:2]
    transparent_point_sets = get_sets_of_colored_points(cv_img, 4, pil_image = pil_img)
    if len(transparent_point_sets) > 0:
        white_point_sets = get_sets_of_colored_points(cv_img, 3)
        cv_img = remove_transparent_border(cv_img, pil_img)
        for point_set in transparent_point_sets:
            for y, x in point_set:
                cv_img[y, x] = (255, 255, 255)
        new_white_point_sets = get_sets_of_colored_points(cv_img, 3)
        background_white_sets = []
        for point_set in new_white_point_sets:
            point_set = set(point_set)
            is_background_set = False
            for point in point_set:
                y, x = point
                if y == 0 or y == height - 1 or x == 0 or x == width - 1:
                    is_background_set = True
                    break
            if is_background_set:
                background_white_sets.append(point_set)
        for point_set in white_point_sets:
            point_set = set(point_set)
            is_background_set = False
            for other_set in background_white_sets:
                other_set = set(other_set)
                if point_set <= other_set:
                    is_background_set = True
                    break
            if not is_background_set:
                for y, x in point_set:
                    cv_img[y, x] = (0, 0, 0)
    cv2.imwrite(output_path + "opaque_" + img_name, cv_img)
    return cv_img

'''                      ^
    IMAGE PREPARATION    |
                         |   '''

# -------------------------------

'''                      |
    IMAGE ANALYSIS       |
                         v   '''

def calculate_slope(line, approximate):
    return ["UND", 10000][approximate] if line[2] - line[0] == 0 else (line[3] - line[1])/(line[2] - line[0])

def floor_point(point):
    return (math.floor(point[0]), math.floor(point[1]))

def is_near_subset(A, B, closeness_bound, nbrs = -1): # A is near subset of B
    if nbrs != -1:
        '''# A is set of points.
        # B is grid (2D list) of marks if a point is there or not.
        A_set = A[0]
        B_set, B_grid = B[:2]
        if len(A_set) > len(B_set):
            return False
        B_width = len(B_grid[0])
        B_length = len(B_grid)
        for A_y, A_x in A_set:
            if B_grid[A_y][A_x]:
                continue
            for i in range(1, closeness_bound + 1):
                near = False
                min_x = A_x - i
                max_x = A_x + i
                min_y = A_y - i
                max_y = A_y + i
                for x in range(min_x, max_x + 1):
                    if 0 <= x < B_width:
                        if 0 <= min_y < B_length:
                            if B_grid[min_y][x]:
                                near = True
                                break
                        if 0 <= max_y < B_length:
                            if B_grid[max_y][x]:
                                near = True
                                break
                else:
                    for y in range(min_y + 1, max_y):
                        if 0 <= y < B_length:
                            if 0 <= min_x < B_width:
                                if B_grid[y][min_x]:
                                    near = True
                                    break
                            if 0 <= max_x < B_width:
                                if B_grid[y][max_x]:
                                    near = True
                                    break
                    else:
                        if i == closeness_bound:
                            return False
                if near:
                    break
        return True'''

        if len(A) > len(B):
            return False
        for point in A:
            if True not in nbrs[point]:
                return False
        return True
    else:
        if len(A) > len(B):
            return False
        is_near_subset = True
        for A_point in A:
            is_near_in_B = False
            for B_point in B:
                A_y, A_x = A_point
                B_y, B_x = B_point
                if A_point == B_point or (abs(A_y - B_y) <= closeness_bound and abs(A_x - B_x) <= closeness_bound):
                    is_near_in_B = True
                    break
            if not(is_near_in_B):
                is_near_subset = False
                break
        return is_near_subset

def calculate_y_val_on_circle(circle, x, sign):
    a, b, r = circle
    return "none" if (r ** 2) - ((x - a) ** 2) < 0 else sign * math.sqrt((r ** 2) - ((x - a) ** 2)) + b

def remove_near_subset_elements(lst, closeness_bound):
    '''new_list = []
    for item in lst:
        if item not in new_list:
            new_list.append(item)
    lst = new_list
    final_list = []
    list_copy = lst.copy()
    for element in lst:
        other_elements = []
        for other_element in list_copy:
            if other_element != element:
                for point in other_element[1]:
                    if point not in other_elements:
                        other_elements.append(point)
        if not is_near_subset(element[1], other_elements, closeness_bound):
            final_list.append(element)
        else: # MAYBE REMOVE? IDK... IF ISSUES --> CONSIDER
            list_copy.remove(element)
    return final_list'''

    init_list = []
    for elem in lst:
        if elem not in init_list:
            init_list.append(elem)
    seen_idcs = set()
    result_list = []
    for i, elem in enumerate(init_list):
        other_point_sets = [other_elem[1][0] for other_i, other_elem in enumerate(init_list) if other_i not in seen_idcs and other_elem != elem]
        other_points = set().union(*other_point_sets)
        if not is_near_subset(elem[1][0], other_points, closeness_bound, grid = False):
            result_list.append(elem)
        else:
            seen_idcs.add(i)
    return result_list

def hc_accum_array(img_idx, radius_values):
    HEIGHT, WIDTH = IMG_INFO[img_idx][0]
    BLACK_POINTS = IMG_INFO[img_idx][3]
    accum_array = np.zeros((len(radius_values), HEIGHT, WIDTH))
    for i_pt in BLACK_POINTS:
        y, x = img_arr_idx(WIDTH, i_pt)
        for r in range(len(radius_values)):
            rr = radius_values[r]
            hdown = max(0, y - rr)
            for a in range(hdown, y):
                b = round(x + math.sqrt(rr * rr - (a - y) * (a - y)))
                if b >= 0 and b <= WIDTH - 1:
                    accum_array[r][a][b] += 1
                    if 2 * y - a >= 0 and 2 * y - a <= HEIGHT - 1:
                        accum_array[r][2 * y - a][b] += 1
                if 2 * x - b >= 0 and 2 * x - b <= WIDTH - 1:
                    accum_array[r][a][2 * x - b] += 1
                if 2 * y - a >= 0 and 2 * y - a <= HEIGHT - 1 and 2 * x - b >= 0 and 2 * x - b <= WIDTH - 1:
                    accum_array[r][2 * y - a][2 * x - b] += 1
    return accum_array

def find_circles(accum_array, radius_values, hough_thresh):
    returnlist = []
    hlist = []
    wlist = []
    rlist = []
    for r in range(accum_array.shape[0]):
        for h in range(accum_array.shape[1]):
            for w in range(accum_array.shape[2]):
                if accum_array[r][h][w] > hough_thresh:
                    tmp = 0
                    for i in range(len(hlist)):
                        if abs(w-wlist[i])<10 and abs(h-hlist[i])<10:
                            tmp = 1
                            break
                    if tmp == 0:
                        rr = radius_values[r]
                        returnlist.append((w,h,rr))
                        hlist.append(h)
                        wlist.append(w)
                        rlist.append(rr)
    return returnlist

def hough_circles(img_idx):
    radius_values = []
    for radius in range(30):
        radius_values.append(radius)
    accum_array = hc_accum_array(img_idx, radius_values)
    hough_thresh = 30
    result_list = find_circles(accum_array, radius_values, hough_thresh)
    return result_list

def find_curves(img_idx):
    HEIGHT, WIDTH = IMG_INFO[img_idx][0]
    thicken_img(img_idx, 2)
    BLACK_POINTS = IMG_INFO[img_idx][3]
    curve_list = []
    detected_circles = hough_circles(img_idx)
    if detected_circles:
        for circle in detected_circles:
            upper_curve_marks = {}
            lower_curve_marks = {}
            a_int, b_int, r_int = circle
            for x in range(a_int - r_int, a_int + r_int):
                if x >= 0 and x < WIDTH:
                    upper_curve_marks[x] = []
                    y_1 = math.floor(calculate_y_val_on_circle(circle, x, 1))
                    y_2 = calculate_y_val_on_circle(circle, x + 1, 1)
                    y_top = math.floor(y_2)
                    if y_2.is_integer():
                        y_top -= 1
                    if y_top < y_1:
                        temp = y_top
                        y_top = y_1
                        y_1 = temp
                    for y in range(y_1, y_top + 1):
                        i_pt = img_dct_idx(WIDTH, y, x)
                        if i_pt in BLACK_POINTS:
                            upper_curve_marks[x].append(y)
                    lower_curve_marks[x] = []
                    second_y_1 = math.floor(calculate_y_val_on_circle(circle, x, -1))
                    second_y_2 = calculate_y_val_on_circle(circle, x + 1, -1)
                    y_bottom = math.floor(second_y_2)
                    if second_y_2.is_integer():
                        y_bottom -= 1
                    if y_bottom < second_y_1:
                        temp = y_bottom
                        y_bottom = second_y_1
                        second_y_1 = temp
                    for y in range(second_y_1, y_bottom + 1):
                        i_pt = img_dct_idx(WIDTH, y, x)
                        if i_pt in BLACK_POINTS:
                            lower_curve_marks[x].append(y)
            y_bound = 1
            last_y = -1
            curve_between_halves = [set(), [[False for _ in range(WIDTH)] for _ in range(HEIGHT)], []]
            front_curve = []
            x = a_int - r_int
            while x in range(a_int - r_int, a_int + r_int) and x >= 0 and x < WIDTH:
                if len(upper_curve_marks[x]) != 0:
                    current_curve_marks = upper_curve_marks[x]
                    last_y = current_curve_marks[0]
                    this_curve_list = []
                    this_curve_set = set()
                    this_curve_grid = [[False for _ in range(WIDTH)] for _ in range(HEIGHT)]
                    near_y = True
                    while x in upper_curve_marks and len(current_curve_marks) > 0 and near_y:
                        current_curve_marks.sort()
                        if x >= a_int:
                            current_curve_marks.reverse()
                        for y in current_curve_marks:
                            if abs(y - last_y) > y_bound:
                                near_y = False
                                break
                            this_curve_list.append((y, x))
                            this_curve_set.add((y, x))
                            this_curve_grid[y][x] = True
                            last_y = y
                        x += 1
                        if x in upper_curve_marks:
                            current_curve_marks = upper_curve_marks[x]
                    this_curve = (this_curve_set, this_curve_grid, this_curve_list)
                    if front_curve == []:
                        front_curve = this_curve
                    elif x in upper_curve_marks:
                        if len(this_curve_set) >= 3:
                            is_new_curve = True
                            new_curve_list = []
                            for curve in curve_list:
                                if is_near_subset(curve[1], this_curve, 4):
                                    continue
                                elif is_near_subset(this_curve, curve[1], 4):
                                    is_new_curve = False
                                new_curve_list.append(curve)
                            if is_new_curve:
                                new_curve_list.append(((a_int, b_int, r_int), this_curve))
                            curve_list = new_curve_list
                    if x >= a_int + r_int - 1:
                        curve_between_halves = this_curve
                x += 1
                if front_curve == []:
                    front_curve = -1
            x = a_int + r_int - 1
            while x >= a_int - r_int and x >= 0 and x < WIDTH:
                if len(lower_curve_marks[x]) != 0:
                    current_curve_marks = lower_curve_marks[x]
                    last_y = current_curve_marks[0]
                    if x == a_int + r_int - 1 and curve_between_halves[2] != []:
                        last_y = curve_between_halves[2][len(curve_between_halves[2]) - 1][0]
                    this_curve = curve_between_halves
                    this_curve_set, this_curve_grid, this_curve_list = this_curve
                    curve_between_halves = [set(), [[False for _ in range(WIDTH)] for _ in range(HEIGHT)], []]
                    near_y = True
                    while x in lower_curve_marks and len(current_curve_marks) > 0 and near_y:
                        current_curve_marks.sort()
                        if x >= a_int:
                            current_curve_marks.reverse()
                        for y in current_curve_marks:
                            if abs(y - last_y) > y_bound:
                                near_y = False
                                break
                            this_curve_list.append((y, x))
                            this_curve_set.add((y, x))
                            this_curve_grid[y][x] = True
                            last_y = y
                        x -= 1
                        if x in lower_curve_marks:
                            current_curve_marks = lower_curve_marks[x]
                    if x not in lower_curve_marks and front_curve != -1:
                        this_curve += front_curve
                    if len(this_curve_set) >= 3:
                        is_new_curve = True
                        new_curve_list = []
                        for curve in curve_list:
                            if is_near_subset(curve[1], this_curve, 4):
                                continue
                            elif is_near_subset(this_curve, curve[1], 4):
                                is_new_curve = False
                            new_curve_list.append(curve)
                        if is_new_curve:
                            new_curve_list.append(((a_int, b_int, r_int), this_curve))
                        curve_list = new_curve_list
                x -= 1
    final_list = remove_near_subset_elements(curve_list, 4)
    print("done with FC")
    return final_list

def hough_lines(img_idx, num_rhos = 180, num_thetas = 180, t_count = 3):
    HEIGHT, WIDTH = IMG_INFO[img_idx][0]
    height_half, width_half = HEIGHT / 2, WIDTH / 2
    d = np.sqrt(np.square(HEIGHT) + np.square(WIDTH))
    dtheta = 180 / num_thetas
    drho = (2 * d) / num_rhos
    thetas = np.arange(0, 180, step = dtheta)
    rhos = np.arange(-d, d, step = drho)
    cos_thetas = np.cos(np.deg2rad(thetas))
    sin_thetas = np.sin(np.deg2rad(thetas))
    accumulator = np.zeros((len(rhos), len(rhos)))
    BLACK_POINTS = IMG_INFO[img_idx][3]
    for i_pt in BLACK_POINTS:
        y, x = img_arr_idx(WIDTH, i_pt)
        edge_point = [y - height_half, x - width_half]
        ys, xs = [], []
        for theta_idx in range(len(thetas)):
            rho = (edge_point[1] * cos_thetas[theta_idx]) + (edge_point[0] * sin_thetas[theta_idx])
            theta = thetas[theta_idx]
            rho_idx = np.argmin(np.abs(rhos - rho))
            accumulator[rho_idx][theta_idx] += 1
            ys.append(rho)
            xs.append(theta)
    lines = set()
    for y in range(accumulator.shape[0]):
        for x in range(accumulator.shape[1]):
            if accumulator[y][x] > t_count:
                rho = rhos[y]
                theta = thetas[x]
                a = np.cos(np.deg2rad(theta))
                b = np.sin(np.deg2rad(theta))
                x0 = (a * rho) + width_half
                y0 = (b * rho) + height_half
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                line = (x1, y1, x2, y2)
                lines.add(line)
    return lines

def find_edges(img_idx, intersection_mode, img_objects = -1):
    '''global ins_time
    ins_time = 0
    thickness_factor = 2 if intersection_mode else 4
    thicken_img(img_idx, thickness_factor)
    if img_objects != -1:
        for object in img_objects:
            new_object = set()
            for y, x in object:
                for iter_y in range(y - 1, y + 2):
                    for iter_x in range(x - 1, x + 2):
                        if 0 <= iter_y < height and 0 <= iter_x < width:
                            new_object.add((iter_y, iter_x))
            img_objects.remove(object)
            img_objects.append(new_object)
    BLACK_POINTS = IMG_INFO[img_idx][3]
    HEIGHT, WIDTH = IMG_INFO[img_idx][0]
    lines = hough_lines(img_idx)
    #print(f"No. of Lines: {len(lines)}")
    edge_list = []
    line_ct = len(lines)
    ct = 0
    print(len(lines))
    for line in lines:
        ct += 1
        print("Position: " + str(line_ct - ct))
        m = calculate_slope(line, False)
        if m != "UND":
            edge_marks = {}
            for x in range(0, WIDTH):
                edge_marks[x] = []
                y_1 = math.floor(calculate_y_val(line, x))
                y_2 = calculate_y_val(line, x + 1)
                y_top = math.floor(y_2)
                if y_2.is_integer():
                    y_top -= 1
                if y_top < y_1:
                    temp = y_top
                    y_top = y_1
                    y_1 = temp
                for y in range(y_1, y_top + 1):
                    i_pt = img_dct_idx(WIDTH, y, x)
                    if i_pt in BLACK_POINTS:
                        edge_marks[x].append(y)
            last_y = -1
            x = 0
            while x in range(0, WIDTH):
                if len(edge_marks[x]) != 0:
                    current_edge_marks = edge_marks[x]
                    if m < 0:
                        current_edge_marks.reverse()
                    last_y = current_edge_marks[0]
                    y_bound = 1
                    this_edge_set = set()
                    near_y = True
                    while x in edge_marks and len(current_edge_marks) > 0 and near_y:
                        current_edge_marks = edge_marks[x]
                        if m < 0:
                            current_edge_marks.reverse()
                        for y in current_edge_marks:
                            if abs(y - last_y) > y_bound:
                                near_y = False
                                break
                            this_edge_set.add((y, x))
                            last_y = y
                        x += 1
                    if len(this_edge_set) >= 3:
                        this_edge_grid = [[(j, i) in this_edge_set for i in range(WIDTH)] for j in range(HEIGHT)]
                        this_edge = (this_edge_set, this_edge_grid)
                        is_new_edge = True
                        new_edge_list = []
                        for edge in edge_list:
                            if is_near_subset(edge[1], this_edge, 2*thickness_factor):
                                continue
                            elif is_near_subset(this_edge, edge[1], 2*thickness_factor):
                                is_new_edge = False
                            new_edge_list.append(edge)
                        if is_new_edge:
                            if img_objects != -1:
                                spl = sample(this_edge_set, math.ceil(len(this_edge_set)/10))
                                object_point_counts = {}
                                for object in img_objects:
                                    object = tuple(object)
                                    object_point_counts[object] = 0
                                for point in spl:
                                    for object in object_point_counts:
                                        object = tuple(object)
                                        if point in object:
                                            object_point_counts[object] += 1
                                max_count = 0
                                for object in object_point_counts:
                                    object = tuple(object)
                                    if (count := object_point_counts[object]) > max_count:
                                        max_count = count
                                edge_object = -1
                                for object in object_point_counts:
                                    object = tuple(object)
                                    if object_point_counts[object] == max_count:
                                        edge_object = object
                                new_edge_list.append((line, this_edge, edge_object))
                            else:
                                new_edge_list.append((line, this_edge))
                        edge_list = new_edge_list
                x += 1
        else:
            edge_marks = {}
            line_x_val = math.floor(line[0])
            if line_x_val >= 0 and line_x_val < WIDTH:
                for y in range(0, HEIGHT):
                    i_pt = img_dct_idx(img_idx, y, line_x_val)
                    if i_pt in BLACK_POINTS:
                        edge_marks[y] = True
                    else:
                        edge_marks[y] = False
                y = 0
                while y in range(0, HEIGHT):
                    if edge_marks[y] == True:
                        this_edge_set = set()
                        this_edge_grid = [[False for _ in range(WIDTH)] for _ in range(HEIGHT)]
                        current_edge_mark = edge_marks[y]
                        while y in edge_marks and current_edge_mark == True:
                            current_edge_mark = edge_marks[y]
                            this_edge_set.add((y, line_x_val))
                            this_edge_grid[y][line_x_val] = True
                            y += 1
                        this_edge = (this_edge_set, this_edge_grid)
                        if len(this_edge_set) >= 3:
                            is_new_edge = True
                            new_edge_list = []
                            for edge in edge_list:
                                if is_near_subset(edge[1], this_edge, 2*thickness_factor):
                                    continue
                                elif is_near_subset(this_edge, edge[1], 2*thickness_factor):
                                    is_new_edge = False
                                new_edge_list.append(edge)
                            if is_new_edge:
                                if img_objects != -1:
                                    spl = sample(this_edge_set, math.ceil(len(this_edge_set)/10))
                                    object_point_counts = {}
                                    for object in img_objects:
                                        object = tuple(object)
                                        object_point_counts[object] = 0
                                    for point in spl:
                                        for object in object_point_counts:
                                            object = tuple(object)
                                            if point in object:
                                                object_point_counts[object] += 1
                                    max_count = 0
                                    for object in object_point_counts:
                                        object = tuple(object)
                                        if (count := object_point_counts[object]) > max_count:
                                            max_count = count
                                    edge_object = -1
                                    for object in object_point_counts:
                                        object = tuple(object)
                                        if object_point_counts[object] == max_count:
                                            edge_object = object
                                    new_edge_list.append((line, this_edge, edge_object))
                                else:
                                    new_edge_list.append((line, this_edge))
                            edge_list = new_edge_list
                    y += 1
    print("check")
    print(f"INS TIME: {ins_time}s")
    final_remove_factor = 1 if intersection_mode else 2
    print(f"RAW EDGE NUM: {len(edge_list)}")
    while len(edge_list) > 60:
        edge_list.sort(key = lambda edge: len(edge[1]))
        edge_list = edge_list[int(len(edge_list) * (3/4)):]
        print(len(edge_list))
    print(f"PRE REMOVAL EDGE NUM: {len(edge_list)}")
    final_list = remove_near_subset_elements(edge_list, thickness_factor * final_remove_factor)
    print(f"No. of Edges: {len(final_list)}")
    return final_list'''
    #h, w = IMG_INFO[img_idx][0]#####
    #bp = IMG_INFO[img_idx][3]

    thickness_factor = 2 if intersection_mode else 4
    '''cv_img = np.zeros((h, w, 3), np.uint8)
    cv_img = ~cv_img
    for i_pt in bp:
        y,x = img_arr_idx(w, i_pt)
        cv_img[y, x] = (0,0,0)
    cv2.imwrite("CIM_first_FE_pre-thick.jpg", cv_img)'''

    thicken_img(img_idx, thickness_factor)

    '''bp = IMG_INFO[img_idx][3]
    cv_img = np.zeros((h, w, 3), np.uint8)
    cv_img = ~cv_img
    for i_pt in bp:
        y,x = img_arr_idx(w, i_pt)
        cv_img[y, x] = (0,0,0)
    cv2.imwrite("CIM_first_FE_post-thick.jpg", cv_img)'''

    if img_objects != -1:
        for object in img_objects:
            new_object = set()
            for y, x in object:
                for iter_y in range(y - 1, y + 2):
                    for iter_x in range(x - 1, x + 2):
                        if 0 <= iter_y < height and 0 <= iter_x < width:
                            new_object.add((iter_y, iter_x))
            img_objects.remove(object)
            img_objects.append(new_object)
    BLACK_POINTS = IMG_INFO[img_idx][3]
    HEIGHT, WIDTH = IMG_INFO[img_idx][0]
    lines = hough_lines(img_idx)
    #print(f"No. of Lines: {len(lines)}")
    edge_list = []
    line_ct = len(lines)
    ct = 0
    print(line_ct)
    print("done")
    print(stop)
    closeness_bound = 2 * thickness_factor
    edge_grid = [[False for i in range(WIDTH)] for j in range(HEIGHT)]
    point_nbrs = {(y, x) : {edge_grid[j][i] for j in range(y - closeness_bound, y + closeness_bound + 1) for i in range(x - closeness_bound, x + closeness_bound + 1) if 0 <= j < HEIGHT and 0 <= i < WIDTH} for y in range(HEIGHT) for x in range(WIDTH)}
    for line in lines:
        ct += 1
        print("Position: " + str(line_ct - ct))
        m = calculate_slope(line, False)
        if m != "UND":
            edge_marks = {}
            for x in range(0, WIDTH):
                edge_marks[x] = []
                y_1 = math.floor(calculate_y_val(line, x))
                y_2 = calculate_y_val(line, x + 1)
                y_top = math.floor(y_2)
                if y_2.is_integer():
                    y_top -= 1
                if y_top < y_1:
                    temp = y_top
                    y_top = y_1
                    y_1 = temp
                for y in range(y_1, y_top + 1):
                    i_pt = img_dct_idx(WIDTH, y, x)
                    if i_pt in BLACK_POINTS:
                        edge_marks[x].append(y)
            last_y = -1
            x = 0
            while x in range(0, WIDTH):
                if len(edge_marks[x]) != 0:
                    current_edge_marks = edge_marks[x]
                    if m < 0:
                        current_edge_marks.reverse()
                    last_y = current_edge_marks[0]
                    y_bound = 1
                    this_edge = set()
                    near_y = True
                    while x in edge_marks and len(current_edge_marks) > 0 and near_y:
                        current_edge_marks = edge_marks[x]
                        if m < 0:
                            current_edge_marks.reverse()
                        for y in current_edge_marks:
                            if abs(y - last_y) > y_bound:
                                near_y = False
                                break
                            this_edge.add((y, x))
                            last_y = y
                        x += 1
                    if len(this_edge) >= 3:
                        edge_grid = [[(j, i) in this_edge for i in range(WIDTH)] for j in range(HEIGHT)]
                        is_new_edge = True
                        new_edge_list = []
                        for edge in edge_list:
                            if is_near_subset(edge[1], this_edge, closeness_bound, nbrs = point_nbrs):
                                continue
                            elif is_near_subset(this_edge, edge[1], closeness_bound): # Shiiiiiit; I have to account for this too... point_nbrs is only designed for the first call. (obv. I can just do no nbrs).
                                is_new_edge = False
                            new_edge_list.append(edge)
                        if ct % 500 == 0:
                            print(len(edge_list))
                        if is_new_edge:
                            if img_objects != -1:
                                spl = sample(this_edge, math.ceil(len(this_edge)/10))
                                object_point_counts = {}
                                for object in img_objects:
                                    object = tuple(object)
                                    object_point_counts[object] = 0
                                for point in spl:
                                    for object in object_point_counts:
                                        object = tuple(object)
                                        if point in object:
                                            object_point_counts[object] += 1
                                max_count = 0
                                for object in object_point_counts:
                                    object = tuple(object)
                                    if (count := object_point_counts[object]) > max_count:
                                        max_count = count
                                edge_object = -1
                                for object in object_point_counts:
                                    object = tuple(object)
                                    if object_point_counts[object] == max_count:
                                        edge_object = object
                                new_edge_list.append((line, this_edge, edge_object))
                            else:
                                new_edge_list.append((line, this_edge))
                        edge_list = new_edge_list
                x += 1
        else:
            edge_marks = {}
            line_x_val = math.floor(line[0])
            if line_x_val >= 0 and line_x_val < WIDTH:
                for y in range(0, HEIGHT):
                    i_pt = img_dct_idx(img_idx, y, line_x_val)
                    if i_pt in BLACK_POINTS:
                        edge_marks[y] = True
                    else:
                        edge_marks[y] = False
                y = 0
                while y in range(0, HEIGHT):
                    if edge_marks[y] == True:
                        this_edge = set()
                        current_edge_mark = edge_marks[y]
                        while y in edge_marks and current_edge_mark == True:
                            current_edge_mark = edge_marks[y]
                            this_edge.add((y, line_x_val))
                            y += 1
                        if len(this_edge) >= 3:
                            edge_grid = [[(j, i) in this_edge for i in range(WIDTH)] for j in range(HEIGHT)]
                            is_new_edge = True
                            new_edge_list = []
                            for edge in edge_list:
                                if is_near_subset(edge[1], this_edge, closeness_bound, nbrs = point_nbrs):
                                    continue
                                elif is_near_subset(this_edge, edge[1], closeness_bound):
                                    is_new_edge = False
                                new_edge_list.append(edge)
                            if is_new_edge:
                                if img_objects != -1:
                                    spl = sample(this_edge, math.ceil(len(this_edge)/10))
                                    object_point_counts = {}
                                    for object in img_objects:
                                        object = tuple(object)
                                        object_point_counts[object] = 0
                                    for point in spl:
                                        for object in object_point_counts:
                                            object = tuple(object)
                                            if point in object:
                                                object_point_counts[object] += 1
                                    max_count = 0
                                    for object in object_point_counts:
                                        object = tuple(object)
                                        if (count := object_point_counts[object]) > max_count:
                                            max_count = count
                                    edge_object = -1
                                    for object in object_point_counts:
                                        object = tuple(object)
                                        if object_point_counts[object] == max_count:
                                            edge_object = object
                                    new_edge_list.append((line, this_edge, edge_object))
                                else:
                                    new_edge_list.append((line, this_edge))
                            edge_list = new_edge_list
                    y += 1
    print("check")
    final_remove_factor = 1 if intersection_mode else 2
    print(f"RAW EDGE NUM: {len(edge_list)}")
    while len(edge_list) > 60:
        edge_list.sort(key = lambda edge: len(edge[1]))
        edge_list = edge_list[int(len(edge_list) * (3/4)):]
        print(len(edge_list))
    print(f"PRE REMOVAL EDGE NUM: {len(edge_list)}")
    final_list = remove_near_subset_elements(edge_list, thickness_factor * final_remove_factor)
    print(f"No. of Edges: {len(final_list)}")
    return final_list

def calculate_y_val(line, x_val):
    x1, y1, x2, y2 = line
    return "none" if x2 - x1 == 0 else ((y2 - y1)/(x2 - x1)) * (x_val - x1) + y1

def calculate_x_val(line, y_val):# *
    x1, y1, x2, y2 = line
    return "none" if x2 - x1 == 0 else (y_val - y1)/((y2 - y1)/(x2 - x1)) + x1

def solve_quadratic(a, b, c):
    return "none" if  a == 0 or (b ** 2) - (4 * a * c) < 0 else ( (-b + math.sqrt((b ** 2) - (4 * a * c)))/(2 * a) , (-b - math.sqrt((b ** 2) - (4 * a * c)))/(2 * a) )

def calculate_intersection(structure1, structure2):
    len1 = len(structure1)
    len2 = len(structure2)
    if len1 == len2 == 3:
        circle1 = structure1
        circle2 = structure2
        if circle1 == circle2:
            return "none"
        else:
            h1, k1, r1 = circle1
            h2, k2, r2 = circle2
            line = x1 = y1 = x2 = y2 = -1
            if k1 == k2:
                front_circle = circle1
                behind_circle = circle2
                if h1 < h2:
                    front_circle = circle2
                    behind_circle = circle1
                first_x = behind_circle[0] + behind_circle[2]
                second_x = front_circle[0] - front_circle[2]
                x1 = x2 = (first_x + second_x)/2
                y1 = 0
                y2 = 1
            else:
                m = (-2 * h2 + 2 * h1)/(-2 * k1 + 2 * k2)
                b = (h2 ** 2 - (h1 ** 2) + r1 ** 2 - (r2 ** 2) - (k1 ** 2) + k2 ** 2)/(-2 * k1 + 2 * k2)
                x1 = 0
                y1 = b
                x2 = 1
                y2 = m + b
            line = (x1, y1, x2, y2)
            return calculate_intersection(line, circle1)
    elif len1 == len2 == 4:
        line1 = structure1
        line2 = structure2
        l1x1, l1y1, l1x2, l1y2 = line1
        l2x1, l2y1, l2x2, l2y2 = line2
        m1 = m2 = 0
        if l1x2 - l1x1 == 0:
            m1 = "UND"
        else:
            m1 = (l1y2 - l1y1)/(l1x2 - l1x1)
        if l2x2 - l2x1 == 0:
            m2 = "UND"
        else:
            m2 = (l2y2 - l2y1)/(l2x2 - l2x1)
        if not(m2 == "UND" and m1 == "UND"):
            if m2 == "UND" or m1 == "UND":
                x = y = 0
                if m1 == "UND":
                    x = l1x1
                    y = m2 * (x - l2x1) + l2y1
                elif m2 == "UND":
                    x = l2x1
                    y = m1 * (x - l1x1) + l1y1
                else:
                    x = (m2 * l2x1 - (m1 * l1x1) - l2y1 + l1y1)/(m2 - m1)
                    y = m2 * (x - l2x1) + l2y1
                return [(y, x)] # USED TO BE (x, y) SO FIX ALL OLD USAGES OF THIS FXN !!!!!!!!!!!!!!!!
            if m2 - m1 != 0:
                x = y = 0
                if m1 == "UND":
                    x = l1x1
                    y = m2 * (x - l2x1) + l2y1
                elif m2 == "UND":
                    x = l2x1
                    y = m1 * (x - l1x1) + l1y1
                else:
                    x = (m2 * l2x1 - (m1 * l1x1) - l2y1 + l1y1)/(m2 - m1)
                    y = m2 * (x - l2x1) + l2y1
                return [(y, x)]
            else:
                return "none"
        else:
            return "none"
    elif len1 == 3 and len2 == 4 or len1 == 4 and len2 == 3:
        circle = line = -1
        if len1 < len2:
            circle = structure1
            line = structure2
        else:
            circle = structure2
            line = structure1
        h, k, r = circle
        x1, y1, x2, y2 = line
        m = calculate_slope(line, False)
        if m != "UND":
            q = calculate_intersection(line, (0, 0, 0, 1))[0][0]
            a = -(m ** 2) - 1
            b = 2 * h + 2 * (k - q) * m
            c = -(-(r ** 2) + h ** 2 + (k - q) ** 2)
            x_vals = solve_quadratic(a, b, c)
            if x_vals == "none":
                return "none"
            first_x, second_x = x_vals
            first_y = calculate_y_val(line, first_x)
            second_y = calculate_y_val(line, second_x)
            first_point = (first_y, first_x)
            second_point = (second_y, second_x)
            return [first_point, second_point]
        else:
            x = x1
            first_y = calculate_y_val_on_circle(circle, x, 1)
            second_y = calculate_y_val_on_circle(circle, x, -1)
            if first_y == "none":
                return "none"
            first_point = (first_y, x)
            second_point = (second_y, x)
            return [first_point, second_point]

def find_intersections(img_idx):
    HEIGHT, WIDTH = IMG_INFO[img_idx][0]
    #curves = find_curves(img_idx)
    print("START FIND EDGES")
    edges = find_edges(img_idx, True)
    structure_list = curves + edges
    final_structures = remove_near_subset_elements(structure_list, 2)
    models = set()
    for structure in final_structures:
        models.add(structure[0])
    intersecting_pairs = []
    for model in models:
        for other_model in models:
            if model != other_model:
                intersections = calculate_intersection(model, other_model)
                if intersections != "none":
                    intersecting_pair = (model, other_model)
                    reverse_intersecting_pair = (other_model, model)
                    if intersecting_pair not in intersecting_pairs and reverse_intersecting_pair not in intersecting_pairs:
                        intersecting_pairs.append(intersecting_pair)
    final_intersections = set()
    for intersecting_pair in intersecting_pairs:
        model1, model2 = intersecting_pair
        intersections = calculate_intersection(model1, model2)
        for intersection in intersections:
            rounded_intersection = floor_point(intersection)
            rounded_y = rounded_intersection[0]
            rounded_x = rounded_intersection[1]
            potential_intersections = {}
            range_length = 5
            for y in range(rounded_y - range_length, rounded_y + range_length + 1):
                for x in range(rounded_x - 5, rounded_x + 6):
                    if y >= 0 and y < height and x >= 0 and x < width:
                        if img[y][x] <= 5:
                            is_new_intersection = True
                            for point in final_intersections:
                                check_y = point[0]
                                check_x = point[1]
                                closeness_bound = 1
                                if abs(y - check_y) <= closeness_bound and abs(x - check_x) <= closeness_bound:
                                    is_new_intersection = False
                                    break
                            if is_new_intersection:
                                potential_intersections[(y, x)] = abs(y - rounded_y) + abs(x - rounded_x)
            min_distance = 2 * range_length
            for point in potential_intersections:
                distance = potential_intersections[point]
                if distance < min_distance:
                    min_distance = distance
            for point in potential_intersections:
                distance = potential_intersections[point]
                if distance == min_distance:
                    final_intersections.add(point)
                    break
    final_intersections = [img_dct_idx(img_idx, y, x) for y, x in final_intersections]
    return final_intersections

def find_open_points(img_idx):
    BLACK_POINTS_BLACK_NBRS = IMG_INFO[img_idx][7]
    for i_pt in BLACK_POINTS_BLACK_NBRS:
        if len(BLACK_POINTS_BLACK_NBRS[i_pt]) == 1:
            return i_pt
    return -1

def propagation_helper(img_idx, i_pt_start):
    idx_list = [i_pt_start]
    start_tree = Node(i_pt_start)
    tree_stack = [start_tree]
    while tree_stack:
        current_tree = tree_stack.pop(len(tree_stack) - 1)
        current_idx = current_tree.name
        next_segments = get_neighbor_segments(img_idx, current_idx)
        next_idxs = []
        for segment in next_segments:
            for i_pt in segment:
                if i_pt not in idx_list:
                    idx_list.append(next_idx)
                    next_idxs.append(next_idx)
                    break
        for i_pt in next_idxs:
            next_tree = Node(point, parent = current_tree)
            tree_stack.append(next_tree)
    return (start_tree, idx_list)

def get_ordered_point_tree(img_idx, intersections, open_point): # assume given img has 1 obj
    start_point = -1
    if intersections:
        if open_point != -1:
            BLACK_POINTS = IMG_INFO[img_idx][3]
            start_point = BLACK_POINTS[0]
        else:
            start_point = open_point
    else:
        start_point = intersections[0]
    return propagation_helper(img_idx, start_point)

def get_set_of_points_from_point_binary(img_idx, i_pt_start, value):
    i_pt_set = set()
    i_pt_queue = [i_pt_start]
    NEIGHBORS = IMG_INFO[img_idx][4] if value == False else IMG_INFO[img_idx][7]
    for i_pt in i_pt_queue:
        if i_pt in NEIGHBORS:
            nbrs = NEIGHBORS[i_pt]
            for nbr in nbrs:
                if nbr not in i_pt_set:
                    i_pt_queue.append(nbr)
                    i_pt_set.add(nbr)
    return i_pt_set

def get_set_of_points_from_point(img, start_point, color, pil_img = -1): # 0 = black ; 1 = white ; 2 = nonwhite ; 3 = pseudowhite ; 4 = transparent # 5 = gradient
    dimensions = img.shape
    height = dimensions[0]
    width = dimensions[1]
    final_point_set = {start_point}
    point_queue = [start_point]
    while len(point_queue) > 0:
        current_point = point_queue.pop(0)
        surrounding_points = []
        up = (current_point[0] + 1, current_point[1])
        right = (current_point[0], current_point[1] + 1)
        down = (current_point[0] - 1, current_point[1])
        left = (current_point[0], current_point[1] - 1)
        adjacent_points  = (up, right, down, left)
        for point in adjacent_points:
            y, x = point
            if (y >= 0 and y < height) and (x >= 0 and x < width):
                if color == 0:
                    if img[y][x] <= 5 and (y, x) not in final_point_set:
                        surrounding_points.append(point)
                elif color == 1:
                    if img[y][x] >= 250 and (y, x) not in final_point_set:
                        surrounding_points.append(point)
                elif color in (2, 3):
                    is_white = True
                    for val in img[y, x]:
                        if val < 248:
                            is_white = False
                    if color == 2 and not(is_white) and (y, x) not in final_point_set:
                        surrounding_points.append(point)
                    elif color == 3 and is_white and (y, x) not in final_point_set:
                        surrounding_points.append(point)
                elif color == 4:
                    if pil_img.getpixel((x, y))[3] == 0 and (y, x) not in final_point_set:
                        surrounding_points.append(point)
                elif color == 5:# RGB
                    cur_y, cur_x = current_point
                    cur_color = img[cur_y, cur_x]
                    iter_color = img[y, x]
                    is_near_color = True
                    for val in range(3):
                        if abs(cur_color[val] - iter_color[val]) > 8:
                            is_near_color = False
                    if is_near_color and (y, x) not in final_point_set:
                         surrounding_points.append(point)
        top_right = (current_point[0] + 1, current_point[1] + 1)
        bottom_right = (current_point[0] - 1, current_point[1] + 1)
        bottom_left = (current_point[0] - 1, current_point[1] - 1)
        top_left = (current_point[0] + 1, current_point[1] - 1)
        diagonal_points = (top_right, bottom_right, bottom_left, top_left)
        for point in diagonal_points:
            y, x = point
            if (y >= 0 and y < height) and (x >= 0 and x < width):
                if color == 0:
                    if img[y][x] <= 5 and (y, x) not in final_point_set:
                        surrounding_points.append(point)
                elif color == 1:
                    if img[y][x] >= 250 and (y, x) not in final_point_set:
                        surrounding_points.append(point)
                elif color in (2, 3):
                    is_white = True
                    for val in img[y, x]:
                        if val < 248:
                            is_white = False
                    if color == 2 and not(is_white) and (y, x) not in final_point_set:
                        surrounding_points.append(point)
                    elif color == 3 and is_white and (y, x) not in final_point_set:
                        surrounding_points.append(point)
                elif color == 4:
                    if pil_img.getpixel((x, y))[3] == 0 and (y, x) not in final_point_set:
                        surrounding_points.append(point)
                elif color == 5:# RGB
                    cur_y, cur_x = current_point
                    cur_color = img[cur_y, cur_x]
                    iter_color = img[y, x]
                    is_near_color = True
                    for val in range(3):
                        if abs(cur_color[val] - iter_color[val]) > 8:
                            is_near_color = False
                    if is_near_color and (y, x) not in final_point_set:
                         surrounding_points.append(point)
        for point in surrounding_points:
            final_point_set.add(point)
            point_queue.append(point)
    return final_point_set

def get_separate_objects(img_idx):
    object_point_lists = []
    BLACK_POINTS = IMG_INFO[img_idx][3]
    for i_pt in BLACK_POINTS:
        in_new_object = True
        for object in object_point_lists:
            if i_pt in object:
                in_new_object = False
                break
        if in_new_object:
            object_point_list = get_set_of_points_from_point_binary(img_idx, i_pt, True)
            object_point_lists.append(object_point_list)
    objects = []
    DIMENSIONS = IMG_INFO[img_idx][0]
    for object_point_list in object_point_lists:
        intersections = ordered_list_of_points = False
        object_img_idx = initialize_img(DIMENSIONS)
        change_points(object_point_list, True, object_img_idx, 0)
        intersections = find_intersections(object_img_idx)
        open_point = find_open_points(object_img_idx)
        (ordered_point_tree, point_list) = get_ordered_point_tree(object_img_idx, intersections, open_point)
        object = (intersections, ordered_point_tree, point_list)
        objects.append(object)
    return objects

def get_branches_of_points_from_tree(tree):
    ordered_lists_of_points = []
    ordered_list_of_points = []
    sets_of_points = []
    set_of_points = set()
    tree_parser = tree
    while len(tree_parser.children) == 1:
        point = tree_parser.name
        ordered_list_of_points.append(point)
        set_of_points.add(point)
        tree_parser = tree_parser.children[0]
    point = tree_parser.name
    ordered_list_of_points.append(point)
    set_of_points.add(point)
    ordered_lists_of_points.append(ordered_list_of_points)
    sets_of_points.append(set_of_points)
    for subtree in tree_parser.children:
        sub_ordered_lists_of_points, sub_sets_of_points = get_branches_of_points_from_tree(subtree)
        ordered_lists_of_points += sub_ordered_lists_of_points
        sets_of_points += sub_sets_of_points
    return ordered_lists_of_points, sets_of_points

def linear_approximation(img_idx):
    objects = get_separate_objects(img_idx)
    DIMENSIONS = IMG_INFO[img_idx][0]
    lin_app_img_idx = initialize_img(DIMENSIONS)
    HEIGHT, WIDTH = DIMENSIONS
    drawn_objects = []
    for object in objects:
        intersections, ordered_point_tree, point_list = object
        ordered_lists_of_points, sets_of_points = get_branches_of_points_from_tree(ordered_point_tree)
        new_intersections = []
        for intersection in intersections:
            y, x = img_arr_idx(WIDTH, intersection)
            surrounding_points = [img_dct_idx(y + iy, x + ix) for iy in range(-1, 2) for ix in range(-1, 2) if (iter_y != 0 or iter_x != 0) and 0 <= y < HEIGHT and 0 <= x < WIDTH]
            current_new_intersections = [intersection]
            for point in surrounding_points:
                is_new_intersection = False
                for set_of_points in sets_of_points:
                    if point in set_of_points:
                        is_alone = True
                        for current_new_intersection in current_new_intersections:
                            if current_new_intersection in set_of_points:
                                is_alone = False
                                break
                        if is_alone:
                            is_new_intersection = True
                        break
                if is_new_intersection:
                    current_new_intersections.append(point)
            new_intersections += current_new_intersections
        point_num = sum(len(set_of_points) for set_of_points in sets_of_points)
        segment_length = 10
        possible_length = math.floor(point_num/8)
        if possible_length > segment_length:
            segment_length = possible_length
        draw_points = linear_approximation_helper(new_intersections, ordered_lists_of_points, 0, segment_length, set())
        drawn_objects.append(draw_points)
        change_points(draw_points, True, lin_app_img_idx, 1)
    return lin_app_img_idx, drawn_objects

def linear_approximation_helper(intersections, ordered_lists_of_points, branch_idx, segment_length, draw_points): # Branch index scheme might not work
    ordered_list_of_points = ordered_lists_of_points[branch_idx]
    branch_idx += 1
    focus_points_indices = []
    focus_points_indices.append(len(ordered_list_of_points) - 1)
    for intersection in intersections:
        for index in range(len(ordered_list_of_points)):
            if ordered_list_of_points[index] == intersection:
                focus_points_indices.append(len(ordered_list_of_points) - 1 - index)
                break
    ordered_list_of_points = reversed(ordered_list_of_points)
    focus_points_indices.sort()
    stop_points_indices = [0]
    first_point_index = second_point_index = 0
    focus_points_encountered = 0
    if 0 in focus_points_indices:
        focus_points_encountered = 1
    while second_point_index != len(ordered_list_of_points) - 1:
        next_focus_point_index = focus_points_indices[focus_points_encountered]
        first_point_index = second_point_index
        second_point_index += segment_length
        if next_focus_point_index - second_point_index < segment_length:
            second_point_index = next_focus_point_index
        if second_point_index in focus_points_indices:
            focus_points_encountered += 1
        stop_points_indices.append(second_point_index)
    for index in range(0, len(stop_points_indices) - 1):
        i_pt_1 = ordered_list_of_points[stop_points_indices[index]]
        i_pt_2 = ordered_list_of_points[stop_points_indices[index + 1]]
        y1, x1 = img_arr_idx(WIDTH, i_pt_1)
        y2, x2 = img_arr_idx(WIDTH, i_pt_2)
        draw_points.add(i_pt_1)
        if i_pt_2 == ordered_list_of_points[len(ordered_list_of_points) - 1]:
            draw_points.add(i_pt_2)
        line = (x1, y1, x2, y2)
        if calculate_slope(line, False) == "UND":
            if y2 < y1:
                y1, y2 = y2, y1
            for y in range(y1, y2):
                draw_points.add((y, x1))
        else:
            if x2 < x1:
                x1, x2 = x2, x1
            for x in range(x1, x2):
                y_val = calculate_y_val(line, x)
                next_y_val = calculate_y_val(line, x + 1)
                m_sign = 1
                if next_y_val < y_val:
                    y_val, next_y_val = next_y_val, y_val
                    m_sign = -1
                for y in range(math.floor(y_val), math.ceil(next_y_val)):
                    x_start = calculate_x_val(line, y)
                    x_end = calculate_x_val(line, y + 1)
                    if y == math.ceil(next_y_val) - 1 * m_sign:
                        x_end = x + 1 * m_sign
                    x_mid = (x_start + x_end)/2
                    y_mid = calculate_y_val(line, x_mid)
                    draw_points.add((math.floor(y_mid), x))
                draw_points.add((math.floor(y_val), x))
    for child in tree_parser.children:
        draw_points = linear_approximation_helper(intersections, child, branch_idx, segment_length, draw_points)
    return draw_points

'''                      ^
    IMAGE ANALYSIS       |
                         |   '''

# -------------------------------

'''                       |
    DATA TRANSCRIPTION    |
                          v   '''

def derive_data(text, file):
    formatted_text = text.replace(" ", "_")
    image_ending = ""
    with open("math_obj_image_endings/" + formatted_text + "_IMAGEENDINGS.txt") as f:
        ending = ".png"
        for line in f:
            ending = line
            break
        if ending[len(ending) - 1] == "\n":
            ending = ending[:len(ending) - 1]
        image_ending = ending
    img_path = "math_obj_images/original_images/" + formatted_text + "_images/"
    img_name = formatted_text + "1" + image_ending
    opaque_output_path = "math_obj_images/opaque_images/opaque_" + formatted_text + "_images/"
    opaqued_img = opaque_img(img_path, img_name, opaque_output_path)
    resize_output_path = "math_obj_images/resized_images/resized_" + formatted_text + "_images/"
    resized_img = resize_img(opaque_output_path, "opaque_" + img_name, resize_output_path, 300)
    DIMENSIONS = resized_img.shape[:2]
    main_img_idx = initialize_img(DIMENSIONS)
    prep_for_img_analysis(resized_img, main_img_idx)

    print(len(IMG_INFO[main_img_idx][3]))
    cv_img = to_cv_img(main_img_idx)
    cv2.imwrite("CIM_init_prep.jpg", cv_img)

    lin_app_img_idx, objects = linear_approximation(main_img_idx)
    remove_small_objects(lin_app_img_idx, 60)
    edges = find_edges(lin_app_img_idx, False, img_objects = objects)
    edge_lengths = []
    for edge in edges:
        length = len(edge[1])
        if length not in edge_lengths:
            edge_lengths.append(length)
    edge_lengths.sort(reverse = True)
    edge_length_dict = []
    for length in edge_lengths:
        edge_length_dict.append([length, []])
    for edge in edges:
        length = len(edge[1])
        for item in edge_length_dict:
            if item[0] == length:
                item[1].append(edge)
                break
    sorted_edges = []
    for item in edge_length_dict:
        for edge in item[1]:
            sorted_edges.append(edge)
    points = []
    segments = []
    for edge in sorted_edges:
        closeness_bound = 5
        possible_bound = math.floor(len(edge[1])/8)
        if possible_bound > closeness_bound:
            closeness_bound = possible_bound
        edge_points, points_in_edge, object = edge
        endpoint_1 = points_in_edge[0]
        endpoint_2 = points_in_edge[len(points_in_edge) - 1]
        for point, iter_obj in points:
            endpoint_y = endpoint_1[0]
            endpoint_x = endpoint_1[1]
            point_y = point[0]
            point_x = point[1]
            if abs(endpoint_y - point_y) <= closeness_bound and abs(endpoint_x - point_x) <= closeness_bound and object == iter_obj:
                endpoint_1 = point
                break
        for point, iter_obj in points:
            endpoint_y = endpoint_2[0]
            endpoint_x = endpoint_2[1]
            point_y = point[0]
            point_x = point[1]
            if abs(endpoint_y - point_y) <= closeness_bound and abs(endpoint_x - point_x) <= closeness_bound and object == iter_obj:
                endpoint_2 = point
                break
        if (endpoint_1, object) not in points:
            points.append((endpoint_1, object))
        if (endpoint_2, object) not in points:
            points.append((endpoint_2, object))
        segment = [points.index((endpoint_1, object)), points.index((endpoint_2, object))]
        reverse_segment = [segment[1], segment[0]]
        if segment not in segments and reverse_segment not in segments:
            segments.append(segment)
    init_point, init_object = points[0]
    for point, object in points:
        if point[1] < init_point[1]:
            init_point = point
            init_object = object
        elif point[1] == init_point[1]:
            if point[0] < init_point[0]:
                init_point = point
                init_object = object
    init_point_index = points.index((init_point, init_object))
    points.remove((init_point, init_object))
    for segment in segments:
        for index in segment:
            add_index = segment.pop(0)
            if add_index < init_point_index:
                add_index += 1
            elif add_index == init_point_index:
                add_index = 0
            segment.append(add_index)
    init_y, init_x = init_point
    shifted_points = []
    for point, object in points:
        shifted_y = point[0] - init_y
        shifted_x = point[1] - init_x
        shifted_points.append((shifted_y, shifted_x))
    max_value = max([max(abs(y), abs(x)) for y, x in shifted_points])
    scale_constant = max_value/3
    final_points = []
    for point in shifted_points:
        final_y = -1 * point[0]/scale_constant
        final_x = point[1]/scale_constant
        final_points.append((final_y, final_x))
    return final_points, segments

def get_parameters(text):
    object = create_object(text)
    obj_text, obj_type = object
    name = " ".join([word[0].upper() + word[1:] for word in text.split(" ")])
    with open("object_frames.txt", "a") as f:
        f.write("OBJ - " + name + "\n")
        if obj_type == 1:
            f.write("TXT - " + obj_text)
        elif obj_type == 2:
            points = []
            segments = []
            file = get_img_urls(text, "mathematics")
            data = derive_data(text, file)
            points = data[0]
            segments = data[1]
            f.write("PTS - ")
            for index in range(0, len(points)):
                point = points[index]
                point = (point[1], point[0])
                result = str(point).replace(", ", ":") + ","
                if index == len(points) - 1:
                    result = result[0:len(result) - 1] + "\n"
                f.write(result)
            f.write("SEG - ")
            for index in range(0, len(segments)):
                segment = segments[index]
                result = str(segment).replace(", ", ":") + ","
                if index == len(segments) - 1:
                    result = result[0:len(result) - 1] + "\n"
                f.write(result)
        f.write("\n")
        f.close()

'''                       ^
    DATA TRANSCRIPTION    |
                          |   '''

def main():
    args = sys.argv[1:]
    term = "rectangle"#args[0]
    print("check")
    #get_parameters(term)
    #print(term)


    '''A_set = {(8, 5), (9, 6), (9, 7), (9, 8), (10, 8), (11, 7)}
    B_set = {(8, 5), (9, 6), (9, 7), (9, 8), (10, 8), (11, 7)}
    B_grid = [
    [False for i in range(20)] for j in range(20)
    ]
    for y, x in B_set:
        B_grid[y][x] = True
    A = (A_set,)
    B = (B_set, B_grid)
    print(is_near_subset(A, B, 3))'''



    '''B_width = 20
    B_length = 20
    closeness_bound = 3
    for A_y, A_x in A_set:
        for i in range(closeness_bound + 1):
            min_x = A_x - i
            max_x = A_x + i
            min_y = A_y - i
            max_y = A_y + i
            for x in range(min_x, max_x + 1):
                if 0 <= x < B_width:
                    if 0 <= min_y < B_length:
                        B_grid[min_y][x] = "*"
                    if 0 <= max_y < B_length:
                        B_grid[max_y][x] = "*"
            else:
                for y in range(min_y, max_y + 1):
                    if 0 <= y < B_length:
                        print(True)
                        if 0 <= min_x < B_width:
                            B_grid[y][min_x] = "*"
                        if 0 <= max_x < B_width:
                            B_grid[y][max_x] = "*"
    for row in B_grid:
        print(row)'''

if __name__ == "__main__":
    main()