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

def capitalize_first_letters(text):
    return " ".join(word[0].upper() + word[1:] for word in text.split(" "))

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
    symbol_dict = {"plus" : "+"} # WILL FILL LATER
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
def remove_small_objects(img, max_size):
    dimensions = img.shape
    height = dimensions[0]
    width = dimensions[1]

    # List of white point groups in img.
    white_object_point_lists = []
    for y in range(0, height):
        for x in range(0, width):
            if img[y][x] >= 250:
                point = (y, x)
                in_new_object = True
                for list in white_object_point_lists:
                    if point in list:
                        in_new_object = False
                        break
                if in_new_object:
                    set_of_points = get_set_of_points_from_point(img, point, 1)
                    white_object_point_lists.append(set_of_points)

    # List of black point groups in img.
    black_object_point_lists = []
    black_point_count = 0
    for y in range(0, height):
        for x in range(0, width):
            if img[y][x] <= 5:
                black_point_count += 1
                point = (y, x)
                in_new_object = True
                for list in black_object_point_lists:
                    if point in list:
                        in_new_object = False
                        break
                if in_new_object:
                    set_of_points = get_set_of_points_from_point(img, point, 0)
                    black_object_point_lists.append(set_of_points)

    # If max_size is -1, make it a "smart" size equal to roughly log base 2 of the number of black points in img. The
    # idea behind this choice is that the black points are the main points I care about for this function.
    if max_size == -1:
        max_size = black_point_count.bit_length() - 1

    # Fill in all the small white groups.
    for list in white_object_point_lists:
        if len(list) < max_size:
            for point in list:
                y, x = point
                img[y][x] = 0

    # Fill in all the small black groups.
    for list in black_object_point_lists:
        if len(list) < max_size:
            for point in list:
                y, x = point
                img[y][x] = 255
    return img

# PRECONDITION: img has no transparent points.
#
# Returns a black and white image containing the objects in img.
def get_thresholded_img(img, current_points, edge_points, thresh):
    '''edge_colors = []
    height, width = img.shape[0:2]
    for y, x in current_points:
        if y == 0 or y == height - 1 or x == 0 or x == width - 1:
            edge_colors.append(img[y, x])
        else:
            is_edge_pixel = False
            for iter_y in range(y - 1, y + 2):
                for iter_x in range(x - 1, x + 2):
                    if (y, x) not in current_points:
                        is_edge_pixel = True
            if is_edge_pixel:
                edge_colors.append(img[y, x])
    color_mode = 1
    edge_color_set = set(edge_colors)
    for color in edge_color_set:
        if edge_colors.count(color) > color_mode:
            color_mode = edge_colors.count(color)
    background_color = (255, 255, 255)
    for color in edge_color_set:
        if edge_colors.count(color) == color_mode:
            background_color = color
            break
    current_thresh = background_threshold(img, background_color, current_points)
    for y, x in current_points:
        if current_thresh[y, x][0] == current_thresh[y, x][1] == current_thresh[y, x][2] == 0:
            thresh[y, x] = (0, 0, 0)
    next_objects = get_sets_of_colored_points(current_thresh, 0)
    for object in next_objects:
        thresh = get_thresholded_img(img, object, thresh)
    return thresh'''

    # Don't continue if the current points are small in number (extraneous).
    if len(current_points) <= 5:
        return thresh
    else:
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

        # The keys are colors and the values are the number of points in edge_points with that color.
        '''edge_colors_dict = {}
        for y, x in edge_points:
            edge_colors_dict[(color := img[y, x])] if color not in edge_colors_dict else edge_colors_dict[color] + 1'''

        edge_point = edge_points[0]

        # current_thresh is an image with points in current_points colored black if they have significantly
        # different colors in img than the background color.
        current_thresh = background_threshold(img, background_color, current_points, edge_point)

        print(f"BLACK POINTS: {sum([1 for y in range(height) for x in range(width) if current_thresh[y, x][0] == 0])}")

        current_thresh = cv2.cvtColor(current_thresh, cv2.COLOR_BGR2GRAY)

        # Do a smart removal of small objects from current_thresh.
        current_thresh = remove_small_objects(current_thresh, -1)

        # All of the black groups of points in current_thresh.
        current_objects = get_sets_of_colored_points(current_thresh, 0)
        for object in current_objects:
            current_object = np.zeros((height, width, 3), np.uint8)
            current_object = ~current_object
            current_object = cv2.cvtColor(current_object, cv2.COLOR_BGR2GRAY)

            # Draw object to current_object.
            for y, x in object:
                current_object[y, x] = 0
            hollow, drawn_points = is_hollow(current_object)

            # If object is hollow, draw it exactly to thresh.

            DP = set()

            if hollow:
                for y, x in object:
                    DP.add((y, x))
                    thresh[y, x] = (0, 0, 0)

            # Otherwise, draw its edge to thresh.
            else:
                for y, x in drawn_points:
                    DP.add((y, x))
                    thresh[y, x] = (0, 0, 0)

            print(f"GET THRESH (NON CIM): {len(DP)}")

            # removed_points are the first five outer layers of object, and current_edge_points is the sixth.
            current_edge_points, removed_points = get_edge_points(current_object, object, [], 6)

            # Remove the first five layers from object.
            for point in removed_points:
                object.remove(point)

            # Draw onto thresh the threshold of all objects present within img. What this does is it gets objects
            # inside of other objects that have distinctly different colors. For an example, look at "set" on
            # wikipedia.
            thresh = get_thresholded_img(img, object, current_edge_points, thresh)
        return thresh

# PRECONDITION: img is black and white.
#
# Thins img to specific requirements.
def thin_img(img):
    dimensions = img.shape
    height = dimensions[0]
    width = dimensions[1]

    # The coordinates of a point's range 1 neighbors relative to it.
    neighbor_coords = ((1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1))
    for y in range(0, height):
        for x in range(0, width):
            if img[y][x] <= 5:

                # List of booleans for each neighbor in the range 1 neighborhood around the current point
                # where True means that the neghbor is in img and is black, and False means otherwise.
                marked_neighbors = []
                for coord in neighbor_coords:
                    new_y = y + coord[0]
                    new_x = x + coord[1]
                    if new_y >= 0 and new_y < height and new_x >= 0 and new_x < width:
                        marked_neighbors.append(img[new_y][new_x] <= 5)
                    else:
                        marked_neighbors.append(False)

                # List of consecutive segments of points in the range 1                           o . .
                # neighborhood around the current point. For example,         ----->              o * o
                # the point to the right has two neighbor segments                                . o o
                # (a 'o' means a black point).
                neighbor_segments = [[]]
                first_index_marked = False
                for index in range(0, 8):
                    front_segment = -1
                    front_segment = neighbor_segments[len(neighbor_segments) - 1]
                    if marked_neighbors[index] == False:
                        if front_segment != []:
                            neighbor_segments.append([])
                    else:
                        coord = neighbor_coords[index]
                        new_y = y + coord[0]
                        new_x = x + coord[1]
                        front_segment.append((new_y, new_x))
                        if index == 0:
                            first_index_marked = True
                        if index == 7 and first_index_marked and len(neighbor_segments) > 1:
                            first_segment = neighbor_segments.pop(0)
                            for point in first_segment:
                                front_segment.append(point)
                if [] in neighbor_segments:
                    neighbor_segments.remove([])

                # If there are no neigbor segments around the current point, then make the current point white. If
                # there is only one segment, and if it contains more than one point, them make the current point
                # white.
                if len(neighbor_segments) <= 1:
                    if len(neighbor_segments) == 0:
                        img[y][x] = 255
                    else:
                        if len(neighbor_segments[0]) > 1:
                            img[y][x] = 255
    return img

# PRECONDITION: img is black and white.
#
# Thickens img by an amount designatied by thick_factor.
def thicken_img(img, thick_factor):
    dimensions = img.shape
    height = dimensions[0]
    width = dimensions[1]

    # The result image.
    thick = np.zeros((height, width, 3), np.uint8)
    thick = ~thick

    # Go thorugh all the points and only care about the black points.
    for y in range(0, height):
        for x in range(0, width):
            if img[y][x] <= 5:

                # Mark the result image black at the current point.
                thick[y][x] = 0

                # For each value i in the range of [1, range_factor], in the result image, mark each point adjacent
                # to the current point i pixels away.
                for i in range(1, thick_factor + 1):
                    if y - i >= 0 and y - i < height and x >= 0 and x < width:
                        thick[y - i][x] = 0
                    if y >= 0 and y < height and x - i >= 0 and x - i < width:
                        thick[y][x - i] = 0
                    if y + i >= 0 and y + i < height and x >= 0 and x < width:
                        thick[y + i][x] = 0
                    if y >= 0 and y < height and x + i >= 0 and x + i < width:
                        thick[y][x + i] = 0
    return thick

# One-stop-shop for image preparation according to everything required for the analysis to work.
def prep_for_img_analysis(img):
    cv2.imwrite("pre-prep.jpg", img)
    dimensions = img.shape
    height = dimensions[0]
    width = dimensions[1]
    thresh = np.zeros((height, width, 3),np.uint8)
    thresh = ~thresh
    points = []

    # The initial edge points (on the borders of img.)
    edge_points = []
    for y in range(height):
        for x in range(width):
            points.append((y, x))
            if y == 0 or y == height - 1 or x == 0 or x == width - 1:
                edge_points.append((y, x))
    thresh = get_thresholded_img(img, points, edge_points, thresh)
    frame_img = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("init_prep_pre-thick.jpg", frame_img)
    frame_img = thicken_img(frame_img, 1)
    cv2.imwrite("init_prep_pre-thin.jpg", frame_img)
    frame_img = cv2.cvtColor(frame_img, cv2.COLOR_BGR2GRAY)
    frame_img = ~frame_img
    thin = cv2.ximgproc.thinning(frame_img)
    thin = ~thin
    final = thin_img(thin)
    return final

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
    '''cv_height, cv_width = cv_img.shape[0:2]
    for y in range(0, cv_height):
        for x in range(0, cv_width):
            is_white = True
            for val in cv_img[y, x]:
                if val < 230:
                    is_white = False
                    break
            if is_white:
                cv_img[y, x] = (255, 255, 255)
    cv2.imwrite(output_path + "R" + str(max_dim) + "-" + img_name, cv_img)'''
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
    '''img = Image.open(img_path + img_name)
    img = img.convert("RGBA")
    datas = img.getdata()
    new_data = []
    for item in datas:
        is_white = True
        for val in item[0:3]:
            if val < 240:
                is_white = False
        if item[3] == 0:
            new_data.append((255, 255, 255, 255))
        elif is_white:
            new_data.append((0, 0, 0, 255))
        else:
            new_data.append(item)
    img.putdata(new_data)
    img.save(output_path + "opaque_" + img_name)
    opaque_img = cv2.imread(output_path + "opaque_" + img_name)
    return opaque_img'''

    '''cv_img = cv2.imread(img_path + img_name)
    pil_img = Image.open(img_path + img_name)
    pil_img = pil_img.convert("RGBA")
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
    for point in transparent_points:
        y, x = point
        cv_img[y, x] = (255, 255, 255)
    white_point_sets = get_sets_of_colored_points(cv_img, 3)
    for point_set in white_point_sets:
        is_border_set = False
        for point in point_set:
            y, x = point
            if y == 0 or y == height - 1 or x == 0 or x == width - 1:
                is_border_set = True
        if not is_border_set:
            for point in point_set:
                y, x = point
                cv_img[y, x] = (0, 0, 0)
    cv2.imwrite(output_path + "opaque_" + img_name, cv_img)
    return cv_img'''

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
    '''m = 0
    if line[2] - line[0] == 0:
        if approximate == True:
            m = 10000
        else:
            m = "UND"
    else:
        m = (line[3] - line[1])/(line[2] - line[0])
    return m'''
    return ["UND", 10000][approximate] if line[2] - line[0] == 0 else (line[3] - line[1])/(line[2] - line[0])

def floor_point(point):
    return (math.floor(point[0]), math.floor(point[1]))

def is_near_subset(A, B, closeness_bound): # A is near subset of B
    if len(A) > len(B):
        return False
    else:
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
    '''a, b, r = circle
    if (r ** 2) - ((x - a) ** 2) < 0:
        return "none"'''
    a, b, r = circle
    return "none" if (r ** 2) - ((x - a) ** 2) < 0 else sign * math.sqrt((r ** 2) - ((x - a) ** 2)) + b

def remove_near_subset_elements(lst, closeness_bound):
    new_list = []
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
    return final_list
    '''lst = [item for i, item in enumerate(lst) if item not in lst[:i]]
    lst_copy = list[:]'''


def hc_accum_array(img, radius_values):
    h = img.shape[0]
    w = img.shape[1]
    accum_array = np.zeros((len(radius_values), h, w))
    for i in range(h):
        for j in range(w):
            if img[i][j] != 0:
                for r in range(len(radius_values)):
                    rr = radius_values[r]
                    hdown = max(0, i - rr)
                    for a in range(hdown, i):
                        b = round(j+math.sqrt(rr*rr - (a - i) * (a - i)))
                        if b>=0 and b<=w-1:
                            accum_array[r][a][b] += 1
                            if 2 * i - a >= 0 and 2 * i - a <= h - 1:
                                accum_array[r][2 * i - a][b] += 1
                        if 2 * j - b >= 0 and 2 * j - b <= w - 1:
                            accum_array[r][a][2 * j - b] += 1
                        if 2 * i - a >= 0 and 2 * i - a <= h - 1 and 2 * j - b >= 0 and 2 * j - b <= w - 1:
                            accum_array[r][2 * i - a][2 * j - b] += 1
    return accum_array

def find_circles(img, accum_array, radius_values, hough_thresh):
    returnlist = []
    hlist = []
    wlist = []
    rlist = []
    returnimg = img.copy()
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

def hough_circles(img):
    radius_values = []
    for radius in range(30):
        radius_values.append(radius)
    inv_img = ~img
    accum_array = hc_accum_array(inv_img, radius_values)
    hough_thresh = 30
    result_list = find_circles(img, accum_array, radius_values, hough_thresh)
    return result_list

def find_curves(img):
    #img = thicken_img(img, 1)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thick = thicken_img(img, 2)
    thick = cv2.cvtColor(thick, cv2.COLOR_BGR2GRAY)
    dimensions = img.shape
    height = dimensions[0]
    width = dimensions[1]
    '''new = np.zeros((height, width, 3), np.uint8)
    new = ~new
    for y in range(0,height):
        for x in range(0,width):
            if img[y][x] <= 5:
                new[y][x] = (0,0,0)'''
    curve_list = []
    detected_circles = hough_circles(img)#cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 20, param1 = 20, param2 = 3, minRadius = 3, maxRadius = width)
    if detected_circles is not None:
        #detected_circles = np.uint16(np.around(detected_circles))
        for circle in detected_circles:#[0, :]:
            upper_curve_marks = {}
            lower_curve_marks = {}
            a_int, b_int, r_int = circle[0], circle[1], circle[2]
            #cv2.circle(new, (a_int, b_int), r_int, (0,0,255), 1)
            for x in range(a_int - r_int, a_int + r_int):
                if x >= 0 and x < width:
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
                        if y >= 0 and y < height:
                            if thick[y][x] <= 5:
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
                        if y >= 0 and y < height:
                            if thick[y][x] <= 5:
                                lower_curve_marks[x].append(y)
            y_bound = 1
            last_y = -1
            curve_between_halves = []
            front_curve = []
            x = a_int - r_int
            while x in range(a_int - r_int, a_int + r_int) and x >= 0 and x < width:
                if len(upper_curve_marks[x]) != 0:
                    current_curve_marks = upper_curve_marks[x]
                    last_y = current_curve_marks[0]
                    this_curve = []
                    near_y = True
                    while x in upper_curve_marks and len(current_curve_marks) > 0 and near_y:
                        current_curve_marks.sort()
                        if x >= a_int:
                            current_curve_marks.reverse()
                        for y in current_curve_marks:
                            if abs(y - last_y) > y_bound:
                                near_y = False
                                break
                            this_curve.append((y, x))
                            last_y = y
                        x += 1
                        if x in upper_curve_marks:
                            current_curve_marks = upper_curve_marks[x]
                    if front_curve == []:
                        front_curve = this_curve
                    elif x in upper_curve_marks:
                        if len(this_curve) >= 3:
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
            while x >= a_int - r_int and x >= 0 and x < width:
                if len(lower_curve_marks[x]) != 0:
                    current_curve_marks = lower_curve_marks[x]
                    last_y = current_curve_marks[0]
                    if x == a_int + r_int - 1 and curve_between_halves != []:
                        last_y = curve_between_halves[len(curve_between_halves) - 1][0]
                    this_curve = curve_between_halves
                    curve_between_halves = []
                    near_y = True
                    while x in lower_curve_marks and len(current_curve_marks) > 0 and near_y:
                        current_curve_marks.sort()
                        if x >= a_int:
                            current_curve_marks.reverse()
                        for y in current_curve_marks:
                            if abs(y - last_y) > y_bound:
                                near_y = False
                                break
                            this_curve.append((y, x))
                            last_y = y
                        x -= 1
                        if x in lower_curve_marks:
                            current_curve_marks = lower_curve_marks[x]
                    if x not in lower_curve_marks and front_curve != -1:
                        this_curve += front_curve
                    if len(this_curve) >= 3:
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
    return final_list

def find_edges_line_analysis(img, lines):
    dimensions = img.shape
    height = dimensions[0]
    width = dimensions[1]
    edge_list = []
    #line_ct = len(lines)
    #ct = 0
    for line in lines:
        #ct += 1
        #print("Position: " + str(line_ct - ct))
        m = calculate_slope(line, False)
        if m != "UND":
            '''b = calculate_y_val(line, 0)
            if abs(m - desired_slope) <= 0.01 and abs(b - desired_b) <= 0.05:
                print(line)
                print(point_slope_form(line))
                print("---")'''
            x1 = line[0]
            y1 = line[1] - calculate_y_val(line, 0)
            '''a = 1 + m ** 2
            b = (2 * x1 * (m ** 2)) + (-2 * m * y1)
            c = -1 * (1 + ((m ** 2) * (x1 ** 2)) - (2 * m * x1 * y1) + (y1 ** 2))
            first_x = (-b + math.sqrt((b ** 2) - (4 * a * c)))/(2 * a)
            second_x = (-b - math.sqrt((b ** 2) - (4 * a * c)))/(2 * a)
            x_step = first_x
            if second_x > 0:
                x_step = second_x'''
            edge_marks = {}
            '''for x_value in range(0, width):
                y_value = calculate_y_val(line, x_value)
                next_x_value = x_value + 1
                next_y_value = calculate_y_val(line, next_x_value)
                if x_value not in edge_marks:
                    edge_marks[x_value] = []
                start_index = math.floor(y_value)
                end_index = math.floor(next_y_value)
                if start_index > end_index:
                    start_index = math.floor(next_y_value)
                    end_index = math.floor(y_value)
                for y_coordinate in range(start_index, end_index):
                    if y_coordinate >= 0 and y_coordinate < height:
                        if prep[y_coordinate][x_value] <= 5:
                            edge_marks[x_value].append(y_coordinate)
                calc_x_val = calculate_x_val(line, end_index)
                if calc_x_val == "none":
                    calc_x_val = x_value
                if math.floor(calc_x_val) == x_value:
                    if end_index >= 0 and end_index < height:
                        if prep[end_index][x_value] <= 5:
                            edge_marks[x_value].append(end_index)'''
            for x in range(0, width):
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
                    if y >= 0 and y < height:
                        if img[y][x] <= 5:
                            edge_marks[x].append(y)
            last_y = -1
            x = 0
            while x in range(0, width):
                if len(edge_marks[x]) != 0:
                    current_edge_marks = edge_marks[x]
                    if m < 0:
                        current_edge_marks.reverse()
                    last_y = current_edge_marks[0]
                    y_bound = 1
                    this_edge = []
                    near_y = True
                    while x in edge_marks and len(current_edge_marks) > 0 and near_y:
                        current_edge_marks = edge_marks[x]
                        if m < 0:
                            current_edge_marks.reverse()
                        for y in current_edge_marks:
                            if abs(y - last_y) > y_bound:
                                near_y = False
                                break
                            this_edge.append((y, x))
                            last_y = y
                        x += 1
                    if len(this_edge) >= 3:
                        is_new_edge = True
                        new_edge_list = []
                        for edge in edge_list:
                            if is_near_subset(edge[1], this_edge, 5):
                                continue
                            elif is_near_subset(this_edge, edge[1], 5):
                                is_new_edge = False
                            new_edge_list.append(edge)
                        if is_new_edge:
                            new_edge_list.append((line, this_edge))
                        edge_list = new_edge_list
                x += 1
        else:
            edge_marks = {}
            line_x_val = math.floor(line[0])
            for y in range(0, height):
                if img[y][line_x_val] <= 5:
                    edge_marks[y] = True
                else:
                    edge_marks[y] = False
            y = 0
            while y in range(0, height):
                if edge_marks[y] == True:
                    this_edge = []
                    current_edge_mark = edge_marks[y]
                    while y in edge_marks and current_edge_mark == True:
                        current_edge_mark = edge_marks[y]
                        this_edge.append((y, line_x_val))
                        y += 1
                    if len(this_edge) >= 3:
                        is_new_edge = True
                        new_edge_list = []
                        for edge in edge_list:
                            if is_near_subset(edge[1], this_edge, 5):
                                continue
                            elif is_near_subset(this_edge, edge[1], 5):
                                is_new_edge = False
                            new_edge_list.append(edge)
                        if is_new_edge:
                            new_edge_list.append((line, this_edge))
                        edge_list = new_edge_list
                y += 1
    edge_list = remove_near_subset_elements(edge_list, 5)
    #print(edge_list)
    return edge_list

def hough_lines(edge_image, num_rhos=180, num_thetas=180, t_count=3):
    edge_height, edge_width = edge_image.shape[:2]
    edge_height_half, edge_width_half = edge_height / 2, edge_width / 2
    d = np.sqrt(np.square(edge_height) + np.square(edge_width))
    dtheta = 180 / num_thetas
    drho = (2 * d) / num_rhos
    thetas = np.arange(0, 180, step=dtheta)
    rhos = np.arange(-d, d, step=drho)
    cos_thetas = np.cos(np.deg2rad(thetas))
    sin_thetas = np.sin(np.deg2rad(thetas))
    accumulator = np.zeros((len(rhos), len(rhos)))
    for y in range(edge_height):
        for x in range(edge_width):
            if edge_image[y][x] != 0:
                edge_point = [y - edge_height_half, x - edge_width_half]
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
                x0 = (a * rho) + edge_width_half
                y0 = (b * rho) + edge_height_half
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                line = (x1, y1, x2, y2)
                lines.add(line)
    return lines

def find_edges(img, intersection_mode, img_objects = -1):
    '''thick = thicken_img(img, 2)
    thick = cv2.cvtColor(thick, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(img, 10, 20)
    hough_lines = []
    threshold = 10
    for rho in range(0, 30):
        for theta in range(0, 19):
            current_hough_lines = cv2.HoughLines(edges, 0.1 + rho * 0.1, np.pi/(360 - 13 * theta), threshold, max_theta = 2 * np.pi)
            if current_hough_lines is not None:
                for line in current_hough_lines:
                    hough_lines.append(line)
    line_packages = []
    lines = set()
    for r_theta in hough_lines:
        r,theta = r_theta[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * r
        y0 = b * r
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        lines.add((x1, y1, x2, y2))
        if len(lines) == 20000:
            line_packages.append(lines)
            lines = set()
    if len(lines) > 0:
        line_packages.append(lines)
    line_packages = []
    current_lines = []
    for line in lines:
        current_lines.append(line)
        if len(current_lines) == 20000:
            line_packages.append(current_lines)
            current_lines = []
    if len(current_lines) > 0:
        line_packages.append(current_lines)
    edge_list = []
    with Pool() as pool:
        list_of_packages = [package for package in line_packages]
        package_edge_lists = pool.starmap(find_edges_line_analysis, zip(repeat(thick), list_of_packages))
    for list in package_edge_lists:
        edge_list += list
    final_list = remove_near_subset_elements(edge_list, 4)'''
    dimensions = img.shape
    height = dimensions[0]
    width = dimensions[1]
    thickness_factor = 2 if intersection_mode else 4
    #cv2.imwrite("first_FE_pre-thick.jpg", img)
    thick = thicken_img(img, thickness_factor)
    #cv2.imwrite("first_FE_post-thick.jpg", thick)
    thick = cv2.cvtColor(thick, cv2.COLOR_BGR2GRAY)
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
    lines = hough_lines(~thick)
    edge_list = []
    line_ct = len(lines)
    print(line_ct)
    print("done")
    print(stop)
    ct = 0
    for line in lines:
        ct += 1
        print("Position: " + str(line_ct - ct))
        m = calculate_slope(line, False)
        if m != "UND":
            '''b = calculate_y_val(line, 0)
            if abs(m - desired_slope) <= 0.01 and abs(b - desired_b) <= 0.05:
                print(line)
                print(point_slope_form(line))
                print("---")'''
            x1 = line[0]
            y1 = line[1] - calculate_y_val(line, 0)
            '''a = 1 + m ** 2
            b = (2 * x1 * (m ** 2)) + (-2 * m * y1)
            c = -1 * (1 + ((m ** 2) * (x1 ** 2)) - (2 * m * x1 * y1) + (y1 ** 2))
            first_x = (-b + math.sqrt((b ** 2) - (4 * a * c)))/(2 * a)
            second_x = (-b - math.sqrt((b ** 2) - (4 * a * c)))/(2 * a)
            x_step = first_x
            if second_x > 0:
                x_step = second_x'''
            edge_marks = {}
            '''for x_value in range(0, width):
                y_value = calculate_y_val(line, x_value)
                next_x_value = x_value + 1
                next_y_value = calculate_y_val(line, next_x_value)
                if x_value not in edge_marks:
                    edge_marks[x_value] = []
                start_index = math.floor(y_value)
                end_index = math.floor(next_y_value)
                if start_index > end_index:
                    start_index = math.floor(next_y_value)
                    end_index = math.floor(y_value)
                for y_coordinate in range(start_index, end_index):
                    if y_coordinate >= 0 and y_coordinate < height:
                        if prep[y_coordinate][x_value] <= 5:
                            edge_marks[x_value].append(y_coordinate)
                calc_x_val = calculate_x_val(line, end_index)
                if calc_x_val == "none":
                    calc_x_val = x_value
                if math.floor(calc_x_val) == x_value:
                    if end_index >= 0 and end_index < height:
                        if prep[end_index][x_value] <= 5:
                            edge_marks[x_value].append(end_index)'''
            for x in range(0, width):
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
                    if y >= 0 and y < height:
                        if thick[y][x] <= 5:
                            edge_marks[x].append(y)
            last_y = -1
            x = 0
            while x in range(0, width):
                if len(edge_marks[x]) != 0:
                    current_edge_marks = edge_marks[x]
                    if m < 0:
                        current_edge_marks.reverse()
                    last_y = current_edge_marks[0]
                    y_bound = 1
                    this_edge = []
                    near_y = True
                    while x in edge_marks and len(current_edge_marks) > 0 and near_y:
                        current_edge_marks = edge_marks[x]
                        if m < 0:
                            current_edge_marks.reverse()
                        for y in current_edge_marks:
                            if abs(y - last_y) > y_bound:
                                near_y = False
                                break
                            this_edge.append((y, x))
                            last_y = y
                        x += 1
                    if len(this_edge) >= 3:
                        is_new_edge = True
                        new_edge_list = []
                        for edge in edge_list:
                            if is_near_subset(edge[1], this_edge, 2*thickness_factor):
                                continue
                            elif is_near_subset(this_edge, edge[1], 2*thickness_factor):
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
                                    count = object_point_counts[object]
                                    if count > max_count:
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
            if line_x_val >= 0 and line_x_val < width:
                for y in range(0, height):
                    if thick[y][line_x_val] <= 5:
                        edge_marks[y] = True
                    else:
                        edge_marks[y] = False
                y = 0
                while y in range(0, height):
                    if edge_marks[y] == True:
                        this_edge = []
                        current_edge_mark = edge_marks[y]
                        while y in edge_marks and current_edge_mark == True:
                            current_edge_mark = edge_marks[y]
                            this_edge.append((y, line_x_val))
                            y += 1
                        if len(this_edge) >= 3:
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
                                        count = object_point_counts[object]
                                        if count > max_count:
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
    final_remove_factor = 1 if intersection_mode else 2
    if len(edge_list) > 60:
        edge_list.sort(key = lambda edge: len(edge[1]))
        edge_list = edge_list[int(len(edge_list) * (3/4)):]
    final_list = remove_near_subset_elements(edge_list, thickness_factor * final_remove_factor)
    return final_list

def calculate_y_val(line, x_val):
    '''x1 = line[0]
    y1 = line[1]
    x2 = line[2]
    y2 = line[3]
    m = 0
    if x2 - x1 == 0:
        m = "UND"
    else:
        m = (y2 - y1)/(x2 - x1)
    if m != "UND":
        return m * (x_val - x1) + y1
    else:
        return "none"# *'''
    x1, y1, x2, y2 = line
    return "none" if x2 - x1 == 0 else ((y2 - y1)/(x2 - x1)) * (x_val - x1) + y1

def calculate_x_val(line, y_val):# *
    '''x1 = line[0]
    y1 = line[1]
    x2 = line[2]
    y2 = line[3]
    m = 0
    if x2 - x1 == 0:
        m = "UND"
    else:
        m = (y2 - y1)/(x2 - x1)
    if m != "UND" and m != 0:
        return (y_val - y1)/m + x1
    elif m == "UND":
        return x1
    else:
        return "none"'''
    x1, y1, x2, y2 = line
    return "none" if x2 - x1 == 0 else (y_val - y1)/((y2 - y1)/(x2 - x1)) + x1

def solve_quadratic(a, b, c):
    '''if a == 0 or (b ** 2) - (4 * a * c) < 0:
        return "none"
    first_x = (-b + math.sqrt((b ** 2) - (4 * a * c)))/(2 * a)
    second_x = (-b - math.sqrt((b ** 2) - (4 * a * c)))/(2 * a)
    return (first_x, second_x)'''
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

def find_intersections(img):
    dimensions = img.shape
    height = dimensions[0]
    width = dimensions[1]
    curves = find_curves(img)
    edges = find_edges(img, True)
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
            '''rounded_y = rounded_intersection[0]
            rounded_x = rounded_intersection[1]
            found = False
            for y in range(rounded_y - 1, rounded_y + 2):
                for x in range(rounded_x - 1, rounded_x + 2):
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
                                final_intersections.add((y, x))
                                found = True
                                break
                if found:
                    break'''
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
            '''y, x = rounded_intersection
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
                        final_intersections.add((y, x))'''
    final_intersections = list(final_intersections)
    return final_intersections

def find_open_points(img):
    dimensions = img.shape
    height = dimensions[0]
    width = dimensions[1]
    open_points = []
    for x in range(0, width):
        for y in range(0, height):
            if img[y][x] <= 5:
                nearby_points = []
                for x_add in range(-1, 2):
                    for y_add in range(-1, 2):
                        new_x = x + x_add
                        new_y = y + y_add
                        if new_x < 0 or new_x >= width or new_y < 0 or new_y >= height or (x_add == y_add == 0):
                            continue
                        if img[new_y][new_x] <= 5:
                            nearby_points.append((new_y, new_x))
                if len(nearby_points) <= 1:
                    open_points.append((y, x))
    return open_points

def propagation_helper(img, start_point):
    '''print(current_point)
    print(point_list)
    #print(all_point_lists)
    dimensions = img.shape
    height = dimensions[0]
    width = dimensions[1]
    if current_point not in point_list:
        point_list.append(current_point)
    next_points = []
    up = (current_point[0] + 1, current_point[1])
    right = (current_point[0], current_point[1] + 1)
    down = (current_point[0] - 1, current_point[1])
    left = (current_point[0], current_point[1] - 1)
    adjacent_points  = (up, right, down, left)
    for point in adjacent_points:
        y = point[0]
        x = point[1]
        if (y >= 0 and y < height) and (x >= 0 and x < width):
            if img[y][x] <= 5:
                if point not in point_list and point not in next_points:
                    next_points.append(point)
    top_right = (current_point[0] + 1, current_point[1] + 1)
    bottom_right = (current_point[0] - 1, current_point[1] + 1)
    bottom_left = (current_point[0] - 1, current_point[1] - 1)
    top_left = (current_point[0] + 1, current_point[1] - 1)
    diagonal_points = (top_right, bottom_right, bottom_left, top_left)
    for point in diagonal_points:
        y = point[0]
        x = point[1]
        if (y >= 0 and y < height) and (x >= 0 and x < width):
            if img[y][x] <= 5:
                if point not in point_list and point not in next_points:
                    next_points.append(point)
    if next_points == []:
        surrounding_points = adjacent_points + diagonal_points
        for point in surrounding_points:
            y = point[0]
            x = point[1]
            if (y >= 0 and y < height) and (x >= 0 and x < width):
                if img[y][x] <= 5:
                    new_up = (current_point[0] + 1, current_point[1])
                    new_right = (current_point[0], current_point[1] + 1)
                    new_down = (current_point[0] - 1, current_point[1])
                    new_left = (current_point[0], current_point[1] - 1)
                    new_adjacent_points  = (new_up, new_right, new_down, new_left)
                    for new_point in new_adjacent_points:
                        y = new_point[0]
                        x = new_point[1]
                        if (y >= 0 and y < height) and (x >= 0 and x < width):
                            if img[y][x] <= 5:
                                if new_point not in point_list and new_point not in next_points:
                                    next_points.append(new_point)
                    new_top_right = (current_point[0] + 1, current_point[1] + 1)
                    new_bottom_right = (current_point[0] - 1, current_point[1] + 1)
                    new_bottom_left = (current_point[0] - 1, current_point[1] - 1)
                    new_top_left = (current_point[0] + 1, current_point[1] - 1)
                    new_diagonal_points = (new_top_right, new_bottom_right, new_bottom_left, new_top_left)
                    for new_point in new_diagonal_points:
                        y = new_point[0]
                        x = new_point[1]
                        if (y >= 0 and y < height) and (x >= 0 and x < width):
                            if img[y][x] <= 5:
                                if new_point not in point_list and new_point not in next_points:
                                    next_points.append(new_point)
    print("NP: " + str(next_points))
    print("---")
    for existing_point_list in all_point_lists:
        if set(existing_point_list).issubset(set(point_list)):
            all_point_lists.remove(existing_point_list)
    all_point_lists.append(point_list)
    for point in next_points:
        copy_point_list = point_list.copy()
        copy_all_point_lists = all_point_lists.copy()
        new_point_lists = propagation_helper(img, point, copy_point_list, copy_all_point_lists)
        for list in new_point_lists:
            if list not in all_point_lists:
                all_point_lists.append(list)
    return all_point_lists'''

    '''dimensions = img.shape
    height = dimensions[0]
    width = dimensions[1]
    if current_point not in point_list:
        point_list.append(current_point)
    y, x = current_point
    next_segments = [[]]
    first_level_neighbor_coords = ((1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1))
    first_level_marked_neighbors = []
    for coord in first_level_neighbor_coords:
        new_y = y + coord[0]
        new_x = x + coord[1]
        if new_y >= 0 and new_y < height and new_x >= 0 and new_x < width:
            first_level_marked_neighbors.append(img[new_y][new_x] <= 5)
        else:
            first_level_marked_neighbors.append(False)
    first_level_first_index_marked = False
    for index in range(0, 8):
        front_segment = -1
        front_segment = next_segments[len(next_segments) - 1]
        if first_level_marked_neighbors[index] == False:
            if front_segment != []:
                next_segments.append([])
        else:
            coord = first_level_neighbor_coords[index]
            new_y = y + coord[0]
            new_x = x + coord[1]
            if index % 2 == 1:
                front_segment.insert(0, (new_y, new_x))
            else:
                front_segment.append((new_y, new_x))
            if index == 0:
                first_level_first_index_marked = True
            if index == 7 and first_level_first_index_marked and len(next_segments) > 1:
                first_segment = next_segments.pop(0)
                for point in first_segment:
                    point_y, point_x = point
                    relative_coords = (point_y - y, point_x - x)
                    point_index = first_level_neighbor_coords.index(relative_coords)
                    if point_index % 2 == 1:
                        front_segment.insert(0, point)
                    else:
                        front_segment.append(point)
    if next_segments == [[]]:
        second_level_neighbor_coords = ((2, 2), (2, 1), (2, 0), (2, -1), (2, -2), (1, -2), (0, -2), (-1, -2), (-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2), (-1, 2), (0, 2), (1, 2))
        second_level_marked_neighbors = []
        for coord in second_level_neighbor_coords:
            new_y = y + coord[0]
            new_x = x + coord[1]
            if new_y >= 0 and new_y < height and new_x >= 0 and new_x < width:
                second_level_marked_neighbors.append(img[new_y][new_x] <= 5)
            else:
                second_level_marked_neighbors.append(False)
        second_level_first_index_marked = False
        for index in range(0, 8):
            front_segment = -1
            front_segment = next_segments[len(next_segments) - 1]
            if second_level_marked_neighbors[index] == False:
                if front_segment != []:
                    next_segments.append([])
            else:
                coord = second_level_neighbor_coords[index]
                new_y = y + coord[0]
                new_x = x + coord[1]
                front_segment.append((new_y, new_x))
                if index == 0:
                    second_level_first_index_marked = True
                if index == 7 and second_level_first_index_marked and len(next_segments) > 1:
                    first_segment = next_segments.pop(0)
                    for point in first_segment:
                        front_segment.append(point)
    for existing_point_list in all_point_lists:
        if set(existing_point_list).issubset(set(point_list)):
            all_point_lists.remove(existing_point_list)
    all_point_lists.append(point_list)
    for segment in next_segments:
        copy_point_list = point_list.copy()
        copy_all_point_lists = all_point_lists.copy()
        next_point = -1
        for point in segment:
            if point not in point_list:
                next_point = point
        if next_point != -1:
            new_point_lists = propagation_helper(img, next_point, copy_point_list, copy_all_point_lists)
            for list in new_point_lists:
                if list not in all_point_lists:
                    all_point_lists.append(list)
    return all_point_lists'''

    '''dimensions = img.shape
    height = dimensions[0]
    width = dimensions[1]
    if point_tree == -1:
        point_list.append(current_point)
        point_tree = Node(current_point)
    else:
        point_tree = Node(current_point, parent = point_tree)
    y, x = current_point
    next_segments = [[]]
    first_level_neighbor_coords = ((1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1))
    first_level_marked_neighbors = []
    for coord in first_level_neighbor_coords:
        new_y = y + coord[0]
        new_x = x + coord[1]
        if new_y >= 0 and new_y < height and new_x >= 0 and new_x < width:
            first_level_marked_neighbors.append(img[new_y][new_x] <= 5)
        else:
            first_level_marked_neighbors.append(False)
    first_level_first_index_marked = False
    for index in range(0, 8):
        front_segment = -1
        front_segment = next_segments[len(next_segments) - 1]
        if first_level_marked_neighbors[index] == False:
            if front_segment != []:
                next_segments.append([])
        else:
            coord = first_level_neighbor_coords[index]
            new_y = y + coord[0]
            new_x = x + coord[1]
            if index % 2 == 1:
                front_segment.insert(0, (new_y, new_x))
            else:
                front_segment.append((new_y, new_x))
            if index == 0:
                first_level_first_index_marked = True
            if index == 7 and first_level_first_index_marked and len(next_segments) > 1:
                first_segment = next_segments.pop(0)
                for point in first_segment:
                    point_y, point_x = point
                    relative_coords = (point_y - y, point_x - x)
                    point_index = first_level_neighbor_coords.index(relative_coords)
                    if point_index % 2 == 1:
                        front_segment.insert(0, point)
                    else:
                        front_segment.append(point)
    next_points = []
    for segment in next_segments:
        next_point = -1
        for point in segment:
            if point not in point_list:
                next_point = point
                break
        if next_point != -1:
            point_list.append(next_point)
            next_points.append(next_point)
    for point in next_points:
        return_tree = propagation_helper(img, point, point_list, point_tree)
    return (point_tree, point_list)'''

    dimensions = img.shape
    height = dimensions[0]
    width = dimensions[1]
    point_list = [start_point]
    start_tree = Node(start_point)
    tree_stack = [start_tree]
    while len(tree_stack) > 0:
        current_tree = tree_stack.pop(len(tree_stack) - 1)
        current_point = current_tree.name
        y, x = current_point
        next_segments = [[]]
        first_level_neighbor_coords = ((1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1))
        first_level_marked_neighbors = []
        for coord in first_level_neighbor_coords:
            new_y = y + coord[0]
            new_x = x + coord[1]
            if new_y >= 0 and new_y < height and new_x >= 0 and new_x < width:
                first_level_marked_neighbors.append(img[new_y][new_x] <= 5)
            else:
                first_level_marked_neighbors.append(False)
        first_level_first_index_marked = False
        for index in range(0, 8):
            front_segment = -1
            front_segment = next_segments[len(next_segments) - 1]
            if first_level_marked_neighbors[index] == False:
                if front_segment != []:
                    next_segments.append([])
            else:
                coord = first_level_neighbor_coords[index]
                new_y = y + coord[0]
                new_x = x + coord[1]
                if index % 2 == 1:
                    front_segment.insert(0, (new_y, new_x))
                else:
                    front_segment.append((new_y, new_x))
                if index == 0:
                    first_level_first_index_marked = True
                if index == 7 and first_level_first_index_marked and len(next_segments) > 1:
                    first_segment = next_segments.pop(0)
                    for point in first_segment:
                        point_y, point_x = point
                        relative_coords = (point_y - y, point_x - x)
                        point_index = first_level_neighbor_coords.index(relative_coords)
                        if point_index % 2 == 1:
                            front_segment.insert(0, point)
                        else:
                            front_segment.append(point)
        next_points = []
        for segment in next_segments:
            next_point = -1
            for point in segment:
                if point not in point_list:
                    next_point = point
                    break
            if next_point != -1:
                point_list.append(next_point)
                next_points.append(next_point)
        for point in next_points:
            next_tree = Node(point, parent = current_tree)
            tree_stack.append(next_tree)
    return (start_tree, point_list)

def get_ordered_point_tree(img, intersections, open_points): # assume given img has 1 obj
    dimensions = img.shape
    height = dimensions[0]
    width = dimensions[1]
    start_point = -1
    if len(intersections) == 0:
        if len(open_points) == 0:
            for x in range(0, width):
                for y in range(0, height):
                    if img[y][x] <= 5:
                        start_point = (y, x)
                        break
                if start_point != -1:
                    break
        else:
            start_point = open_points[0]
    else:
        start_point = intersections[0]
    return propagation_helper(img, start_point)

def get_set_of_points_from_point(img, start_point, color, pil_img = -1): # 0 = black ; 1 = white ; 2 = nonwhite ; 3 = pseudowhite ; 4 = transparent # 5 = gradient
    '''dimensions = img.shape
    height = dimensions[0]
    width = dimensions[1]
    point_set.add(current_point)
    surrounding_points = []
    up = (current_point[0] + 1, current_point[1])
    right = (current_point[0], current_point[1] + 1)
    down = (current_point[0] - 1, current_point[1])
    left = (current_point[0], current_point[1] - 1)
    adjacent_points  = (up, right, down, left)
    for point in adjacent_points:
        y = point[0]
        x = point[1]
        if (y >= 0 and y < height) and (x >= 0 and x < width):
            if color == 0:
                if img[y][x] <= 5 and (y, x) not in point_set:
                    surrounding_points.append(point)
            elif color == 1:
                if img[y][x] >= 250 and (y, x) not in point_set:
                    surrounding_points.append(point)
    top_right = (current_point[0] + 1, current_point[1] + 1)
    bottom_right = (current_point[0] - 1, current_point[1] + 1)
    bottom_left = (current_point[0] - 1, current_point[1] - 1)
    top_left = (current_point[0] + 1, current_point[1] - 1)
    diagonal_points = (top_right, bottom_right, bottom_left, top_left)
    for point in diagonal_points:
        y = point[0]
        x = point[1]
        if (y >= 0 and y < height) and (x >= 0 and x < width):
            if color == 0:
                if img[y][x] <= 5 and (y, x) not in point_set:
                    surrounding_points.append(point)
            elif color == 1:
                if img[y][x] >= 250 and (y, x) not in point_set:
                    surrounding_points.append(point)
    for point in surrounding_points:
        point_set.add(point)
        point_set = point_set.union(get_set_of_points_from_point(img, point, point_set, color))
    return point_set'''
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

def get_separate_objects(img):
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
                    object_point_lists.append(set_of_points)
    for object_point_list in object_point_lists:
        intersections = ordered_list_of_pts = closed = False
        obj_img = np.zeros((height, width, 3), np.uint8)
        obj_img = ~obj_img
        for point in object_point_list:
            y = point[0]
            x = point[1]
            obj_img[y][x] = 0
        obj_img = cv2.cvtColor(obj_img, cv2.COLOR_BGR2GRAY)
        intersections = find_intersections(obj_img)
        open_points = find_open_points(obj_img)
        (ordered_point_tree, point_list) = get_ordered_point_tree(obj_img, intersections, open_points)
        if len(open_points) == 0:
            closed = True
        object = (intersections, ordered_point_tree, point_list)
        objects.append(object)
    return objects

def get_ordered_lists_of_points_from_tree(tree):
    ordered_lists_of_points = []
    ordered_list_of_points = []
    tree_parser = tree
    while len(tree_parser.children) == 1:
        ordered_list_of_points.append(tree_parser.name)
        tree_parser = tree_parser.children[0]
    ordered_list_of_points.append(tree_parser.name)
    ordered_lists_of_points.append(ordered_list_of_points)
    for subtree in tree_parser.children:
        ordered_lists_of_points += get_ordered_lists_of_points_from_tree(subtree)
    return ordered_lists_of_points

def linear_approximation(img):
    objects = get_separate_objects(img)
    dimensions = img.shape
    height = dimensions[0]
    width = dimensions[1]
    linear_approximation_img = np.zeros((height, width, 3), np.uint8)
    linear_approximation_img = ~linear_approximation_img
    drawn_objects = []
    for object in objects:
        intersections, ordered_point_tree, point_list = object
        ordered_lists_of_points = get_ordered_lists_of_points_from_tree(ordered_point_tree)
        new_intersections = []
        for intersection in intersections:
            surrounding_points = []
            y, x = intersection
            for iter_y in range(-1, 2):
                for iter_x in range(-1, 2):
                    if iter_y != 0 or iter_x != 0:
                        surrounding_points.append((y + iter_y, x + iter_x))
            current_new_intersections = [intersection]
            for point in surrounding_points:
                is_new_intersection = False
                for list_of_points in ordered_lists_of_points:
                    if point in list_of_points:
                        is_alone = True
                        for current_new_intersection in current_new_intersections:
                            if current_new_intersection in list_of_points:
                                is_alone = False
                                break
                        if is_alone:
                            is_new_intersection = True
                        break
                if is_new_intersection:
                    current_new_intersections.append(point)
            new_intersections += current_new_intersections
        all_points = []
        for list in ordered_lists_of_points:
            all_points += list
        segment_length = 10
        possible_length = math.floor(len(all_points)/8)
        if possible_length > segment_length:
            segment_length = possible_length
        linear_approximation_img, drawn_object = linear_approximation_helper(linear_approximation_img, new_intersections, ordered_point_tree, segment_length, [])
        drawn_objects.append(set(drawn_object))
    return linear_approximation_img, drawn_objects

def linear_approximation_helper(linear_approximation_img, intersections, ordered_point_tree, segment_length, drawn_object):
    ordered_list_of_points = []
    tree_parser = ordered_point_tree
    while len(tree_parser.children) == 1:
        ordered_list_of_points.append(tree_parser.name)
        tree_parser = tree_parser.children[0]
    ordered_list_of_points.append(tree_parser.name)
    focus_points_indices = []
    focus_points_indices.append(len(ordered_list_of_points) - 1)
    for intersection in intersections:
        for index in range(0, len(ordered_list_of_points)):
            if ordered_list_of_points[index] == intersection:
                focus_points_indices.append(len(ordered_list_of_points) - 1 - index)
                break
    ordered_list_of_points.reverse()
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
        first_point = ordered_list_of_points[stop_points_indices[index]]
        second_point = ordered_list_of_points[stop_points_indices[index + 1]]
        linear_approximation_img[first_point[0]][first_point[1]] = (0, 0, 0)
        drawn_object.append((first_point[0], first_point[1]))
        if second_point == ordered_list_of_points[len(ordered_list_of_points) - 1]:
            linear_approximation_img[second_point[0]][second_point[1]] = (0, 0, 0)
            drawn_object.append((second_point[0], second_point[1]))
        line = (first_point[1], first_point[0], second_point[1], second_point[0])
        if calculate_slope(line, False) == "UND":
            y_value = first_point[0]
            next_y_value = second_point[0]
            if next_y_value < y_value:
                y_value = second_point[0]
                next_y_value = first_point[0]
            for y in range(y_value, next_y_value):
                linear_approximation_img[y][first_point[1]] = (0, 0, 0)
                drawn_object.append((y, first_point[1]))
        else:
            x_value = first_point[1]
            next_pt_x_value = second_point[1]
            if next_pt_x_value < x_value:
                x_value = second_point[1]
                next_pt_x_value = first_point[1]
            for x in range(x_value, next_pt_x_value):
                y_value = calculate_y_val(line, x)
                next_y_value = calculate_y_val(line, x + 1)
                slope_sign = 1
                if next_y_value < y_value:
                    y_value = calculate_y_val(line, x + 1)
                    next_y_value = calculate_y_val(line, x)
                    slope_sign = -1
                for y in range(math.floor(y_value), math.ceil(next_y_value)):
                    start_x = calculate_x_val(line, y)
                    end_x = calculate_x_val(line, y + 1)
                    if y == math.ceil(next_y_value) - 1 * slope_sign:
                        end_x = x + 1 * slope_sign
                    middle_x = (start_x + end_x)/2
                    middle_y = calculate_y_val(line, middle_x)
                    linear_approximation_img[math.floor(middle_y)][x] = (0, 0, 0)
                    drawn_object.append((math.floor(middle_y), x))
                linear_approximation_img[math.floor(y_value)][x] = (0, 0, 0)
                drawn_object.append((math.floor(y_value), x))
    for child in tree_parser.children:
        linear_approximation_img, drawn_object = linear_approximation_helper(linear_approximation_img, intersections, child, segment_length, drawn_object)
    return linear_approximation_img, drawn_object

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
    prepped_img = prep_for_img_analysis(resized_img)
    cv2.imwrite("initial_prep.jpg", prepped_img)
    lin_app_img, objects = linear_approximation(prepped_img)
    lin_app_img = cv2.cvtColor(lin_app_img, cv2.COLOR_BGR2GRAY)
    pre_find_edges = remove_small_objects(lin_app_img, 60)
    edges = find_edges(lin_app_img, False, img_objects = objects)
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
    init_y = init_point[0]
    init_x = init_point[1]
    shifted_points = []
    for point, object in points:
        shifted_y = point[0] - init_y
        shifted_x = point[1] - init_x
        shifted_points.append((shifted_y, shifted_x))
    max_value = max([max(abs(point[0]), abs(point[1])) for point in shifted_points])
    scale_constant = max_value/3
    final_points = []
    for point in shifted_points:
        final_y = -1 * point[0]/scale_constant
        final_x = point[1]/scale_constant
        final_points.append((final_y, final_x))
    return (final_points, segments)

def get_parameters(text):
    object = create_object(text)
    obj_text, obj_type = object
    name = capitalize_first_letters(text)
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
    term = args[0]
    print("working...")
    get_parameters(term)

if __name__ == "__main__":
    main()
