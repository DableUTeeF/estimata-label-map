import os
import numpy as np
import cv2
import math
import pandas as pd
from matplotlib import pyplot as plt
from PIL import Image
from pillow_heif import register_heif_opener
import easyocr


register_heif_opener()

def check_dupe(i1, i2, i3):
    if i1 == i2:
        return True
    if i1 == i3:
        return True
    if i2 == i3:
        return True
    return False


def angle_between(p1, p2):
    return math.atan2(p1[1] - p2[1], p1[0] - p2[0])

def radian_diff(p1, p2):
    ang1 = np.arctan2(*p1[::-1])
    ang2 = np.arctan2(*p2[::-1])
    return ang1 - ang2


def delta(a, b):
    """
        calculates if angle `a` and angle `b` are far apart
    """
    acceptable_diff = 0.3
    # positive_angle_diff = abs(a - b) > acceptable_diff
    positive_angle_diff = False
    if (abs(a) > 0 and abs(b) > 0) or (abs(a) < 0 and abs(b) < 0):
        positive_angle_diff = abs(a - b) < acceptable_diff
        if positive_angle_diff:
            return False

    if not positive_angle_diff:
        if abs(a) > math.pi - acceptable_diff and abs(b) > math.pi - acceptable_diff:
            diff = (math.pi - abs(a)) + (math.pi - abs(b))
            negative_angle_diff = diff < acceptable_diff
            if negative_angle_diff:
                return False
        if abs(a) < acceptable_diff and abs(b) < acceptable_diff:
            diff = abs(a) + abs(b)
            negative_angle_diff = diff < acceptable_diff
            if negative_angle_diff:
                return False
    return True

def get_all_contours(image):
    """
        todo: better detection algorithm than this would be great
    """
    hue = 80, 130
    sat = 90
    hsvim = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    blue = hsvim.copy()
    blue[(hsvim[..., 0] > hue[0]) & (hsvim[..., 0] < hue[1]) & (hsvim[..., 1] > sat)] = 0
    blue = blue.sum(2)
    blue[blue > 0] = 255
    blue = blue.astype('uint8')

    # hacky way to clean up the mask
    all_contours, _ = cv2.findContours(blue, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(cnt) for cnt in all_contours]
    blackwhite = np.zeros((image.shape[0], image.shape[1]), dtype='uint8')
    for contour, area in zip(all_contours, areas):
        if 50000 > area > 10000:
            blackwhite = cv2.drawContours(blackwhite, [contour], -1, 255, -1)

    all_contours, _ = cv2.findContours(blackwhite, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    all_contours_2 = list(all_contours)
    all_contours = []
    centers = []
    for i, contour in enumerate(all_contours_2):
        ellipse = cv2.fitEllipse(contour)
        width = (contour[..., 0].max() - contour[..., 0].min())
        height = (contour[..., 1].max() - contour[..., 1].min())
        area_sqrt = np.sqrt(cv2.contourArea(contour))
        if max([width, height]) > area_sqrt * 1.5:
            continue
        if 1.25 > ellipse[1][0] / ellipse[1][1] > .65:
            x = width // 2 + contour[..., 0].min()
            y = height // 2 + contour[..., 1].min()
            all_contours.append(contour)
            centers.append((x, y))
    centers = np.array(centers)
    return all_contours, centers


def get_distances(centers):
    distances = np.zeros((len(centers), len(centers)))
    for i in range(len(centers)):
        for j in range(len(centers)):
            distances[i, j] = np.sqrt(((centers[i] - centers[j]) ** 2).sum())
    return distances


def get_line_grid(length_argsort, indices):
    """
        todo: try find parallel lines, see ST022/IMG_8404
    """
    grids = []
    existing = set()
    for idx in length_argsort:
        if len(grids) >= 2:
            break
        line = indices[idx]
        sortedline = tuple(sorted(line))
        if sortedline not in existing:
            grids.append(line)
            existing.add(sortedline)
    return grids


def get_grid_centers(grids, centers):
    grid_centers = []
    for i in range(2):
        grid = grids[i]
        grid_center = []
        for j in range(3):
            grid_center.append(centers[grid[j]])
        grid_centers.append(grid_center)
    grid_centers = np.array(grid_centers)
    return grid_centers


def foundtext(reader, image):
    out = reader.readtext(image)
    for line in out:
        if line[1].startswith('ST'):
            return True, None, line
    out = reader.readtext(cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE))
    for line in out:
        if line[1].startswith('ST'):
            return True, cv2.ROTATE_90_CLOCKWISE, line
    out = reader.readtext(cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE))
    for line in out:
        if line[1].startswith('ST'):
            return True, cv2.ROTATE_90_COUNTERCLOCKWISE, line
    out = reader.readtext(cv2.rotate(image, cv2.ROTATE_180))
    for line in out:
        if line[1].startswith('ST'):
            return True, cv2.ROTATE_180, line
    return False, None, None


def calculate_lines_length(centers, distances):
    total_lengths = []
    indices = []
    for i1 in range(len(centers)):
        for i2 in range(0, len(centers)):
            if i1 == i2:
                continue
            angle12 = angle_between(centers[i1], centers[i2])
            for i3 in range(0, len(centers)):
                if check_dupe(i1, i2, i3):
                    continue
                angle13 = angle_between(centers[i1], centers[i3])
                angle23 = angle_between(centers[i2], centers[i3])
                if delta(angle12, angle13) or delta(angle12, angle23):
                    continue
                total_lengths.append(
                    distances[i1, i2] + distances[i2, i3]
                )
                indices.append((i1, i2, i3))
    indices = np.array(indices)
    return total_lengths, indices

def main():
    reader = easyocr.Reader(['en'])
    srcs = '/media/palm/Data/Estimata/Random 2/{}/'
    dsts = '/media/palm/Data/Estimata/Output 2/{}'

    df = pd.read_excel('/media/palm/Data/Estimata/random2.xlsx')
    all_labels = {}
    curr_st = None
    for i, row in df.iterrows():
        if i < 62:
            continue
        if isinstance(row['ST'], str):
            curr_st = row['ST']
            all_labels[curr_st] = []
        if np.isnan(row.values[1]):
            continue
        for j in [1, 3, 5]:
            all_labels[curr_st].append(row.values[j])

    first_st = ['ST013', 'ST014', 'ST016', 'ST017', 'ST020', 'ST021', 'ST022', 'ST023', 'ST025', 'ST026', 'ST028', 'ST029', 'ST030', 'ST032', 'ST033', 'ST035', 'ST036', 'ST037',
                'ST038', 'ST039', 'ST042', 'ST043', 'ST044', 'ST045', 'ST046', 'ST048', 'ST049', 'ST051', 'ST052', 'ST053', 'ST054', 'ST055', 'ST057', 'ST058', 'ST059', 'ST060',
                'ST061', 'ST062', 'ST064', 'ST065', 'ST067', 'ST068', 'ST069', 'ST070', 'ST071']
    # first_st = ['ST014']
    for st in first_st:
        labels = np.reshape(all_labels[st], (2, 3))
        src = srcs.format(st)
        dst = dsts.format(st)
        os.makedirs(os.path.join(dst, 'grids'), exist_ok=True)
        for file in os.listdir(src):
            if not file.endswith('HEIC'):
                continue
            fname = file.replace('.HEIC', '')
            # if not file == 'IMG_2395.HEIC':
            #     continue
            image = Image.open(os.path.join(src, file))
            ori_width = image.width
            ori_height = image.height
            image = np.array(image)[..., ::-1]
            image = cv2.resize(image, None, None, 0.2, 0.2)
            found, rotatecode, ocr_out = foundtext(reader, image)
            if not found:
                continue
            ocr_bbox = np.array(ocr_out[0])
            ocr_center = np.array((ocr_bbox[:, 0].mean().astype('int'), ocr_bbox[:, 1].mean().astype('int')))
            if rotatecode is not None:
                image = cv2.rotate(image, rotatecode)

            all_contours, centers = get_all_contours(image)
            if len(all_contours) != 6:
                continue

            distances = get_distances(centers)
            total_lengths, indices = calculate_lines_length(centers, distances)
            length_argsort = np.argsort(total_lengths)
            grids = get_line_grid(length_argsort, indices)
            if len(grids) != 2:
                continue
            grid_centers = get_grid_centers(grids, centers)

            rotation_angle = angle_between(grid_centers[0][1], ocr_center)
            if not (2.5 > rotation_angle > 1):
                continue

            line_indice = np.argsort(grid_centers[..., 1].mean(1))
            image2 = image.copy()
            os.makedirs(os.path.join(dst, 'outs', fname), exist_ok=True)
            for i, k in enumerate(line_indice):
                line_centers = grid_centers[k]
                line_idx = indices[k]
                if line_centers[0, 0] > line_centers[2, 0]:
                    line_centers = line_centers[::-1]
                    line_idx = line_idx[::-1]
                for j in range(2):
                    image2 = cv2.line(image2, line_centers[j], line_centers[j+1], (i * 255, 0, 255 - i * 255), 10)

                for j in range(3):
                    blackwhite = np.zeros((image.shape[0], image.shape[1]), dtype='uint8')
                    contour = all_contours[line_idx[j]]
                    label = labels[i, j]
                    center = line_centers[j]
                    blackwhite = cv2.drawContours(blackwhite, [contour], -1, 255, -1)
                    if rotatecode == cv2.ROTATE_90_COUNTERCLOCKWISE:
                        blackwhite = cv2.rotate(blackwhite, cv2.ROTATE_90_CLOCKWISE)
                    elif rotatecode == cv2.ROTATE_90_CLOCKWISE:
                        blackwhite = cv2.rotate(blackwhite, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    elif rotatecode == cv2.ROTATE_180:
                        blackwhite = cv2.rotate(blackwhite, cv2.ROTATE_180)
                    blackwhite = cv2.resize(blackwhite, (ori_width, ori_height), interpolation=cv2.INTER_NEAREST)

                    cv2.imwrite(os.path.join(dst, 'outs', fname, f'{i}_{j}_{label}.png'), blackwhite)
                    image2 = cv2.putText(image2, f'{i * 3 + j}', (center[0] - 15, center[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3, 2)
                    image2 = cv2.putText(image2, f'{label}ml', (center[0] - 15, center[1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3, 2)

            cv2.imwrite(os.path.join(dst, 'grids', file.replace('HEIC', 'jpg')), image2)


if __name__ == '__main__':
    main()
