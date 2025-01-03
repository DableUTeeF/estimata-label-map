import os
import numpy as np
import cv2
import math
import pandas as pd
from matplotlib import pyplot as plt
from PIL import Image
from pillow_heif import register_heif_opener
register_heif_opener()


def check_dupe(i1, i2, i3, i4, i5):
    if i1 == i2:
        return True
    if i1 == i3:
        return True
    if i2 == i3:
        return True
    if i4 is not None:
        if i1 == i4:
            return True
        if i2 == i4:
            return True
        if i3 == i4:
            return True
    if i5 is not None:
        if i2 == i5:
            return True
        if i1 == i5:
            return True
        if i3 == i5:
            return True
        if i4 == i5:
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
        if area > 1000:
            blackwhite = cv2.drawContours(blackwhite, [contour], -1, 255, -1)

    all_contours, _ = cv2.findContours(blackwhite, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    all_contours_2 = list(all_contours)
    all_contours = []
    for i, contour in enumerate(all_contours_2):
        ellipse = cv2.fitEllipse(contour)
        if 1.25 > ellipse[1][0] / ellipse[1][1] > .65:
            all_contours.append(contour)
    return all_contours


def seperate_big_patches(all_contours):
    areas = [cv2.contourArea(cnt) for cnt in all_contours]
    argsort_areas = np.argsort(areas)
    big_indices = sorted([argsort_areas[-1], argsort_areas[-2]])
    big_contours = [all_contours.pop(big_indices[1]), all_contours.pop(big_indices[0])]
    big_patch_centers = []
    for i, contour in enumerate(big_contours):
        x = (contour[..., 0].max() - contour[..., 0].min()) // 2 + contour[..., 0].min()
        y = (contour[..., 1].max() - contour[..., 1].min()) // 2 + contour[..., 1].min()
        big_patch_centers.append((x, y))
    big_patch_centers = np.array(big_patch_centers)
    return all_contours, big_contours, big_patch_centers


def get_center(all_contours):
    centers = []
    for i, contour in enumerate(all_contours):
        x = (contour[..., 0].max() - contour[..., 0].min()) // 2 + contour[..., 0].min()
        y = (contour[..., 1].max() - contour[..., 1].min()) // 2 + contour[..., 1].min()
        centers.append((x, y))
    centers = np.array(centers)
    return centers


def get_distances(centers):
    distances = np.zeros((len(centers), len(centers)))
    for i in range(len(centers)):
        for j in range(len(centers)):
            distances[i, j] = np.sqrt(((centers[i] - centers[j]) ** 2).sum())
    return distances


def calculate_lines_length(centers, distances):
    total_lengths = []
    indices = []
    for i1 in range(len(centers)):
        for i2 in range(0, len(centers)):
            if i1 == i2:
                continue
            angle12 = angle_between(centers[i1], centers[i2])
            for i3 in range(0, len(centers)):
                if check_dupe(i1, i2, i3, None, None):
                    continue
                angle13 = angle_between(centers[i1], centers[i3])
                angle23 = angle_between(centers[i2], centers[i3])
                if delta(angle12, angle13) or delta(angle12, angle23):
                    continue
                for i4 in range(0, len(centers)):
                    if check_dupe(i1, i2, i3, i4, None):
                        continue

                    angle14 = angle_between(centers[i1], centers[i4])
                    angle24 = angle_between(centers[i2], centers[i4])
                    angle34 = angle_between(centers[i3], centers[i4])
                    # todo: change to only the last angle instead of all
                    if delta(angle12, angle14) or delta(angle12, angle24) or delta(angle12, angle34):
                        continue
                    if delta(angle23, angle24) or delta(angle23, angle34):
                        continue
                    for i5 in range(0, len(centers)):
                        if check_dupe(i1, i2, i3, i4, i5):
                            continue
                        if i4 == i5:
                            continue
                        angle15 = angle_between(centers[i4], centers[i5])
                        angle25 = angle_between(centers[i4], centers[i5])
                        angle35 = angle_between(centers[i4], centers[i5])
                        angle45 = angle_between(centers[i4], centers[i5])
                        if delta(angle12, angle15) or delta(angle12, angle25) or delta(angle12, angle35) or delta(angle12, angle45):
                            continue
                        if delta(angle23, angle15) or delta(angle23, angle25) or delta(angle23, angle35) or delta(angle23, angle45):
                            continue
                        if delta(angle34, angle15) or delta(angle34, angle25) or delta(angle34, angle35) or delta(angle34, angle45):
                            continue
                        total_lengths.append(
                            distances[i1, i2] + distances[i2, i3] + distances[i3, i4] + distances[i4, i5]
                        )
                        indices.append((i1, i2, i3, i4, i5))
    return total_lengths, indices


def get_line_grid(length_argsort, indices):
    grids = []
    existing = set()
    for idx in length_argsort:
        if len(grids) >= 10:
            break
        line = indices[idx]
        sortedline = tuple(sorted(line))
        if sortedline not in existing:
            grids.append(line)
            existing.add(sortedline)
    return grids


def get_grid_centers(grids, centers):
    grid_centers = []
    for i in range(10):
        grid = grids[i]
        grid_center = []
        for j in range(5):
            grid_center.append(centers[grid[j]])
        grid_centers.append(grid_center)
    grid_centers = np.array(grid_centers)
    return grid_centers

def main():
    srcs = '/media/palm/Data/Estimata/Random 2/{}/'
    dsts = '/media/palm/Data/Estimata/Random 2/{}/grids'
    df = pd.read_excel('/media/palm/Data/Estimata/random2.xlsx')
    st = []
    curr_st = None
    for i, row in df.iterrows():
        if isinstance(row['ST'], str):
            curr_st = row['ST']
        st.append(curr_st)
    df['ST'] = st

    label1 = np.zeros((5, 5))
    label2 = []
    for i, row in df.iterrows():
        if row['ST'] != 'ST001':
            break
        for j in range(1, 6):
            label1[i, j-1] = row.values[j]
        if not np.isnan(row.values[6]):
            label2.append(row.values[6])

    first_st = ['ST008', 'ST011']
    for st in first_st:
        src = srcs.format(st)
        dst = dsts.format(st)
        os.makedirs(dst, exist_ok=True)
        for file in os.listdir(src):
            if not file.endswith('HEIC'):
                continue
            # if not file == 'IMG_2783.HEIC':
            #     continue
            image = Image.open(os.path.join(src, file))
            image = np.array(image)[..., ::-1]
            image = cv2.resize(image, None, None, 0.2, 0.2)  # todo: need to do something about size too

            all_contours = get_all_contours(image)
            if len(all_contours) < 25:
                print(os.path.join(src, file), len(all_contours))
                continue
            all_contours, big_contours, big_patch_centers = seperate_big_patches(all_contours)
            centers = get_center(all_contours)
            distances = get_distances(centers)
            total_lengths, indices = calculate_lines_length(centers, distances)

            length_argsort = np.argsort(total_lengths)
            grids = get_line_grid(length_argsort, indices)
            if len(grids) != 10:
                continue
            grid_centers = get_grid_centers(grids, centers)
            grid_center_point = (np.mean(grid_centers[..., 0]), np.mean(grid_centers[..., 1]))
            big_patch_center_point = (np.mean(big_patch_centers[..., 0]), np.mean(big_patch_centers[..., 1]))
            rotation_angle = angle_between(grid_center_point, big_patch_center_point)
            if math.pi / 4 * 3 > abs(rotation_angle):
                # right
                pass
            elif abs(rotation_angle) < math.pi / 4:
                # left
                pass
            elif -math.pi / 4 > rotation_angle > -math.pi / 4 * 3:
                # bottom
                pass
            elif math.pi / 4 * 3 > rotation_angle > math.pi / 4:
                # top
                pass

            image2 = image.copy()
            for idx, line in enumerate(grids):
                for i in range(4):
                    # print(radian_diff(centers[line[i]], centers[line[i+1]]))
                    image2 = cv2.line(image2, centers[line[i]], centers[line[i+1]], (idx * 25, 0, 255 - idx * 25), 10)
                    image2 = cv2.line(image2, centers[line[i]], centers[line[i+1]], (i * 80, i * 80, i * 80), 2)

            for i, center in enumerate(centers):
                image2 = cv2.putText(image2, f'{i}', (center[0] - 15, center[1] + 12), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3, 2)

            cv2.imwrite(os.path.join(dst, file.replace('HEIC', 'jpg')), image2)


if __name__ == '__main__':
    main()
