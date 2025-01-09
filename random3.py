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
    acceptable_diff = 0.4
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
        if contour.shape[0] < 5:
            continue
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
    for i in range(2):
        grid = grids[i]
        grid_center = []
        for j in range(5):
            grid_center.append(centers[grid[j]])
        grid_centers.append(grid_center)
    grid_centers = np.array(grid_centers)
    return grid_centers


def compare_center(line_centers, mode):
    if mode == 0:  # big patch at the right
        return line_centers[0, 0] < line_centers[2, 0]
    elif mode == 1:  # big patch at the left
        return line_centers[0, 0] > line_centers[2, 0]
    elif mode == 1:  # big patch at the top
        return line_centers[0, 1] < line_centers[2, 1]
    else:  # big patch at the bottom
        return line_centers[0, 1] > line_centers[2, 1]


def drawgrid(line_indice, image, indices, all_contours, labels, folder, outputs, grid_centers, mode):
    image2 = image.copy()
    for i, k in enumerate(line_indice):
        line_centers = grid_centers[k]
        line_idx = indices[k]
        if compare_center(line_centers, mode):
            line_centers = line_centers[::-1]
            line_idx = line_idx[::-1]
        for j in range(4):
            image2 = cv2.line(image2, line_centers[j], line_centers[j + 1], (i * 255, 0, 255 - i * 255), 10)
        for j in range(5):
            contour = all_contours[line_idx[j]]
            if len(contour.shape) == 3:
                contour = contour[:, 0]
            label = labels[folder]['small'][i][j]
            center = line_centers[j]
            image2 = cv2.putText(image2, f'{i * 5 + j}', (center[0] - 15, center[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3, 2)
            image2 = cv2.putText(image2, f'{label}ml', (center[0] - 15, center[1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3, 2)

            outputs['type'].append('gauze')
            outputs['row'].append(i)
            outputs['column'].append(j)
            outputs['ml'].append(label)
            outputs['x'].append((contour[:, 0].min()) * 5)  # todo: keep ratio here
            outputs['w'].append((contour[:, 0].max() - contour[:, 1].min()) * 5)
            outputs['y'].append((contour[:, 1].min()) * 5)
            outputs['h'].append((contour[:, 1].max() - contour[:, 1].min()) * 5)
    return outputs, image2


def get_path_indices(big_patch_centers, vertical, mode):
    if mode == 0:
        if vertical:
            patch_indice = np.argsort(big_patch_centers[..., 1])[::-1]
        else:
            patch_indice = np.argsort(big_patch_centers[..., 0])[::-1]
    elif mode == 1:
        if vertical:
            patch_indice = np.argsort(big_patch_centers[..., 1])
        else:
            patch_indice = np.argsort(big_patch_centers[..., 0])
    elif mode == 2:
        if vertical:
            patch_indice = np.argsort(big_patch_centers[..., 0])[::-1]
        else:
            patch_indice = np.argsort(big_patch_centers[..., 1])
    else:
        if vertical:
            patch_indice = np.argsort(big_patch_centers[..., 0])
        else:
            patch_indice = np.argsort(big_patch_centers[..., 1])[::-1]
    return patch_indice


def draw_big_patches(image2, big_contours, big_patch_centers, labels, folder, outputs, vertical, mode):
    patch_indice = get_path_indices(big_patch_centers, vertical, mode)
    for i in patch_indice:
        contour = big_contours[i]
        if len(contour.shape) == 3:
            contour = contour[:, 0]
        label = labels[folder]['big'][i]
        center = big_patch_centers[i]
        image2 = cv2.putText(image2, f'{i}', (center[0] - 15, center[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3, 2)
        image2 = cv2.putText(image2, f'{label}ml', (center[0] - 15, center[1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3, 2)
        if vertical:
            row = i
            col = 0
        else:
            row = 0
            col = i
        outputs['type'].append('swab')
        outputs['row'].append(row)
        outputs['column'].append(col)
        outputs['ml'].append(label)
        outputs['x'].append((contour[:, 0].min()) * 5)  # todo: keep ratio here
        outputs['w'].append((contour[:, 0].max() - contour[:, 1].min()) * 5)
        outputs['y'].append((contour[:, 1].min()) * 5)
        outputs['h'].append((contour[:, 1].max() - contour[:, 1].min()) * 5)
    return outputs, image2


def main():
    srcs = '/mnt/d/work/estimata/Random 3/{}/{}'
    dsts = '/mnt/d/work/estimata/Output 3/{}/{}'
    # df = pd.read_excel('/mnt/d/work/estimata/random3.xlsx')
    # st = []
    # curr_st = None
    # for i, row in df.iterrows():
    #     if isinstance(row['ST'], str):
    #         curr_st = row['ST']
    #     st.append(curr_st)
    # df['ST'] = st
    #
    # label1 = np.zeros((5, 5))
    # label2 = []
    # for i, row in df.iterrows():
    #     if row['ST'] != 'ST001':
    #         break
    #     for j in range(1, 6):
    #         label1[i, j-1] = row.values[j]
    #     if not np.isnan(row.values[6]):
    #         label2.append(row.values[6])

    stations = ['STATION 1']
    for st in stations:
        df = pd.read_excel('/mnt/d/work/estimata/random3.xlsx', sheet_name=st)
        labels = {}
        for i in range(len(df) // 2):
            if np.isnan(df.values[i * 2, 5]):
                continue
            folder = os.path.join(df.values[i * 2, 1], df.values[i * 2, 2])
            big = [df.values[i * 2, 3]]
            if df.values.shape[1] == 10:
                big.append(df.values[i * 2, 4])
                small = [df.values[i * 2, 5:10], df.values[i * 2 + 1, 5:10]]
                vertical = False
            elif df.values.shape[1] == 9:
                big.append(df.values[i * 2 + 1, 3])
                small = [df.values[i * 2, 4:9], df.values[i * 2 + 1, 4:9]]
                vertical = True
            elif df.values.shape[1] == 11:
                big = [df.values[i * 2, 4], df.values[i * 2, 5]]
                small = [df.values[i * 2, 6:11], df.values[i * 2 + 1, 6:11]]
                vertical = False
            else:
                raise Exception
            labels[folder] = {'big': big, 'small': small}

        for folder in labels:
            src = srcs.format(st, folder)
            dst = dsts.format(st, folder)
            os.makedirs(dst, exist_ok=True)
            for file in os.listdir(src):
                if not file.endswith('jpeg'):
                    continue
                if not file == '1924IMG_3484.jpeg':
                    continue
                fname = file.replace('.jpeg', '')
                image = Image.open(os.path.join(src, file))
                ori_width = image.width
                ori_height = image.height
                image = np.array(image)[..., ::-1]
                image = cv2.resize(image, None, None, 0.2, 0.2)

                all_contours = get_all_contours(image)
                if len(all_contours) < 10:
                    print(os.path.join(src, file), len(all_contours))
                    continue
                all_contours, big_contours, big_patch_centers = seperate_big_patches(all_contours)
                centers = get_center(all_contours)
                distances = get_distances(centers)
                total_lengths, indices = calculate_lines_length(centers, distances)

                length_argsort = np.argsort(total_lengths)
                grids = get_line_grid(length_argsort, indices)
                if len(grids) != 2:
                    continue
                grid_centers = get_grid_centers(grids, centers)
                grid_center_point = (np.mean(grid_centers[..., 0]), np.mean(grid_centers[..., 1]))
                big_patch_center_point = (np.mean(big_patch_centers[..., 0]), np.mean(big_patch_centers[..., 1]))
                rotation_angle = angle_between(grid_center_point, big_patch_center_point)
                outputs = {'type': [], 'row': [], 'column': [], 'ml': [], 'x': [], 'y': [], 'w': [], 'h': []}
                if math.pi / 4 * 3 < abs(rotation_angle):
                    line_indice = np.argsort(grid_centers[..., 1].mean(1))[::-1]
                    outputs, image2 = drawgrid(line_indice, image, indices, all_contours, labels, folder, outputs, grid_centers, mode=0)
                    outputs, image2 = draw_big_patches(image2, big_contours, big_patch_centers, labels, folder, outputs, vertical, mode=0)
                elif abs(rotation_angle) < math.pi / 4:
                    line_indice = np.argsort(grid_centers[..., 1].mean(1))
                    outputs, image2 = drawgrid(line_indice, image, indices, all_contours, labels, folder, outputs, grid_centers, mode=1)
                    outputs, image2 = draw_big_patches(image2, big_contours, big_patch_centers, labels, folder, outputs, vertical, mode=1)

                elif math.pi / 4 < abs(rotation_angle) < math.pi / 4 * 3:
                    line_indice = np.argsort(grid_centers[..., 0].mean(1))[::-1]
                    outputs, image2 = drawgrid(line_indice, image, indices, all_contours, labels, folder, outputs, grid_centers, mode=2)
                    outputs, image2 = draw_big_patches(image2, big_contours, big_patch_centers, labels, folder, outputs, vertical, mode=2)
                else:
                    line_indice = np.argsort(grid_centers[..., 0].mean(1))
                    outputs, image2 = drawgrid(line_indice, image, indices, all_contours, labels, folder, outputs, grid_centers, mode=3)
                    outputs, image2 = draw_big_patches(image2, big_contours, big_patch_centers, labels, folder, outputs, vertical, mode=3)
                cv2.imwrite(os.path.join(dst, file.replace('HEIC', 'jpg')), image2)


if __name__ == '__main__':
    main()
