__author__ = 'wonderland'
__version__ = '0.0'

import cv2
from PIL import Image
import matplotlib.pylab as plt
import sys
import os
import argparse
import numpy as np
from data_transform import WLImageFilter
from save2xml import save_xml
import random


def parse_args():
    """
    parse input arguments
    """
    parser = argparse.ArgumentParser(description='Simultate proposal using perspective transform')

    parser.add_argument('--imgs_dir', dest='imgs_dir',
                        help='source images to use',
                        type=str)
    parser.add_argument('--background_dir', dest='background_dir',
                        help='background images dir to add',
                        type=str)
    parser.add_argument('--output', dest='output_dir',
                        help='output dir to save images',
                        type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    if len(dets) == 0:
        print 'Nothing has found'
        plt.draw()
    for det in dets:
        bbox = det[:4]
        score = det[-1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2],
                          bbox[3], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh), fontsize = 14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    plt.show()

def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference

    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def generate_four_points(w, h, rotation_matrix):
    ctr_x = (w-1 + 0.)/2
    ctr_y = (h-1 + 0.)/2
    src_points = np.mat([
        [-ctr_x, ctr_x, ctr_x, -ctr_x],
        [-ctr_y, -ctr_y, ctr_y, ctr_y],
        [0, 0, 0, 0]
    ], dtype=np.float32)
    transformed_points = rotation_matrix * src_points
    return transformed_points


def get_sincos(max_theta):
    theta = np.random.random_integers(-max_theta, max_theta, size=1)
    theta = np.radians(theta)
    c, s = np.cos(theta), np.sin(theta)
    return c, s


def create_rotation_matrix(max_theta_list):
    """
    rotation_z =[[cos -sin 0] rotation_y =[[cos 0  sin]  rotation_x =[[1  0   0]
                [sin cos 0]                [0   1  0]                 [0 cos -sin]
                [0   0  1]]                [-sin 0 cos]]             [0 sin cos]]

    """
    max_theta_x, max_theta_y, max_theta_z = max_theta_list[0:3]
    cx, sx = get_sincos(max_theta_x)
    rotation_x = np.mat([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    cy, sy = get_sincos(max_theta_y)
    rotation_y = np.mat([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    cz, sz = get_sincos(max_theta_z)
    rotation_z = np.mat([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    rotation_matrix = rotation_x * rotation_y * rotation_z
    return rotation_matrix


def transform3d(w, h, scale):

    w_scaled = (w - 1.)/2 * scale
    h_scaled = (h - 1.)/2 * scale
    '''
        points_w = np.array([
        [-w_scaled, -h_scaled, 0],
        [w_scaled, -h_scaled, 0],
        [w_scaled, h_scaled, 0],
        [-w_scaled, h_scaled, 0]
    ], dtype=np.float32)
    '''
    rotation_matrix = create_rotation_matrix([30, 30, 30])

    points_dst = generate_four_points(w_scaled, h_scaled, rotation_matrix)
    trans_length = 0.3 * max(w_scaled, h_scaled)
    trans_length_z = 0.1 * max(w_scaled, h_scaled)
    #trans_length = 20
    t_x, t_y = np.random.random_integers(-trans_length, trans_length, size=2)
    t_z = np.random.random_integers(-trans_length_z, trans_length_z, size=1)
    t_vector = np.mat([[t_x, t_x, t_x, t_x],
                      [t_y, t_y, t_y, t_y],
                      [t_z, t_z, t_z, t_z]])
    points_dst += t_vector

    focal_length = 1.5 * max(w, h)
    points_dst[-1] += focal_length
    points_dst /= points_dst[-1]
    points_dst *= focal_length
    points_dst = points_dst[0:2, :].T
    return points_dst


def warp_src2dst(src, dst_width=480, dst_height=640):
    # consistent ordering of the points is IMPORTANT
    # We specify in top-left, top-right, bottom-right and bottom-left
    src_height, src_width = src.shape[:2]
    src_Tri = np.array([
                [0, 0],
                [src_width-1, 0],
                [src_width-1, src_height-1],
                [0, src_height-1]
                ], dtype=np.float32)
    dst_Tri = transform3d(src_width, src_height, 1.)
    u0 = dst_width/2
    v0 = dst_height/2
    dst_Tri += np.tile(np.array([u0, v0]), (4, 1))
    dst_rect = order_points(dst_Tri)

    (tl, tr, br, bl) = dst_rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    #compute the perspective transform matrix and then apply it
    m_pers = cv2.getPerspectiveTransform(src_Tri, dst_rect)
    warped = cv2.warpPerspective(src, m_pers, (dst_width, dst_height))

    #src_mask = np.ones(src.shape)*255
    src_mask = np.ones((src.shape[0] - 4, src.shape[1]-4, src.shape[2]))*255
    warped_mask = cv2.warpPerspective(src_mask, m_pers, (dst_width, dst_height))

    min_x, min_y = np.maximum(np.min(dst_rect, axis=0), 0)
    max_x, max_y = np.max(dst_rect, axis=0)
    max_x = np.minimum(max_x, dst_width-1)
    max_y = np.minimum(max_y, dst_height-1)
    bounding_box = np.array([[min_x, min_y, max_x-min_x+1, max_y-min_y+1]])
    return warped, warped_mask, np.array(bounding_box)


def warp_simple_src2dst(src, dst_width=480, dst_height=640):
    src_height, src_width = src.shape[:2]
    src_Tri = np.array([
        [0, 0],
        [src_width - 1, 0],
        [src_width - 1, src_height - 1],
        [0, src_height - 1]
    ], dtype=np.float32)
    dst_img_center = [(dst_width-1.)/2, (dst_height-1.)/2]
    dst_center_x = np.random.randint(-(dst_img_center[0]-100), (dst_img_center[0]-100))
    dst_center_y = np.random.randint(-(dst_img_center[1]-200), (dst_img_center[1]-200))
    dst_center = np.array([[dst_center_x+ dst_img_center[0], dst_center_y+dst_img_center[1]]])
    dst_center = np.tile(dst_center, (4, 1))
    scales = [0.8, 1.0, 1.2]
    warped_img_list = []
    dst_rect_list = []
    for scale in scales:
        w_bd, h_bd = [src_width*scale, src_height*scale]
        dst_Tri_x = np.random.randint(-w_bd, w_bd, size=(4, 1)) * 1.
        dst_Tri_y = np.random.randint(-h_bd, h_bd, size=(4, 1)) * 1.
        dst_Tri = np.hstack((dst_Tri_x, dst_Tri_y))
        dst_Tri += dst_center
        dst_rect = order_points(dst_Tri)

        # compute the perspective transform matrix and then apply it
        m_pers = cv2.getPerspectiveTransform(src_Tri, dst_rect)
        warped = cv2.warpPerspective(src, m_pers, (dst_width, dst_height))
        warped_img_list.append(warped)
        dst_rect_list.append(dst_rect)
    return warped_img_list, dst_rect_list


def fusion_two_imgs(background, foreground, mask_fore,  mask_rgb=[0, 0, 0]):

    r_idx, c_idx = np.where((mask_fore[:, :, 0] == mask_rgb[0]) & (mask_fore[:, :, 1] == mask_rgb[1]) \
                            & (mask_fore[:, :, 2] == mask_rgb[2]))
    mask_fore = Image.new('RGB', (foreground.shape[1], foreground.shape[0]), (255, 255, 255))
    mask_fore_array = np.array(mask_fore)
    mask_fore_array[r_idx, c_idx] = [0, 0, 0]
    mask_fore_array = mask_fore_array[:, :, 0]
    mask_fore = Image.fromarray(mask_fore_array.astype(np.uint8))

    background = Image.fromarray(background.astype(np.uint8))
    foreground = Image.fromarray(foreground.astype(np.uint8))
    result = Image.composite(foreground, background, mask_fore)
    return result

if __name__ == '__main__':
    #args = parse_args()

    #print('Called with args:')
    #print(args)

    #imgs_dir = '/home/zyb/cv/simultate_detection_examples/imgs/'
    imgs_dir = '/home/zyb/cv/simultate_detection_examples/imgs_zhenshiming/'
    img_list = [os.path.join(imgs_dir, img_file) for img_file in os.listdir(imgs_dir)]
    img_list = [os.path.join(imgs_dir, str(ind)+'.jpg') for ind in range(6)]

    bg_dir = '/home/zyb/oss_not_hit'
    bg_dir_list = ['/mnt/exhdd/tomorning_dataset/wonderland/raw_data/background/clutter',
                    '/mnt/exhdd/tomorning_dataset/wonderland/raw_data/background/blur',
                   '/mnt/exhdd/tomorning_dataset/wonderland/raw_data/background/scene',
                   '/mnt/exhdd/tomorning_dataset/wonderland/raw_data/human/JPEGImages',
                   '/mnt/exhdd/tomorning_dataset/wonderland/raw_data/background/oss_not_hit']

    bg_img_list = []
    for sub_bg_dir in bg_dir_list:
        sub_dir_list = [os.path.join(sub_bg_dir, img_file) for img_file in os.listdir(sub_bg_dir)]
        bg_img_list += sub_dir_list

    random.shuffle(bg_img_list)

    save_data_dir = '/mnt/exhdd/tomorning_dataset/wonderland/raw_data/save_simulate/Data'
    save_anns_dir = '/mnt/exhdd/tomorning_dataset/wonderland/raw_data/save_simulate/Annotations'
    for kdx in range(3000):
        #for ind in range(len(img_list)):
            ind = kdx % len(img_list)
            wl_filtered = WLImageFilter()
            try:
                src = cv2.imread(img_list[ind])[:, :, (2, 1, 0)]
                dst = cv2.imread(bg_img_list[kdx])[:, :, (2, 1, 0)]
            except Exception as e:
                print 'IO error'
                continue
            dst_height, dst_width = dst.shape[0:2]
            if dst_height < 200 or dst_width < 200:
                continue

            if kdx % 3 == 0:
                src = wl_filtered.enhance_img(src)
            elif kdx % 3 == 1:
                src = wl_filtered.filter_img(src)
            else:
                src = wl_filtered.im_blur_img(src)

            #rect_dict = {'zhenshiming_red': None, 'zhenshiming_green' : None, 'zhenshiming_blue': None}
            warped_img, warped_mask, rect = warp_src2dst(src, dst_width, dst_height)
            show_warped = warped_img
            if ind == 0 or ind == 3:  # blue
                rect_dict = {'zhenshiming_blue': rect}
            elif ind == 1 or ind == 4:  # red
                rect_dict = {'zhenshiming_red': rect}
            elif ind == 2 or ind == 5:  # gree
                rect_dict = {'zhenshiming_green': rect}
            result_img = fusion_two_imgs(dst, show_warped, warped_mask, mask_rgb=[0, 0, 0])
            result_img_save = result_img
            result_img_name = os.path.join(save_data_dir, rect_dict.keys()[0]+'_'+str(ind) + '_' + str(kdx) + '.jpg')
            result_img_save.save(result_img_name, 'JPEG')

            save_xml(result_img_name, rect_dict, save_anns_dir)

            '''
            if True:
                plt.imshow(src)
                plt.show()
                plt.imshow(show_warped)
                plt.show()
                plt.imshow(result_img)
                plt.show()
                print 'Rect:', rect_dict
                vis_detections(result_img, rect_dict.keys()[0], rect_dict.values()[0], thresh=0.5)
            '''

