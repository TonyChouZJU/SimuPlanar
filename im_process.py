Iimport cv2
import numpy as np
import matplotlib.pyplot as plt


def project_points(W, H, rxhigh=0.6, ryhigh=0.6, rzhigh=0.6):
    """Function project_points takes the width and height of a box
    and project the object with a randomly chosen view and then return projected points.
    """
    # build box's points
    src = np.array([[0, 0, 0],
                    [W, 0, 0],
                    [0, H, 0],
                    [W, H, 0]], dtype=np.float)
    # half of (W + H) as axis z's center of the camera view
    cz = (W + H) / 2.
    # translate the box to the center
    src[:, 0] -= W / 2.
    src[:, 1] -= H / 2.
    # randomly choose rotation parameters
    rx = np.random.uniform(-rxhigh, rxhigh)
    ry = np.random.uniform(-ryhigh, ryhigh)
    rz = np.random.uniform(-rzhigh, rzhigh)
    rvec = np.array([rx / cz, ry / cz, rz], np.float)  # rotation vector
    tvec = np.array([0, 0, 1], np.float)  # translation vector
    fx = fy = 1.0
    cx = cy = 0.0
    # calibrate camera view
    cameraMatrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, cz]])

    result = cv2.projectPoints(src, rvec, tvec, cameraMatrix, None)
    proj_pts = np.vstack((result[0][0], result[0][1], result[0][2], result[0][3]))

    return proj_pts.astype(dtype=np.float32)


def im_fusion(fg_path=None, bg_path=None, fg_im=None, bg_im=None, allow_border=0):
    """Fuse two images, one as foreground and the other as background. allow_border is the
    number of the pixels that foreground can be outside the background. The function returns
    the fused image and box(x1, y1, x2, y2) of the foreground.
    """
    # image path will be considered first
    if fg_path is not None:
        fg_im = cv2.imread(fg_path)
    if bg_path is not None:
        bg_im = cv2.imread(bg_path)
    assert fg_im is not None and bg_im is not None, \
        "Cannot find foreground image or background image."

    FH, FW, _ = fg_im.shape
    BH, BW, _ = bg_im.shape
    assert BH > FH and BW > FW, \
        "[{} and {}] Invalid size: height {} VS {} and width {} VS {}.".format(bg_path, fg_path, BH, FH, BW, FW)
    # add padding to foreground to match the size of the background
    fg_pad = cv2.copyMakeBorder(fg_im, 0, BH - FH, 0, BW - FW,
                                cv2.BORDER_CONSTANT, value=[0, 0, 0])
    fg_mask = np.zeros((BH, BW, 3), dtype=np.float32)
    fg_mask[: FH, : FW] = 1  # fill foreground part with 1
    # build src points and projection points
    src_pts = np.array([[0, 0],
                        [FW, 0],
                        [0, FH],
                        [FW, FH]], dtype=np.float32)
    proj_pts = project_points(FW, FH)
    # translate projection points
    pxmin, pymin = np.min(proj_pts, axis=0)
    pxmax, pymax = np.max(proj_pts, axis=0)
    PFW, PFH = pxmax - pxmin, pymax - pymin
    X_HIGH = BW - PFW  # maximum shift pixel inside the image
    Y_HIGH = BH - PFH  # maximum shift pixel inside the image
    xmin = np.random.uniform(0 - allow_border, X_HIGH + allow_border)
    ymin = np.random.uniform(0 - allow_border, Y_HIGH + allow_border)
    shiftx = xmin - pxmin
    shifty = ymin - pymin
    proj_pts[:, 0] += shiftx
    proj_pts[:, 1] += shifty
    # get the box
    fg_box = np.array([max(xmin, 0), max(ymin, 0),
                       min(xmin + PFW, BW), min(ymin + PFH, BH)]).astype(dtype=np.float32)
    # get warp matrix
    M = cv2.getPerspectiveTransform(src_pts, proj_pts)
    # get warp image
    fg_warp = cv2.warpPerspective(fg_pad, M, (BW, BH))
    fg_mask = cv2.warpPerspective(fg_mask, M, (BW, BH))
    # fuse images
    fg_inds = np.where(fg_mask == 1)
    bg_im[fg_inds] = fg_warp[fg_inds]

    return bg_im, fg_box


def im_blur(im_path=None, im=None, btype='a'):
    """Blur image with four kinds of blur function: average blur, gaussian blur, median blur and bilateral blur, respectively.
    Default blur function is average blur and all the parameters are those suggested by the official docs.
    """
    if im_path is not None:
        im = cv2.imread(im_path)
    assert im is not None, "Cannot find image."
    blur_handles = {'a': lambda im: cv2.blur(im, (5, 5)),
                    'average': lambda im: cv2.blur(im, (5, 5)),
                    'g': lambda im: cv2.GaussianBlur(im, (5, 5), 0),
                    'gaussian': lambda im: cv2.GaussianBlur(im, (5, 5), 0),
                    'm': lambda im: cv2.medianBlur(im, 5),
                    'median': lambda im: cv2.medianBlur(im, 5),
                    'b': lambda im: cv2.bilateralFilter(im, 9, 75, 75),
                    'bilateral': lambda im: cv2.bilateralFilter(im, 9, 75, 75)}
    return blur_handles[btype](im)


def im_change_bright(im_path=None, im=None,
                     min_alpha=0.5, max_alpha=1.5, min_beta=10, max_beta=50):
    """Change image's lightness by adding a randomly chosen value.
    """
    if im_path is not None:
        im = cv2.imread(im_path)
    assert im is not None, "Cannot find image."
    # randomly choose alpha and beta
    alpha = np.random.uniform(min_alpha, max_alpha)
    beta = np.random.uniform(min_beta, max_beta)
    # change image's brightness
    cv2.convertScaleAbs(im, im, alpha, beta)
    return im


def im_noise(im_path=None, im=None, mean=0, sigma=3):
    """Add noise to the image.
    """
    if im_path is not None:
        im = cv2.imread(im_path)
    assert im is not None, "Cannot find image."
    H, W, C = im.shape
    noise = np.zeros(H*W*C)
    cv2.randn(noise, mean, sigma)
    noise = noise.reshape(H, W, C)
    im = np.uint8(im.astype(dtype=np.float64) + noise)
    return im


def im_fusion_alpha(fg_path=None, bg_path=None, fg_im=None, bg_im=None, allow_border=0):
    # image path will be considered first
    if fg_path is not None:
        fg_im = cv2.imread(fg_path, -1)
    if bg_path is not None:
        bg_im = cv2.imread(bg_path)
    assert fg_im is not None and bg_im is not None, \
        "Cannot find foreground image or background image."

    FH, FW, _ = fg_im.shape
    BH, BW, _ = bg_im.shape
    assert BH > FH and BW > FW, \
        "[{} and {}] Invalid size: height {} VS {} and width {} VS {}.".format(bg_path, fg_path, BH, FH, BW, FW)

    X_HIGH = BW - FW  # maximum shift pixel inside the image
    Y_HIGH = BH - FH  # maximum shift pixel inside the image
    shiftx = np.random.uniform(0 - allow_border, X_HIGH + allow_border)
    shifty = np.random.uniform(0 - allow_border, Y_HIGH + allow_border)

    keep_yinds, keep_xinds = np.where(fg_im[:, :, 3] > 0)

    bg_im[(keep_yinds + np.int32(shifty), keep_xinds + np.int32(shiftx))] = fg_im[:, :, :3][(keep_yinds, keep_xinds)]
    return bg_im

if __name__ == '__main__':
    # im, box = im_fusion('detail.jpg', 'bk.jpg')
    # plt.imshow(im[:, :, (2, 1, 0)])
    # plt.gca().add_patch(
    #     plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
    #                   fill=False, color='blue')
    # )
    # plt.show()
    # im = im_change_bright(im=im)
    # plt.imshow(im[:, :, (2, 1, 0)])
    # plt.show()
    # im = im_blur(im=im, btype='m')
    # plt.imshow(im[:, :, (2, 1, 0)])
    # plt.show()
    # im = im_noise(im=im)
    # plt.imshow(im[:, :, (2, 1, 0)])
    # plt.show()
    im = im_fusion_alpha('font.png', 'bk.jpg')
    plt.imshow(im[:, :, (2,1,0)])
    plt.show()