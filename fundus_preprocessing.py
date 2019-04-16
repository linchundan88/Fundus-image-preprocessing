'''
fundus images preprocess

del_black_or_white  delete borders of fundus images

detect_xyr  using HoughCircles detect circle, if not detected
  suppose the center of the image is the center of the circle.


my_crop_xyz  crot the image based on circle detected

after croped, add some black margin areas,
   so that img aug(random rotate clip) will not delete meaningful edge region

'''

import cv2
import numpy as np
import os
# from imgaug import augmenters as iaa


DEL_PADDING_RATIO = 0.02  #used for del_black_or_white
CROP_PADDING_RATIO = 0.02  #used for my_crop_xyr

# del_black_or_white margin
THRETHOLD_LOW = 7
THRETHOLD_HIGH = 180

# HoughCircles
MIN_REDIUS_RATIO = 0.33
MAX_REDIUS_RATIO = 0.6

def del_black_or_white(img1):
    if img1.ndim == 2:
        img1 = np.expand_dims(img1, axis=-1)

    width, height = (img1.shape[1], img1.shape[0])

    (left, bottom) = (0, 0)
    (right, top) = (img1.shape[1], img1.shape[0])

    padding = int(min(width, height) * DEL_PADDING_RATIO)

    #cv2  height, width

    for i in range(width):
        array1 = img1[:, i, :]  #array1.shape[1]=3 RGB
        if np.sum(array1) > THRETHOLD_LOW * array1.shape[0] * array1.shape[1] and \
                np.sum(array1) < THRETHOLD_HIGH * array1.shape[0] * array1.shape[1]:
            left = i
            break
    left = max(0, left-padding) #留一些空白

    for i in range(width - 1, 0 - 1, -1):
        array1 = img1[:, i, :]
        if np.sum(array1) > THRETHOLD_LOW * array1.shape[0] * array1.shape[1] and \
                np.sum(array1) < THRETHOLD_HIGH * array1.shape[0] * array1.shape[1]:
            right = i
            break
    right = min(width, right + padding)  # 留一些空白

    for i in range(height):
        array1 = img1[i, :, :]
        if np.sum(array1) > THRETHOLD_LOW * array1.shape[0] * array1.shape[1] and \
                np.sum(array1) < THRETHOLD_HIGH * array1.shape[0] * array1.shape[1]:
            bottom = i
            break
    bottom = max(0, bottom - padding)

    for i in range(height - 1, 0 - 1, -1):
        array1 = img1[i, :, :]
        if np.sum(array1) > THRETHOLD_LOW * array1.shape[0] * array1.shape[1] and \
                np.sum(array1) < THRETHOLD_HIGH * array1.shape[0] * array1.shape[1]:
            top = i
            break
    top = min(height, top + padding)

    img2 = img1[bottom:top, left:right, :]

    return img2

def detect_xyr(img_source):
    if isinstance(img_source, str):
        try:
            img = cv2.imread(img_source)
        except:
            # Corrupt JPEG data1: 19 extraneous bytes before marker 0xc4
            raise Exception("image file not found:" + img_source)
        if img is None:
            raise Exception("image file error:" + img_source)
    else:
        img = img_source


    width = img.shape[1]
    height = img.shape[0]

    myMinWidthHeight = min(width, height)  # 最短边长1600 宽和高的最小,并不是所有的图片宽>高 train/22054_left.jpeg 相反

    myMinRadius = round(myMinWidthHeight * MIN_REDIUS_RATIO)
    myMaxRadius = round(myMinWidthHeight * MAX_REDIUS_RATIO)

    '''
    parameters of HoughCircles
    minDist：检测到的圆的中心，（x,y）坐标之间的最小距离。如果minDist太小，则可能导致检测到多个相邻的圆。如果minDist太大，则可能导致很多圆检测不到。
    minDist表示两个圆之间圆心的最小距离
    param1：用于处理边缘检测的梯度值方法。
    param2：cv2.HOUGH_GRADIENT方法的累加器阈值。阈值越小，检测到的圈子越多。
    
    According to our test about fundus images, param2 = 30 is enough, too high will miss some circles
    '''

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, minDist=450, param1=120, param2=32,
                               minRadius=myMinRadius,
                               maxRadius=myMaxRadius)

    (x, y, r) = (0, 0, 0)
    found_circle = False

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        if (circles is not None) and (len(circles == 1)):
            # 有些圆心位置很离谱 25.Hard exudates/chen_liang quan_05041954_clq19540405_557410.jpg
            # x width, y height

            x1, y1, r1 = circles[0]
            if x1 > (2 / 5 * width) and x1 < (3 / 5 * width) \
                    and y1 > (2 / 5 * height) and y1 < (3 / 5 * height):
                x, y, r = circles[0]
                found_circle = True

    if not found_circle:
        # suppose the center of the image is the center of the circle.
        x = img.shape[1] // 2
        y = img.shape[0] // 2

        # get radius  according to the distribution of pixels of the middle line
        temp_x = img[int(img.shape[0] / 2), :, :].sum(1)
        r = int((temp_x > temp_x.mean() / 12).sum() / 2)

    return (found_circle, x, y, r)

def my_crop_xyr(img_source, x, y, r, crop_size=None):
    if isinstance(img_source, str):
        # img_source is a file name
        try:
            image1 = cv2.imread(img_source)
        except:
            # Corrupt JPEG data1: 19 extraneous bytes before marker 0xc4
            raise Exception("image file not found:" + img_source)
    else:
        image1 = img_source

    if image1 is None:
        raise Exception("image file error:" + img_source)

    original_width = image1.shape[1]
    original_height = image1.shape[0]

    (image_height, image_width) = (image1.shape[0], image1.shape[1])

    #  裁剪图像 根据半径裁减  判断高是否够  防止超过边界,所以0和width
    # 第一个是高,第二个是宽  r是半径

    img_padding = int(min(original_width, original_height) * CROP_PADDING_RATIO)

    image_left = int(max(0, x - r - img_padding))
    image_right = int(min(x + r + img_padding, image_width - 1))
    image_bottom = int(max(0, y - r - img_padding))
    image_top = int(min(y + r + img_padding, image_height - 1))

    if image_width >= image_height:  # 图像宽比高大
        if image_height >= 2 * (r + img_padding):
            # 图像比圆大
            image1 = image1[image_bottom: image_top, image_left:image_right]
        else:
            # 因为图像高度不够,图像被垂直剪切
            image1 = image1[:, image_left:image_right]
    else:  # 图像宽比高小
        if image_width >= 2 * (r + img_padding):
            # 图像比圆大
            image1 = image1[image_bottom: image_top, image_left:image_right]
        else:
            image1 = image1[image_bottom:image_top, :]

    if crop_size is not None:
        image1 = cv2.resize(image1, (crop_size, crop_size))

    return image1

def add_black_margin(img_source, add_black_pixel_ratio = 0.05):
    if isinstance(img_source, str):
        # img_source is a file name
        try:
            image1 = cv2.imread(img_source)
        except:
            # Corrupt JPEG data1: 19 extraneous bytes before marker 0xc4
            raise Exception("image file not found:" + img_source)
    else:
        image1 = img_source

    if image1 is None:
        raise Exception("image file error:" + img_source)

    height, width = image1.shape[:2]

    add_black_pixel = int(min(height, width) * add_black_pixel_ratio)

    img_h = np.zeros((add_black_pixel, width, 3))
    img_v = np.zeros((height + add_black_pixel*2, add_black_pixel, 3))

    image1 = np.concatenate((img_h, image1, img_h), axis=0)
    image1 = np.concatenate((img_v, image1, img_v), axis=1)

    return image1



def my_preprocess(img_source, crop_size, train_or_valid='train', img_file_dest=None):
    if isinstance(img_source, str):
        try:
            image1 = cv2.imread(img_source)
        except:
            # Corrupt JPEG data1: 19 extraneous bytes before marker 0xc4
            raise Exception("image file not found:" + img_source)
    else:
        image1 = img_source

    if image1 is None:  #file not exists or orther errors
        raise Exception("image file error:" + img_source)

    image1 = del_black_or_white(image1)

    min_width_height = min(image1.shape[0], image1.shape[1])

    if min_width_height < 100: # image too small
        return None

    #image too big, resize
    image_size_before_hough = crop_size * 2
    if min_width_height > image_size_before_hough:
        crop_ratio = image_size_before_hough / min_width_height
        # fx、fy: Scaling factor  x axis and y axis
        image1 = cv2.resize(image1, None, fx=crop_ratio, fy=crop_ratio)

    (found_circle, x, y, r) = detect_xyr(image1)

    if train_or_valid == 'train':
        image1 = my_crop_xyr(image1, x, y, r)
        # add some black margin, for fear that duing img aug(random rotate crop) delete useful areas
        image1 = add_black_margin(image1, add_black_pixel_ratio=0.07)
        image1 = cv2.resize(image1, (crop_size, crop_size))
    else:
        image1 = my_crop_xyr(image1, x, y, r, crop_size)

    # if specify img_file_dest,  save to file
    if img_file_dest is not None:
        if not os.path.exists(os.path.dirname(img_file_dest)):
            os.makedirs(os.path.dirname(img_file_dest))

        cv2.imwrite(img_file_dest, image1)

    return image1


'''
# 预测的时候 multi crop
def multi_crop(img_source, gen_times=5, add_black=True):
    if isinstance(img_source, str):
        # img_source is a file name
        try:
            image1 = cv2.imread(img_source)
        except:
            # Corrupt JPEG data1: 19 extraneous bytes before marker 0xc4
            raise Exception("image file not found:" + img_source)
    else:
        image1 = img_source

    if image1 is None:
        raise Exception("image file error:" + img_source)

    if add_black:
        image1 = add_black(img_source)

    list_image = [image1]

    # sometimes = lambda aug: iaa.Sometimes(0.96, aug)
    seq = iaa.Sequential([
        iaa.Crop(px=(0, min(image1.shape[0], image1.shape[1]) // 20)),  # crop images from each side by 0 to 16px (randomly chosen)
        # iaa.GaussianBlur(sigma=(0, 3.0)),  # blur images with a sigma of 0 to 3.0,
        # iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),  # sharpen images
        # sometimes(iaa.Crop(percent=(0, 0.1))),  # crop images by 0-10% of their height/width
        # shuortcut for CropAndPad

        # improve or worsen the contrast  If PCH is set to true, the process happens channel-wise with possibly different S.
        # sometimes1(iaa.ContrastNormalization((0.9, 1.1), per_channel=0.5), ),
        # change brightness of images (by -5 to 5 of original value)
        # sometimes1(iaa.Add((-6, 6), per_channel=0.5),),
        # sometimes(iaa.Affine(
        #     # scale={"x": (0.92, 1.08), "y": (0.92, 1.08)},
        #     # scale images to 80-120% of their size, individually per axis
        #     # Translation Shifts the pixels of the image by the specified amounts in the x and y directions
        #     translate_percent={"x": (-0.08, 0.08), "y": (-0.06, 0.06)},
        #     # translate by -20 to +20 percent (per axis)
        #     rotate=(0, 360),  # rotate by -45 to +45 degrees
        #     # shear=(-16, 16),  # shear by -16 to +16 degrees
        #     # order=[0, 1],  # use nearest neighbour or bilinear interpolation (fast)
        #     # cval=(0, 255),  # if mode is constant, use a cval between 0 and 255
        #     # mode=ia.ALL  # use any of scikit-image's warping modes (see 2nd image from the top for examples)
        # )),
    ])

    img_results = []

    for i in range(gen_times):
        images_aug = seq.augment_images(list_image)
        img_results.append(images_aug[0])

    return img_results
'''


# simple demo code
if __name__ == '__main__':
    img_file = '/tmp1/img2.jpg'
    if os.path.exists(img_file):
        img_processed = my_preprocess(img_file, crop_size=384)
        cv2.imwrite('/tmp1/tmp2_preprocess.jpg', img_processed)
        print('OK')
    else:
        print('file not exists!')


