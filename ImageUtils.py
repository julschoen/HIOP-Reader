import imutils
import os
import sys
from pytesseract import image_to_osd
import re
import cv2 as cv
import numpy as np
from math import ceil


class ImageUtils:
    """
    Class providing utility functions for general image processing
    and functionality more specific to this program.
    """

    @staticmethod
    def resource_path(relative_path):
        """ Ensures correct file path

        :param relative_path:
        :return: absolute path
        """
        if hasattr(sys, '_MEIPASS'):
            return os.path.join(sys._MEIPASS, relative_path)
        return os.path.join(os.path.abspath("."), relative_path)

    @staticmethod
    def gray_gaussian_blur(image):
        """ Blurs and greyscales image

        :param image: the image to witch filters are to be applied
        :return: filtered image
        """
        gray_image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
        return cv.GaussianBlur(gray_image, (5, 5), 0)

    @staticmethod
    def get_frame(contours, nth_cnt):
        """ finds the nth biggest contour

        :param contours: all contours of the image
        :param nth_cnt: the nth largest contour
        :return: the nth largest contour
        """
        contour_areas = [(cv.contourArea(cnt), cnt) for cnt in contours]
        contour_areas.sort(key=lambda cnt: cnt[0], reverse=True)
        return contour_areas[nth_cnt][1]

    @staticmethod
    def bounding_box(cnt):
        """ Creates bounding box for contour

        :param cnt: contour to be bounded
        :return: bounding box
        """
        rect = cv.minAreaRect(cnt)
        return cv.boxPoints(rect)

    @staticmethod
    def sort_box(box):
        """ Sorts the corners of the box: lower right, upper right, lower left, upper left

        :param box: Box to be sorted
        :return: sorted box
        """

        sorted_box = sorted(box, key=lambda x: x[0], reverse=True)
        right_corners = [sorted_box[0]] + [sorted_box[1]]
        left_corners = [sorted_box[2]] + [sorted_box[3]]
        sorted_box = sorted(right_corners, key=lambda x: x[1], reverse=True)
        sorted_box += sorted(left_corners, key=lambda x: x[1], reverse=True)

        return sorted_box

    @staticmethod
    def get_contours(gray_image):
        """Returns contours of Image

        :param gray_image: gray scale version of Image
        :return: contours
        """
        ret, thresh = cv.threshold(gray_image, 127, 255, 0, cv.ADAPTIVE_THRESH_GAUSSIAN_C)
        contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        return contours

    @staticmethod
    def preprocess_image(path: str):
        """ Crops to region of interest and staightens image
        as well as guaranteeing correct orientation

        :param path: The image to be processed
        :return:  cropped and straightened image 1200x800
        """

        orig_img = cv.imread(path)
        image = np.copy(orig_img)

        if len(image) > len(image[0]):
            image = imutils.rotate_bound(image, 90)

        image = ImageUtils.rotate(image)

        image = cv.resize(image, (1200, 1000), interpolation=cv.INTER_AREA)

        # Cropping to main area of TDK
        gray_img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        gray_img = cv.adaptiveThreshold(gray_img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 27, 10)
        ret, thresh = cv.threshold(gray_img, 200, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C)
        cntrs, hirarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        cnt = ImageUtils.get_frame(cntrs, 0)

        peri = cv.arcLength(cnt, True)
        approx = cv.approxPolyDP(cnt, 0.04 * peri, True)

        if len(approx) != 4:
            # Shape does not correspond to rectangle, probably not entire paper scanned!
            raise Exception(
                "Cannot detect Graph on Image. Probably the square Grid is cut of please make sure to scan entire "
                "Image.")

        approx = [i[0] for i in approx]
        s_app = ImageUtils.sort_box(approx)

        # Ensures approximation is a rectangle if it is not even close
        if ((abs(s_app[0][0] - s_app[1][0])) > 10) or (abs(s_app[2][0] - s_app[3][0]) > 10) \
                or (abs(s_app[0][1] - s_app[2][1]) > 10) or (abs(s_app[1][1] - s_app[3][1]) > 10):
            hi_hi = 0
            hi_lo = 1200
            lo_lo = 1200
            lo_hi = 0
            for x in approx:
                hi, lo = x
                if hi > hi_hi:
                    hi_hi = hi
                if hi < hi_lo:
                    hi_lo = hi
                if lo > lo_hi:
                    lo_hi = lo
                if lo < lo_lo:
                    lo_lo = lo

            s_app = np.array([
                [np.array([hi_hi, lo_hi])],
                [np.array([hi_hi, lo_lo])],
                [np.array([hi_lo, lo_hi])],
                [np.array([hi_lo, lo_lo])]
            ])

        # Removing box underneath tdk
        pts1 = np.float32(s_app)
        pts2 = np.float32([[1200, 1000], [1200, 0], [0, 1000], [0, 0]])
        matrix = cv.getPerspectiveTransform(pts1, pts2)
        image = cv.warpPerspective(image, matrix, (1200, 1000))

        gray_image = ImageUtils.gray_gaussian_blur(image)
        cntrs = ImageUtils.get_contours(gray_image)
        cnt = ImageUtils.get_frame(cntrs, 0)

        peri = cv.arcLength(cnt, True)
        approx = cv.approxPolyDP(cnt, 0.04 * peri, True)

        if len(approx) != 4:
            # Shape does not correspond to rectangle, probably not entire paper scanned!
            raise Exception(
                "Cannot detect Graph on Image. Probably the square Grid is cut of please make sure to scan entire "
                "Image.")
        box = ImageUtils.bounding_box(approx)
        sorted_box = np.array(ImageUtils.sort_box(box))

        # Ensures that the border to the right is correct
        right_border = (ceil(sorted_box[1][0] / 10)) * 10 + 4
        pts1 = np.float32([[right_border, sorted_box[1][1]], [right_border, 0],
                           [0, sorted_box[3][1]], [0, 0]])
        pts2 = np.float32([[1200, 800], [1200, 0], [0, 800], [0, 0]])
        matrix = cv.getPerspectiveTransform(pts1, pts2)
        return cv.warpPerspective(image, matrix, (1200, 800))

    @staticmethod
    def detect_red(image):
        """ Detects red pixels of an image

        :param image: Image to be searched for red pixels
        :return: Black and white mask for red
        """

        rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        (r, g, _) = (rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2])
        r, g = np.array(r, dtype=int), np.array(g, dtype=int)
        c = np.int0(r - g)
        mask = np.zeros_like(image)
        mask[np.where(c > 15)] = [255, 255, 255]
        return cv.cvtColor(mask, cv.COLOR_RGB2GRAY)

    @staticmethod
    def detect_blue(image):
        """Detects blue pixels of an image

        :param image: Image to be searched for blue pixels
        :return: Black and white mask for blue
        """

        rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        (r, _, b) = (rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2])
        r, b = np.array(r, dtype=int), np.array(b, dtype=int)
        c = np.int0(b - r)
        mask = np.zeros_like(image)
        mask[np.where(c > 20)] = [255, 255, 255]
        return cv.cvtColor(mask, cv.COLOR_RGB2GRAY)

    @staticmethod
    def get_date_graph_area_split(preprocessed_img):
        """ Returns the date area and graph area of the form

        :param preprocessed_img: Image on which date/graph area is supposed to be returned
        :return: new Image of Date and Graph area
        """

        img = np.copy(preprocessed_img)

        date = np.copy(img[49:78, 0:1200])
        img[0:110, 0:1200] = 255
        return date, img

    @staticmethod
    def crop_to_box(image, box):
        """ Crops Image to box

        :param image: Image to be cropped
        :param box: Box defining the region of interest
        :return: Cropped Image
        """
        sorted_box = ImageUtils.sort_box(box)
        pts1 = np.float32(sorted_box)

        x1 = [sorted_box[0][0] - sorted_box[2][0], sorted_box[0][1] - sorted_box[3][1]]
        x2 = [sorted_box[0][0] - sorted_box[2][0], 0]
        x3 = [0, sorted_box[0][1] - sorted_box[3][1]]

        pts2 = np.float32([x1, x2, x3, [0, 0]])

        matrix = cv.getPerspectiveTransform(pts1, pts2)
        return cv.warpPerspective(image, matrix, (x1[0], x1[1]))

    @staticmethod
    def get_number_contours(cntrs, hirarchy):
        """ Finds contours of numbers within contours

        :param cntrs: Countours to be searched for numbers
        :param hirarchy: Hirarchy of those contours
        :return: List of contours detected to be numbers
        """
        cnts = [x for i, x in enumerate(cntrs)
                if hirarchy[0, i, 3] == -1 and (3 < cv.boundingRect(x)[0] < 145 or cv.boundingRect(x)[2] > 3)
                and (3 < cv.boundingRect(x)[1] < 27 or cv.boundingRect(x)[3] > 3)]

        sorted_ctrs = sorted(cnts, key=lambda ctr: cv.boundingRect(ctr)[0])
        relevant_cnt = []
        # Check if number delimiter is dots
        dots = False
        for cnt in sorted_ctrs:
            if cv.contourArea(cnt) < 40:
                for p in cnt:
                    if p[0][0] > 3 and p[0][1] > 3:
                        dots = True
                        break
        if dots:
            for cnt in sorted_ctrs:
                if cv.contourArea(cnt) > 30:
                    relevant_cnt.append(cnt)
        else:
            for cnt in sorted_ctrs:
                if cv.contourArea(cnt) > 30:
                    rect = cv.minAreaRect(cnt)
                    size = rect[1]
                    area = size[0] * size[1]
                    if area > 130 or size[1] > 8:
                        relevant_cnt.append(cnt)
        return relevant_cnt

    @staticmethod
    def match_template(image, template):
        """ Matches a template to check if orientation of Image is korrekt

        :param image: Image to match
        :param template: Template for match region
        :return: True if orientation is Correct else false
        """
        image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
        template = cv.cvtColor(template, cv.COLOR_RGB2GRAY)

        # Initiate ORB detector
        orb = cv.ORB_create(nfeatures=1000)
        # find the keypoints and descriptors with ORB
        kp1, des1 = orb.detectAndCompute(image, None)
        kp2, des2 = orb.detectAndCompute(template, None)
        # create BFMatcher object
        bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        # Match descriptors.
        matches = bf.match(des1, des2)
        # Sort them in the order of their distance.
        matches = sorted(matches, key=lambda match: match.distance)
        # Draw first 10 matches.
        matches = matches[:10]

        list_kp1 = [kp1[mat.queryIdx].pt for mat in matches]

        x = 0
        y = 0
        for p in list_kp1:
            x += p[0]
            y += p[1]
        x = x / len(list_kp1)

        return x < 400

    @staticmethod
    def get_time_lines(image):
        """ Finds the lines within the graph area that represent the different
        times. These lines are extracted and the times are added

        :param image: Image of the graph area
        :return: list of tuples representing lines.
                 Tuple consist of 1. Line, 2. Date, 3. Time
        """
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        low_threshold = 100
        high_threshold = 255
        edges = cv.Canny(gray, low_threshold, high_threshold)

        rho = 1  # distance resolution in pixels of the Hough grid
        theta = np.pi / 180  # angular resolution in radians of the Hough grid
        threshold = 150  # minimum number of votes (intersections in Hough grid cell)
        min_line_length = 50  # minimum number of pixels making up a line
        max_line_gap = 20  # maximum gap in pixels between connectable line segments

        # Run Hough on edge detected image
        # Output "lines" is an array containing endpoints of detected line segments
        lines = cv.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                               min_line_length, max_line_gap)

        if lines is not None:
            lines = sorted(lines, key=lambda x: x[0][0])
        else:
            lines = [[[1, 755, 1197, 755]]]

        prev_line_x = 0

        for line in lines:
            x1, y1, x2, y2 = line[0]
            if 5 > abs(x2 - x1):
                prev_line_x = x1
                break
        time_lines = []
        for i, line in enumerate(lines):
            x1, y1, x2, y2 = line[0]
            if len(time_lines) == 0 and abs(x1 - prev_line_x) > 10 and 5 > abs(x2 - x1):
                prev_line_x = x1
                time_lines.append((x1, y1, x2, y2))
            elif abs(x1 - prev_line_x) > 18 and 5 > abs(x2 - x1):
                prev_line_x = x1
                time_lines.append((x1, y1, x2, y2))

        # Add Times
        final_time_lines = []
        date = 0
        times = [6, 8, 11, 14, 17, 21, 24]
        for i in range(len(time_lines)):
            if i % 7 == 0:
                date += 1
            final_time_lines.append((time_lines[i], date, int(times[i % 7])))

        return final_time_lines

    @staticmethod
    def rotate(im):
        """ Rotates image based on pytesseract osd if it is 180 turned

        :param im: image to be rotated
        :return: either image if it had correct orientation or
                 image rotated by 180 degrees
        """
        rot_data = image_to_osd(im)
        rot = re.search('(?<=Rotate: )\d+', rot_data).group(0)

        angle = float(rot)
        if angle > 0:
            angle = 360 - angle
        # Image is already in landscape make it
        # not rotate back
        if abs(angle - 180) > 20:
            angle = 0

        # rotate the image to deskew it
        (h, w) = im.shape[:2]
        center = (w // 2, h // 2)
        m = cv.getRotationMatrix2D(center, angle, 1.0)
        return cv.warpAffine(im, m, (w, h),
                             flags=cv.INTER_CUBIC, borderMode=cv.BORDER_REPLICATE)

    @staticmethod
    def display_image(image):
        """ Displays image for development

        :param image: image to be displayed
        """
        cv.namedWindow('image', cv.WINDOW_NORMAL)
        cv.resizeWindow('image', 1200, 800)
        cv.imshow('image', image)
        cv.waitKey(0)
