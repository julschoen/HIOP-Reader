import cv2 as cv
import numpy as np
from sklearn.cluster import KMeans
import tensorflow as tf
from tensorflow.keras.models import load_model
from ImageUtils import ImageUtils
import pytesseract
import re
from Cross import Cross


def __get_cross(time, mask):
    """Detects crosses at a specific time of the color defiened by the mask

    :param time: Time given as interval of width of x axis on which to check for cross
    :param mask: Mask of the wanted color for original Image
    :return Integer tuple if a cross was detect, else None
    """

    # Check for pixels that match searched color
    m = mask[:, time[0] + 1: time[1]]
    x, y = np.where(m == 255)
    y = y + time[0] + 1
    point_cntr = len(x)
    koord = list(zip(y, x))

    if len(koord) > 0:
        # Create MinRect for detected crosses and check area to rule out lines
        x, w, h = cv.minAreaRect(np.array(koord))
        # Check if found points are a cross or just line by
        # examining area of the points
        if 1000 > w[0] * w[1] > 140 and point_cntr > 20 and abs(w[0] - w[1]) < 35:
            x_mean = 0
            y_mean = 0
            for p in koord:
                x_mean += p[0]
                y_mean += p[1]
            x_mean = int(x_mean / len(koord))
            y_mean = int(y_mean / len(koord))
            return x_mean, y_mean
        elif 1000 < w[0] * w[1]:
            # Area too big => there is something on image that is not just
            # line e.g. writing
            return None


def get_value(y_coordinate):
    """
    Calculates the mmHg value of a cross

    :param y_coordinate: y koordinate of detected cross
    :return: mmHg value as int
    """
    if round(y_coordinate) > 245:
        value = round((y_coordinate - 765) * (-1 / 13))
    elif round(y_coordinate) < 245:
        value = round((y_coordinate - 765 + 520) * (-1 / 6.5) + 40)
    else:
        value = 45
    return value


def __get_any_cross(time, mask, other_color_point):
    """
    Detects any color on the Mask to get any cross value
    This is usefull if a cross of other color was deteced but none of specified color

    :param time: Time at which to check for cross as integer tuple
    :param mask: Mask for wanted color of the original image
    :param other_color_point: the other point that was detected for distance checking
    :return: Integer tuple of detected cross
    """
    koord = []
    for y, row in enumerate(mask):
        for x in range(time[0] + 1, time[1]):
            if mask[y][x] == 255:
                koord.append((x, y))

    if len(koord) == 0:
        return 0, 0

    x, w, h = cv.minAreaRect(np.array(koord))
    if 1000 < w[0] * w[1]:
        # If there is something more than one line
        # cluster points and check which is closest to
        # cross already detected
        cluster = KMeans(2).fit(koord)
        dist1 = ((other_color_point[0] - cluster.cluster_centers_[0][0]) ** 2
                 + (other_color_point[1] - cluster.cluster_centers_[0][1]) ** 2) ** 0.5
        dist2 = ((other_color_point[0] - cluster.cluster_centers_[1][0]) ** 2
                 + (other_color_point[1] - cluster.cluster_centers_[1][1]) ** 2) ** 0.5
        if dist1 > dist2:
            new_koord = []
            for i, p in enumerate(koord):
                if cluster.labels_[i] == 1:
                    new_koord.append(p)
        else:
            new_koord = []
            for i, p in enumerate(koord):
                if cluster.labels_[i] == 0:
                    new_koord.append(p)
        koord = new_koord
    x_mean = 0
    y_mean = 0
    for p in koord:
        x_mean += p[0]
        y_mean += p[1]
    x_mean = int(x_mean / len(koord))
    y_mean = int(y_mean / len(koord))
    return x_mean, y_mean


def __predict_number(image, model):
    """ Gets an Image of a number and predicts what number it is

    :param image: Number to be classified
    :return: Result of classification
    """
    _, binary_image = cv.threshold(image, 200, 255, cv.THRESH_BINARY)
    binary_image = cv.resize(binary_image, (35, 35))
    binary_image = (255 - binary_image)
    binary_image = tf.keras.utils.normalize(binary_image, axis=1)
    binary_image = np.expand_dims(binary_image, axis=-1)
    prediction = model.predict([np.array([binary_image])])

    return str(np.argmax(prediction))


def __get_date(date_box):
    """ Reads a date and cleans it up

    :param date_box: Date box of one date
    :return: Date as String
    """

    model = load_model(ImageUtils.resource_path('model.h5'))
    gray_img = cv.cvtColor(date_box, cv.COLOR_BGR2GRAY)
    gray_img = cv.adaptiveThreshold(gray_img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 27, 10)
    ret, thresh = cv.threshold(gray_img, 200, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C)
    cntrs, hirarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    cntrs = ImageUtils.get_number_contours(cntrs, hirarchy)

    read_date = ""
    j = 0
    for i, cnt in enumerate(cntrs):
        last_rect = None

        if i > 0:
            last_rect = cv.boundingRect(cntrs.__getitem__((i - 1)))
        rect = cv.boundingRect(cnt)

        num_box = cv.boxPoints(((rect[0] + int(rect[2] / 2), rect[1] + int(rect[3] / 2)), (rect[2], rect[3]), 0))
        cropped_image = ImageUtils.crop_to_box(gray_img, num_box)

        read_date += __predict_number(cropped_image, model)

        # Checks if there is a dot or a lot of space between last two numbers and adds a dot there accordingly
        if last_rect is not None and (abs(rect[0] - (last_rect[0] + last_rect[2])) > 5):
            read_date = read_date[:-1] + "." + read_date[-1:]
            j += 1

        j += 1

    # Checks if all dots were added if not add one
    split = read_date.split(".")
    if 3 > len(split) > 1 and len(split[1]) > 2:
        read_date = read_date[:-2] + "." + read_date[-2:]
    elif 3 > len(split) > 1 and len(split[0]) > 2:
        read_date = read_date[:2] + "." + read_date[2:]

    # cleaning up dates
    split = read_date.split(".")
    if len(split) == 4 and len(split[3]) == 1:
        read_date = split[0] + "." + split[1] + "." + split[2] + split[3]
        split = read_date.split(".")

    if len(split) == 3:
        if int(split[0]) > 31 and int(split[0][0]) == 7:
            split[0] = '1' + split[0][1]
        if int(split[0]) > 31 and int(split[0][0]) == 9:
            split[0] = '0' + split[0][1]
        if int(split[1]) > 12 and int(split[1][0]) == 7:
            split[1] = '1' + split[1][1]
        if int(split[1]) > 12 and int(split[1][0]) == 9:
            split[1] = '0' + split[1][1]
        if int(split[1]) > 12 and int(split[1][1]) == 7:
            split[1] = split[1][0] + '1'
        if int(split[1]) > 12 and int(split[1][1]) == 9:
            split[1] = split[1][0] + '0'
        read_date = split[0] + "." + split[1] + "." + split[2]

    if read_date.__eq__(""):
        read_date = None

    return read_date


def __get_first_date(date_area):
    """Gets the first date of the date area

    :param date_area: Area of the image containing dates
    :return: Date as String
    """
    roi = date_area[:, 50:198]
    date = __get_date(roi)
    return date


def get_name(image):
    """ Uses Pytesseract and regex to read name from image

    :param image: Image containing the name
    :return: detected name
    """
    typical_words = ['Str', 'Wirzburg', 'Geb', 'Dat', 'Patum', 'rot', 'Diagnose', 'Bayern', 'Gerbrunn', 'Augendruckkurve', 'Univ',
                     'Augenklinik', 'Wirzbur', 'Hausmedikation', 'Mutter', 'Tel', 'Compl', 'Taiolan', 'Therapie',
                     'Rend',
                     'Wirz', 'Hausmedikation', 'Name', 'Diagnose', 'Datum', 'Augenkl', 'Wurzburg', 'Patum', 'Gob']

    name = pytesseract.image_to_string(image)
    name = re.sub(r"\n", " ", name)
    x = re.findall(r'[a-zA-Z]+', name)

    y = ""
    for s in x:
        if len(s) > 2:
            y += s + " "

    x = re.findall("[A-Z][a-z]+", y)
    y = [s for s in x if not typical_words.__contains__(s)]

    return y[0] + ", " + y[1] if len(y) > 0 else 'XX'


def __clean_date(date):
    """ Cleanes up first date and adds
    consecutive dates

    :param date: First read date
    :return: dict containing 6 dates starting with passed date
    """
    if date is None:
        date = "0.0.00"

    date = date.split(".")

    dates = {}

    day = date[0]
    if len(date) > 1:
        month = date[1]
    else:
        month = '00'
    if len(date) > 2 and len(date[2]) > 1:
        year = date[2]
    else:
        year = '0000'

    dates[1] = day + "." + month + "." + year
    # Generates all 6 Dates out of the first date
    for i in range(2, 7):
        if int(day) < 31:
            day = str(int(day) + 1)
        else:
            if int(month) < 12:
                day = '1'
                month = str(int(month) + 1)
            else:
                day = '1'
                month = '1'
                year = str(int(year) + 1)
        dates[i] = day + "." + month + "." + year

    return dates


def read_img(preprocessed_img):
    """ Gets Date and Value information from an Image

    :param preprocessed_img: Image to be read
    :return: csv style String with the detected cross values and dates
    """

    date_area, graph_area = ImageUtils.get_date_graph_area_split(preprocessed_img)
    date = __get_first_date(date_area)
    dates = __clean_date(date)

    crosses = []
    # Lines on Image that need to be checked for colors
    times_lines = []
    lines = ImageUtils.get_time_lines(preprocessed_img)
    dist = 9

    for i, t in enumerate(lines):
        line = t[0]
        date = t[1]
        time = t[2]
        if line[0] - dist > 0 and line[2] + dist < 1200:
            times_lines.append((line[0] - dist, line[2] + dist, date, time))
        else:
            if line[0] - dist < 0:
                times_lines.append((0, line[2] + dist, date, time))
            else:
                times_lines.append((line[0] - dist, 1199, date, time))
    mask_red = ImageUtils.detect_red(graph_area)
    mask_blue = ImageUtils.detect_blue(graph_area)

    for t in times_lines:

        x_blue = __get_cross(t, mask_blue)
        x_red = __get_cross(t, mask_red)

        if x_red is None and x_blue is not None:
            x_red = __get_any_cross(t, mask_red, x_blue)
        elif x_blue is None and x_red is not None:
            # if there is red cross but no blue get any blue marks
            x_blue = __get_any_cross(t, mask_blue, x_red)
        elif x_blue is None and x_red is None:
            continue

        value_blue = get_value(x_blue[1])
        value_red = get_value(x_red[1])

        if value_blue > 60 or value_red > 60:
            continue
        crosses.append(Cross(t[3], t[2], x_red, value_red, x_blue, value_blue))

    return crosses, list(dates.values()), times_lines


def cross_to_csv(dates, crosses, output_path, name):
    """ Takes complete information and writes csv

    :param dates: All 6 read datas as string
    :param crosses: List of all detected crosses
    :param output_path: Path where to save csv
    :param name: Name of the patient, this is also the name of
                the csv
    """
    csv = name + "\nDate,Time,Left Eye mmHg,Right Eye mmHg\n"
    crosses = sorted(crosses, key=lambda x: (x.date, x.time))
    for cross in crosses:
        date = dates[cross.date - 1]
        time = cross.time
        left_eye = cross.red_value
        right_eye = cross.blue_value
        csv += "{},{},{},{}\n".format(date, time, left_eye, right_eye)

    with open(output_path + '/{}.csv'.format(name), 'w+') as f:
        f.write(csv)
