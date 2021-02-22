from PyQt5 import QtCore, QtGui, QtWidgets
import sys
from PyQt5.QtGui import QImage
import DataReader as reader
from ImageUtils import ImageUtils
import numpy as np
import cv2 as cv
from Cross import Cross
import os
from pdf2image import pdf2image


class UiMainWindow(object):
    """
    PyQT GUI for Glaukoma Reader
    """

    def __init__(self):
        self.name = ''
        self.selected_cross = 0
        self.file_loaded = False
        self.file_list = []
        self.popup = None
        self.crosses = []
        self.dates = []
        self.lines = []
        self.img = None
        self.mainWindow = None

    def setup_ui(self, main_window):
        """ Populates mainwindow with all objects

        :param main_window: main window to add objects to
        """
        self.mainWindow = main_window
        main_window.setObjectName("mainWindow")
        main_window.setFixedSize(900, 750)
        self.centralwidget = QtWidgets.QWidget(main_window)
        self.centralwidget.setObjectName("centralwidget")

        self.verticalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(0, 0, 900, 750))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")

        self.img_label = QtWidgets.QLabel(self.verticalLayoutWidget)
        image_profile = QtGui.QImage(ImageUtils.resource_path('0.jpg'))  # QImage object
        image_profile = image_profile.scaled(900, 750, aspectRatioMode=QtCore.Qt.KeepAspectRatio,
                                             transformMode=QtCore.Qt.SmoothTransformation)
        # To scale image for example and keep its Aspect Ration
        self.img_label.setPixmap(QtGui.QPixmap.fromImage(image_profile))
        self.img_label.mousePressEvent = self.image_clicked

        self.verticalLayout.addWidget(self.img_label)

        self.name_area = QtWidgets.QHBoxLayout()
        self.name_area.setContentsMargins(10, 0, 10, 0)
        self.name_label = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.name_label.setText('Edit detected Name: ')
        self.name_area.addWidget(self.name_label)

        self.name_line_edit = QtWidgets.QLineEdit(self.verticalLayoutWidget)
        self.name_line_edit.setFixedWidth(450)
        self.name_line_edit.setText(self.name)
        self.name_area.addWidget(self.name_line_edit)

        self.verticalLayout.addLayout(self.name_area)

        self.date_update_area = QtWidgets.QHBoxLayout()
        self.date_update_area.setContentsMargins(10, 0, 10, 0)
        self.date_update_label = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.date_update_area.addWidget(self.date_update_label)

        self.date_updateBtn = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.date_updateBtn.setText("Update all Dates")
        self.date_updateBtn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.date_updateBtn.clicked.connect(self.update_date)
        self.date_update_area.addWidget(self.date_updateBtn)

        self.verticalLayout.addLayout(self.date_update_area)

        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setContentsMargins(10, 0, 10, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")

        self.openBtn = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.openBtn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.openBtn.setObjectName("openBtn")
        self.openBtn.clicked.connect(self.open_file)
        self.horizontalLayout.addWidget(self.openBtn)

        self.openFolderBtn = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.openFolderBtn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.openFolderBtn.setObjectName("openFolderBtn")
        self.openFolderBtn.clicked.connect(self.open_folder)
        self.horizontalLayout.addWidget(self.openFolderBtn)

        self.saveBtn = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.saveBtn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.saveBtn.setObjectName("saveBtn")
        self.saveBtn.clicked.connect(self.save_file)
        self.horizontalLayout.addWidget(self.saveBtn)

        self.verticalLayout.addLayout(self.horizontalLayout)
        main_window.setCentralWidget(self.centralwidget)

        self.retranslate_ui(main_window)
        QtCore.QMetaObject.connectSlotsByName(main_window)

    def retranslate_ui(self, main_window):
        """ QT Standart function

        :param main_window: main window of the GUI
        """
        _translate = QtCore.QCoreApplication.translate
        main_window.setWindowTitle(_translate("mainWindow", "Glaukoma Reader"))
        self.openBtn.setText(_translate("mainWindow", "Open File"))
        self.openFolderBtn.setText(_translate("mainWindow", "Open Folder"))
        self.saveBtn.setText(_translate("mainWindow", "Save File"))

    def image_clicked(self, event):
        """ Handles interaction with the image

        :param event: QEvent
        """
        if self.file_loaded:
            if event.type() == QtCore.QEvent.MouseButtonPress:
                if event.button() == QtCore.Qt.LeftButton:
                    y = (event.pos().y() * 950) / 750
                    x = (event.pos().x() * 1200) / 900
                    val = reader.get_value(y) - 1

                    if y < 740:
                        exists = False
                        for i, cross in enumerate(self.crosses):
                            red = cross.red_pos
                            blue = cross.blue_pos
                            if (abs(cross.red_value - val) < 2 and abs(red[0] - x) < 15) or (
                                    abs(cross.blue_value - val) < 2 and abs(blue[0] - x) < 15):
                                self.selected_cross = i
                                self.set_image(self.img, i)
                                exists = True
                        if not exists:
                            self.set_image(self.img)
                    else:
                        for i in range(0, 7):
                            line = self.lines[i]
                            if line[0] < int(x) < line[1]:
                                self.set_image(self.img)
                elif event.button() == QtCore.Qt.RightButton:
                    y = (event.pos().y() * 950) / 750
                    x = (event.pos().x() * 1200) / 900
                    val = reader.get_value(y) - 1

                    if y < 740:
                        exists = False
                        for i, cross in enumerate(self.crosses):
                            red = cross.red_pos
                            blue = cross.blue_pos
                            if (abs(cross.red_value - val) < 2 and abs(red[0] - x) < 15) or (
                                    abs(cross.blue_value - val) < 2 and abs(blue[0] - x) < 15):
                                self.selected_cross = i
                                self.set_image(self.img, i)
                                exists = True
                                self.edit_cross(event)
                        if not exists:
                            self.set_image(self.img)
                            self.add_cross(event)

    def edit_cross(self, event):
        """ Opens cross editor oder deletes cross

        :param event: QTEvent coming from image clicked
        """
        context_menu = QtWidgets.QMenu(self.mainWindow)
        edit_action = context_menu.addAction("Edit Cross")
        del_action = context_menu.addAction("Delete Cross")

        action = context_menu.exec_(self.mainWindow.mapToGlobal(event.pos()))

        if action == del_action:
            self.crosses.pop(self.selected_cross)
            self.set_image(self.img)
        elif action == edit_action:
            self.edit_popup()

    def edit_popup(self):
        """ Sets up and presents cross editing Pop-Up
        """
        if self.file_loaded:
            self.popup = QtWidgets.QMainWindow()
            self.popup.setFixedSize(200, 200)
            self.popwidget = QtWidgets.QWidget(self.popup)
            self.popwidget.setObjectName("pop_central")
            self.pop_verticalWidget = QtWidgets.QWidget(self.popwidget)
            self.pop_verticalWidget.setGeometry(QtCore.QRect(0, 0, 200, 200))
            self.pop_verticalWidget.setObjectName("pop_verticalWidget")

            self.pop_v = QtWidgets.QVBoxLayout(self.pop_verticalWidget)
            self.pop_v.setContentsMargins(10, 10, 10, 10)
            self.pop_v.setObjectName("verticalLayout")

            self.pop_time_area = QtWidgets.QHBoxLayout()
            self.time_label = QtWidgets.QLabel()
            self.time_label.setText("Time:")
            self.pop_time_area.addWidget(self.time_label)
            self.time_text = QtWidgets.QLineEdit()

            self.time_text.setText(str(self.crosses[self.selected_cross].time))
            self.pop_time_area.addWidget(self.time_text)
            self.pop_v.addLayout(self.pop_time_area)

            self.red_cross = QtWidgets.QHBoxLayout()
            self.red_label = QtWidgets.QLabel()
            self.red_label.setText("Red cross value:")
            self.red_cross.addWidget(self.red_label)
            self.red_text = QtWidgets.QLineEdit()
            self.red_text.setText(str(self.crosses[self.selected_cross].red_value))
            self.red_cross.addWidget(self.red_text)
            self.pop_v.addLayout(self.red_cross)

            self.blue_cross = QtWidgets.QHBoxLayout()
            self.blue_label = QtWidgets.QLabel()
            self.blue_label.setText("Blue cross value:")
            self.blue_cross.addWidget(self.blue_label)
            self.blue_text = QtWidgets.QLineEdit()
            self.blue_text.setText(str(self.crosses[self.selected_cross].blue_value))
            self.blue_cross.addWidget(self.blue_text)
            self.pop_v.addLayout(self.blue_cross)

            self.pop_btn_area = QtWidgets.QHBoxLayout()
            self.pop_btn_area.setObjectName("button_layout")

            self.close_btn = QtWidgets.QPushButton()
            self.close_btn.setText("Quit")
            self.close_btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
            self.close_btn.clicked.connect(self.close_popup)
            self.pop_btn_area.addWidget(self.close_btn)

            self.new_date_btn = QtWidgets.QPushButton()
            self.new_date_btn.setText("Save")
            self.new_date_btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
            self.new_date_btn.clicked.connect(self.save_edit_cross)
            self.pop_btn_area.addWidget(self.new_date_btn)

            self.pop_v.addLayout(self.pop_btn_area)

            self.popup.setCentralWidget(self.popwidget)
            self.popup.show()
        else:
            self.no_image_warning()

    def save_edit_cross(self):
        """ Saves the edited values from edit pop up and closes pop up
        """
        try:
            time = int(self.time_text.text())
        except ValueError:
            self.create_error_msg("Time must be Integer")

        if time < 0 or time > 24:
            self.create_error_msg("Time must be in range 0-24")

        try:
            red_value = int(self.red_text.text())
        except ValueError:
            self.create_error_msg("Red value not an Integer")

        try:
            blue_value = int(self.blue_text.text())
        except ValueError:
            self.create_error_msg("Blue value not an Integer")

        cross = self.crosses[self.selected_cross]
        x = self.get_x_coordinate(time, date=cross.date)
        new_cross = Cross(time, cross.date, (x, self.get_y_coordinate(red_value)), red_value,
                          (x, self.get_y_coordinate(blue_value)), blue_value)

        self.crosses[self.selected_cross] = new_cross
        self.crosses = sorted(self.crosses, key=lambda x: (x.time, x.date))
        self.set_image(self.img)
        self.close_popup()

    def close_popup(self):
        """ Closes pop up
        """
        self.popup.close()

    def add_cross(self, event):
        """ Opens the cross editor for adding a new cross

        :param event: QTEvent passed by image_clicked
        """
        context_menu = QtWidgets.QMenu(self.mainWindow)
        add_action = context_menu.addAction("Add Cross")

        action = context_menu.exec_(self.mainWindow.mapToGlobal(event.pos()))

        if action == add_action:
            self.add_popup(event)

    def add_popup(self, event):
        """ Sets up and presents the Pop Up for adding new crosses

        :param event: QTEvent passed by image_clicked
        """
        if self.file_loaded:
            y = (event.pos().y() * 950) / 750
            x = (event.pos().x() * 1200) / 900
            val = reader.get_value(y) - 1

            est_time = 0
            for line in self.lines:
                if x in range(line[0], line[1] + 20):
                    est_time = line[3]
                    break

            self.popup = QtWidgets.QMainWindow()
            self.popup.setFixedSize(200, 200)
            self.popwidget = QtWidgets.QWidget(self.popup)
            self.popwidget.setObjectName("pop_central")
            self.pop_verticalWidget = QtWidgets.QWidget(self.popwidget)
            self.pop_verticalWidget.setGeometry(QtCore.QRect(0, 0, 200, 200))
            self.pop_verticalWidget.setObjectName("pop_verticalWidget")

            self.pop_v = QtWidgets.QVBoxLayout(self.pop_verticalWidget)
            self.pop_v.setContentsMargins(10, 10, 10, 10)
            self.pop_v.setObjectName("verticalLayout")

            self.pop_date_area = QtWidgets.QHBoxLayout()
            self.date_label = QtWidgets.QLabel()
            self.date_label.setText("Date:")
            self.pop_date_area.addWidget(self.date_label)
            self.date_value = QtWidgets.QComboBox()
            self.date_value.addItems(self.dates)
            self.date_value.setCurrentIndex(round((x+100) / 200) - 1)
            self.pop_date_area.addWidget(self.date_value)
            self.pop_v.addLayout(self.pop_date_area)

            self.pop_time_area = QtWidgets.QHBoxLayout()
            self.time_label = QtWidgets.QLabel()
            self.time_label.setText("Time:")
            self.pop_time_area.addWidget(self.time_label)
            self.time_text = QtWidgets.QLineEdit()
            self.time_text.setText(str(est_time))
            self.pop_time_area.addWidget(self.time_text)
            self.pop_v.addLayout(self.pop_time_area)

            self.red_cross = QtWidgets.QHBoxLayout()
            self.red_label = QtWidgets.QLabel()
            self.red_label.setText("Red cross value:")
            self.red_cross.addWidget(self.red_label)
            self.red_text = QtWidgets.QLineEdit()
            self.red_text.setText(str(val))
            self.red_cross.addWidget(self.red_text)
            self.pop_v.addLayout(self.red_cross)

            self.blue_cross = QtWidgets.QHBoxLayout()
            self.blue_label = QtWidgets.QLabel()
            self.blue_label.setText("Blue cross value:")
            self.blue_cross.addWidget(self.blue_label)
            self.blue_text = QtWidgets.QLineEdit()
            self.blue_text.setText(str(val))
            self.blue_cross.addWidget(self.blue_text)
            self.pop_v.addLayout(self.blue_cross)

            self.pop_btn_area = QtWidgets.QHBoxLayout()
            self.pop_btn_area.setObjectName("button_layout")

            self.close_btn = QtWidgets.QPushButton()
            self.close_btn.setText("Quit")
            self.close_btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
            self.close_btn.clicked.connect(self.close_popup)
            self.pop_btn_area.addWidget(self.close_btn)

            self.new_date_btn = QtWidgets.QPushButton()
            self.new_date_btn.setText("Save")
            self.new_date_btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
            self.new_date_btn.clicked.connect(self.save_new_cross)
            self.pop_btn_area.addWidget(self.new_date_btn)

            self.pop_v.addLayout(self.pop_btn_area)

            self.popup.setCentralWidget(self.popwidget)
            self.popup.show()
        else:
            self.no_image_warning()

    def save_new_cross(self):
        """ Saves new cross created with pop up and closes pop up
        """
        try:
            red_value = int(self.red_text.text())
        except ValueError:
            red_value = -1
            self.create_error_msg("Red value not an Integer")

        try:
            blue_value = int(self.blue_text.text())
        except ValueError:
            blue_value = -1
            self.create_error_msg("Blue value not an Integer")

        try:
            time = int(self.time_text.text())
        except ValueError:
            time = -1
            self.create_error_msg("Time not an Integer")

        x = self.get_x_coordinate(time)

        new_cross = Cross(time, self.date_value.currentIndex() + 1,
                          (x + 1, self.get_y_coordinate(red_value)), red_value,
                          (x - 1, self.get_y_coordinate(blue_value)), blue_value)
        exists = False
        for c in self.crosses:
            if c.__eq__(new_cross):
                exists = True
                break
        if exists:
            self.create_error_msg("This Cross already Exists either delete or just update")
        else:
            self.crosses.append(new_cross)
            self.crosses = sorted(self.crosses, key=lambda x: (x.time, x.date))
            self.set_image(self.img)
            self.close_popup()

    def get_x_coordinate(self, time, date=None):
        """ Gets x coordinate based on date and time.
        :param time: time for which x is supposed to be determined
        :param date: date for which x is supposed to be determined (default none)
        :return: x coordinate
        """
        if time in range(0, 25):
            x = 0
            if date == None:
                date = self.date_value.currentIndex() + 1
            for line in self.lines:
                if line[2] == date and line[3] == time:
                    x = line[0] + 9

            lower_line = 0
            upper_line = 0
            if x == 0:
                next_is_upper = False
                for line in self.lines:
                    if line[2] == date and line[3] < time:
                        lower_line = line[0] + 9
                        next_is_upper = True
                    elif next_is_upper:
                        upper_line = line[0] + 9
                        break

                x = int(lower_line + round(abs(upper_line - lower_line) / 2))
            return x

    def get_y_coordinate(self, value):
        """ Gets y coordinate for a certain value

        :param value: value for which y coordinate is supposed to be determined
        :return: y coordinate
        """
        if value < 40:
            y = (-13 * value) + 765
        elif value > 40:
            y = ((value - 40) * -6.5) - 245
        else:
            y = 230
        return y

    def update_date(self):
        """ Sets up and presents a calender pop-up to pick  new date
        """
        if self.file_loaded:
            self.popup = QtWidgets.QMainWindow()
            self.popup.setFixedSize(300, 300)
            self.popwidget = QtWidgets.QWidget(self.popup)
            self.popwidget.setObjectName("pop_central")

            self.pop_verticalWidget = QtWidgets.QWidget(self.popwidget)
            self.pop_verticalWidget.setGeometry(QtCore.QRect(0, 0, 300, 300))
            self.pop_verticalWidget.setObjectName("pop_verticalWidget")
            self.pop_v = QtWidgets.QVBoxLayout(self.pop_verticalWidget)
            self.pop_v.setContentsMargins(0, 0, 0, 0)
            self.pop_v.setObjectName("verticalLayout")

            self.calender = QtWidgets.QCalendarWidget()
            self.pop_v.addWidget(self.calender)
            self.new_date_btn = QtWidgets.QPushButton()
            self.new_date_btn.setText("Set First Date")
            self.new_date_btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
            self.new_date_btn.clicked.connect(self.save_date)
            self.pop_v.addWidget(self.new_date_btn)
            self.popup.setCentralWidget(self.popwidget)
            self.popup.show()
        else:
            self.no_image_warning()

    def save_date(self):
        """ Saves the selected date

        """
        date = self.calender.selectedDate()
        new_dates = [''] * 6
        for i in range(0, 6):
            new_dates[i] = (str(date.day()) + "." + str(date.month()) + "." + str(date.year()))
            date = date.addDays(1)
        self.dates = new_dates
        self.date_update_label.setText("The first read date is {} press Update Button to change "
                                       "the date.".format(self.dates[0]))

        self.popup.close()

    def open_file(self):
        """ Presents a file picker and opens selected file
        """
        dialog = QtWidgets.QFileDialog()
        name = dialog.getOpenFileName(filter="*.jpeg || *.JPEG || *.JPG || *.jpg || *.png || *.pdf")
        name = str(name[0])
        if name != '' and name != None:
            self.__clear()
            self.saveBtn.setText('Save File')
            self.file_list.append(name)
            self.open(name)

    def open_folder(self):
        """ Presents a directory picker and opens that directory
        """
        file = str(QtWidgets.QFileDialog.getExistingDirectory())
        if file != '' and file != None:
            self.__clear()
            self.saveBtn.setText('Save and Next')
            for f in os.listdir(file):
                if f.lower().endswith(('.jpeg', '.jpg', '.png', '.pdf')):
                    self.file_list.append("{}/{}".format(file, f))
            if len(self.file_list) > 0:
                self.open(self.file_list[0])
            else:
                self.create_error_msg('There are no images in this folder!')

    def open(self, file):
        """ Opens one specific TDK image

        :param file: file to be opened
        """
        if not str(file).endswith(".pdf"):
            self.input = file
            try:
                self.img = ImageUtils.preprocess_image(self.input)
            except Exception as e:
                self.create_error_msg(e)
            else:
                try:
                    self.crosses, self.dates, self.lines = reader.read_img(self.img)
                except Exception as e:
                    self.create_error_msg(e)
                else:
                    try:
                        self.name = reader.get_name(cv.imread(self.input))
                    except Exception:
                        self.name = ''
                    finally:
                        self.name_line_edit.setText(self.name)
                        self.date_update_label.setText("The first read date is {} press Update Button to change "
                                                       "the date.".format(self.dates[0]))
                        self.set_image(self.img)
                        self.file_loaded = True
        else:
            pages = pdf2image.convert_from_path(file, 500)
            for page in pages:
                page.save('temp.jpg', 'JPEG')
            self.open('temp.jpg')

    def create_error_msg(self, msg):
        """ Utility function for error presentation
        Opens pop up presenting error message

        :param msg: Error message
        """
        error_dialog = QtWidgets.QErrorMessage()
        error_dialog.showMessage(str(msg))
        error_dialog.exec_()
        
    def no_image_warning(self):
        """ Creates error for when no image is loaded
        """
        self.create_error_msg("No image loaded.")

    def save_file(self):
        """ Opens File dialog to select save location and saves csv of the image to
        specified location
        """
        if self.file_loaded:
            if len(self.file_list) == 1:
                dialog = QtWidgets.QFileDialog()
                try:
                    save_directory = str(dialog.getExistingDirectory())
                    if save_directory != '' and save_directory != None:
                        reader.cross_to_csv(self.dates, self.crosses, save_directory, self.name_line_edit.text())
                        dialog.close()
                        self.set_image(self.img)
                        self.__clear()
                except OSError as e:
                    self.create_error_msg(e)
            elif len(self.file_list) > 1:
                dialog = QtWidgets.QFileDialog()
                try:
                    save_directory = str(dialog.getExistingDirectory())
                    if save_directory != '' and save_directory != None:
                        reader.cross_to_csv(self.dates, self.crosses, save_directory, self.name_line_edit.text())
                        dialog.close()
                        self.file_list.pop(0)
                        self.open(self.file_list[0])
                except OSError as e:
                    self.create_error_msg(e)
            else:
                pass
        else:
            self.no_image_warning()

    def set_image(self, img, selected_cross=-1):
        """ Sets image

        :param img: image to be set
        :param selected_cross: selected cross if applicable
        """
        rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        m_np = np.copy(rgb)
        font = cv.FONT_HERSHEY_SIMPLEX
        # Show Values
        for i in range(1, 5):
            color = (255, 0, 0)
            if i == 4:
                cv.putText(m_np, '{}0'.format(i), (5, int(self.get_y_coordinate(i * 10) + 15)),
                           font, 0.6, color, 1, cv.LINE_AA)
                cv.putText(m_np, '50', (5, int(self.get_y_coordinate(i * 10) - 50)),
                           font, 0.6, color, 1, cv.LINE_AA)
            else:
                cv.putText(m_np, '{}0'.format(i), (5, int(self.get_y_coordinate(i * 10) + 3)),
                           font, 0.6, color, 1, cv.LINE_AA)

        #Show Crosses
        if selected_cross < 0:
            for cross in self.crosses:
                cv.circle(m_np, cross.red_pos, 10, (255, 0, 0))
                cv.circle(m_np, cross.blue_pos, 10, (0, 0, 255))
        else:
            for i, cross in enumerate(self.crosses):
                if i == selected_cross:
                    if cross.red_pos[1] < cross.blue_pos[1]:
                        cv.circle(m_np, cross.red_pos, 13, (255, 0, 0), 2)
                        cv.circle(m_np, cross.blue_pos, 13, (0, 0, 255), 2)
                        cv.putText(m_np, str(cross.red_value),
                                   (cross.red_pos[0] - 20, cross.red_pos[1] - 30),
                                   font, 1, (255, 0, 0), 1, cv.LINE_AA)
                        cv.putText(m_np, str(cross.blue_value),
                                   (cross.blue_pos[0] - 20, cross.blue_pos[1] + 40), font,
                                   1, (0, 0, 255), 1, cv.LINE_AA)
                    else:
                        cv.circle(m_np, cross.red_pos, 13, (255, 0, 0), 2)
                        cv.circle(m_np, cross.blue_pos, 13, (0, 0, 255), 2)
                        cv.putText(m_np, str(cross.red_value), (cross.red_pos[0] - 20, cross.red_pos[1] + 40),
                                   font, 1, (255, 0, 0), 1, cv.LINE_AA)
                        cv.putText(m_np, str(cross.blue_value),
                                   (cross.blue_pos[0] - 20, cross.blue_pos[1] - 30), font,
                                   1, (0, 0, 255), 1, cv.LINE_AA)
                else:
                    cv.circle(m_np, cross.red_pos, 10, (255, 0, 0))
                    cv.circle(m_np, cross.blue_pos, 10, (0, 0, 255))

        image_profile =QImage(m_np, m_np.shape[1], m_np.shape[0],
                     QImage.Format_RGB888)   # QImage object
        image_profile = image_profile.scaled(900, 750, aspectRatioMode=QtCore.Qt.KeepAspectRatio,
                                             transformMode=QtCore.Qt.SmoothTransformation)
        # To scale image for example and keep its Aspect Ration
        self.img_label.setPixmap(QtGui.QPixmap.fromImage(image_profile))

    def __clear(self):
        self.selected_cross = 0
        self.file_loaded = False
        self.file_list = []
        self.popup = None
        self.crosses = []
        self.dates = []
        self.lines = []
        self.img = None

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()

    ui = UiMainWindow()
    ui.setup_ui(MainWindow)

    MainWindow.show()
    sys.exit(app.exec_())
