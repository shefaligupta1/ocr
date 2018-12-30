# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'qt_ocr1.ui'
#
# Created by: PyQt5 UI code generator 5.11.2
#
# WARNING! All changes made in this file will be lost!
import cv2
import numpy as np
import imutils
import pytesseract
from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(QtWidgets.QMainWindow):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.frame = QtWidgets.QFrame(self.centralwidget)
        self.frame.setGeometry(QtCore.QRect(0, 0, 791, 571))
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.pushButton = QtWidgets.QPushButton(self.frame)
        self.pushButton.setGeometry(QtCore.QRect(320, 40, 101, 31))
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(self.frame)
        self.pushButton_2.setGeometry(QtCore.QRect(320, 100, 101, 31))
        self.pushButton_2.setObjectName("pushButton_2")
        self.label = QtWidgets.QLabel(self.frame)
        self.label.setGeometry(QtCore.QRect(40, 190, 721, 331))
        self.label.setObjectName("label")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.frame.show()

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.pushButton.setText(_translate("MainWindow", "Store"))
        self.pushButton_2.setText(_translate("MainWindow", "Recognize"))
        self.label.setText(_translate("MainWindow", "TextLabel"))
        self.pushButton.clicked.connect(self.store)
        self.pushButton_2.clicked.connect(self.recognize)

    def store(self):
        data = []

        image = cv2.imread("chassis_3.jpg")

        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        img = cv2.GaussianBlur(img, (7, 7), 0)

        edged = cv2.Canny(img, 40, 90)
        dilate = cv2.dilate(edged, None, iterations=2)

        mask = np.ones(img.shape[:2], dtype="uint8") * 255

        cnts = cv2.findContours(dilate.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]

        orig = img.copy()
        for c in cnts:
            if cv2.contourArea(c) < 200:
                cv2.drawContours(mask, [c], -1, 0, -1)
                x, y, w, h = cv2.boundingRect(c)
            if (w > h):
                cv2.drawContours(mask, [c], -1, 0, -1)

        newimage = cv2.bitwise_and(dilate.copy(), dilate.copy(), mask=mask)
        img2 = cv2.dilate(newimage, None, iterations=3)
        ret2, th1 = cv2.threshold(img2, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        # th1 = cv2.resize(th1,(50*50,20*3))

        temp = pytesseract.image_to_string(th1)
        # Write results on the image
        fc = cv2.putText(image, temp, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 255), 3)
        fc = cv2.resize(fc, (392407, 1))
        data.append(fc.flatten())

        intClassifications = []
        intValidChars = [ord('0'), ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), ord('6'), ord('7'), ord('8'),
                         ord('9'),
                         ord('A'), ord('B'), ord('C'), ord('D'), ord('E'), ord('F'), ord('G'), ord('H'), ord('I'),
                         ord('J'),
                         ord('K'), ord('L'), ord('M'), ord('N'), ord('O'), ord('P'), ord('Q'), ord('R'), ord('S'),
                         ord('T'),
                         ord('U'), ord('V'), ord('W'), ord('X'), ord('Y'), ord('Z')]
        #intChar = cv2.waitKey(0)
        #if intChar == 27:  # if esc key was pressed
            #sys.exit()  # exit program
        #elif intChar in intValidChars:
        #intClassifications.append(temp)

        #fltClassifications = np.array(intClassifications,np.float32)

        #npaClassifications = fltClassifications.reshape((fltClassifications.size, 1))
        #np.savetxt("classifications.txt", npaClassifications)

        #np.savetxt('fc.txt',[temp])


        f= open("data_1.txt","w+")
        f.write(temp)
        f.close()

        #names = {
            #0: temp
            # 1: 'user2',
        #}

        # if cv2.waitKey(2) == 27 or len(data) >= 20:
        # break
        # cv2.imshow('result', img2)
        # else:
        # print("Some error")

        data = np.asarray(data)
        data = cv2.resize(data, (392407, 1))
        np.save('2.npy', data)
        #np.savetxt('data_1.txt',)
        #data_1 = np.asarray(temp)
        #np.save('3.npy', data_1)
        #np.save('3.csv',names)

        cv2.destroyAllWindows()
        # cap.release()

    def recognize(self):
        self.store()

        def distance(x1, x2):
            return np.sqrt(((x1 - x2) ** 2).sum())

        def knn(x, train, targets, k=5):
            m = train.shape[0]
            dist = []
            for ix in range(m):
                dist.append(distance(x, train[ix]))
            dist = np.asarray(dist)
            indx = np.argsort(dist)
            sorted_labels = labels[indx][:k]
            counts = np.unique(sorted_labels, return_counts=True)
            return counts[0][np.argmax(counts[1])]

        #npaClassifications = np.loadtxt("classifications.txt", np.float32)
        #npaClassifications = npaClassifications.reshape((npaClassifications.size, 1))


        f_01 = np.load('2.npy')
        a_1 = np.load('data_1.txt',np.float32 )
        #names = np.loadtxt("2.npy.txt", np.float32)

        # print(f_01.shape)
        labels = np.zeros((40, 1))
        labels[:20, :] = 0.0  # first 20 for user_1 (0)
        #labels[:20, 1] = np.load('2.csv')



        names = {
            0: a_1
            #1: 'user2',
        }


        data = np.concatenate([f_01])

        # print(data.shape)

        image = cv2.imread("chassis_3.jpg")
        # image = cv2.resize(image,(392407,1))
        # image = cv2.resize(image,(640,480))
        # make it gray

        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # blur it to remove noise
        img = cv2.GaussianBlur(img, (7, 7), 0)

        edged = cv2.Canny(img, 40, 90)
        dilate = cv2.dilate(edged, None, iterations=2)
        # perform erosion if necessay, it completely depends on the image
        # erode = cv2.erode(dilate, None, iterations=1)

        mask = np.ones(img.shape[:2], dtype="uint8") * 255

        # find contours
        cnts = cv2.findContours(dilate.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]

        orig = img.copy()
        for c in cnts:
            # if the contour is not sufficiently large, ignore it
            if cv2.contourArea(c) < 200:
                cv2.drawContours(mask, [c], -1, 0, -1)

            x, y, w, h = cv2.boundingRect(c)

            # filter more contours if nessesary
            if (w > h):
                cv2.drawContours(mask, [c], -1, 0, -1)

        newimage = cv2.bitwise_and(dilate.copy(), dilate.copy(), mask=mask)
        img2 = cv2.dilate(newimage, None, iterations=3)
        ret2, th1 = cv2.threshold(img2, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        th1 = cv2.resize(th1, (392407, 1))
        # Tesseract OCR on the image
        # temp = pytesseract.image_to_string(th1)
        # Write results on the image

        lab = knn(th1.flatten(), data, labels)
        text = names[int(lab)]
        #print(text)

        self.image_3 = cv2.putText(image, text, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 255), 3)

        self.label.setPixmap(self.image_3)

        cv2.waitKey(0)
        cv2.destroyAllWindows()



if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

