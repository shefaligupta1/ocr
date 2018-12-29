import cv2
import numpy as np
import imutils
import pytesseract

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
th1 = cv2.resize(th1,(50*50,20*3))


temp = pytesseract.image_to_string(th1)
# Write results on the image
fc = cv2.putText(image, temp, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 255), 3)
fc = cv2.resize(fc,(39240,1))
data.append(fc.flatten())
#if cv2.waitKey(2) == 27 or len(data) >= 20:
    #break
    #cv2.imshow('result', img2)
#else:
    #print("Some error")

data = np.asarray(data)
np.save('2.npy', data)

cv2.destroyAllWindows()
#cap.release()


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

f_01 = np.load('2.npy')
labels = np.zeros((40, 1))
labels[:20, :] = 0.0  # first 20 for user_1 (0)

names = {
            0: temp
           # 1: 'user2',
        }

data = np.concatenate([f_01])

image = cv2.imread("chassis_3.jpg")
#image = cv2.resize(image,(392407,1))
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

# Tesseract OCR on the image
#temp = pytesseract.image_to_string(th1)
# Write results on the image

lab = knn(th1.flatten(), data, labels)
text = names[int(lab)]

cv2.putText(image, temp, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 255), 3)

cv2.waitKey(0)
cv2.destroyAllWindows()