#author: Jaqueline Alvarenga Silveira
#Numero Usp: 10242771

import cv2
import numpy as np
from scipy.fftpack import dct, idct
import string

#Hiding information on image frequencies through DCT
#This program allows a user to hide and extract information in JPEG images through the discrete cosine transform.
#The idea is to try to embed the information in the image preferably without any visual degradation and
#with some resistance to compression.

#Instruction to run this program:
#python steganography.py text

#Text to be embedded in the image
text = raw_input('Text: ') + " "

#Image to load
nameImage = raw_input('Image name: ')

fileName = "test/" + nameImage
#Read image
im = cv2.imread(fileName, 1)

#@Description this function embed a text in the image
#
#@name encodingDCT
#
#@param img is the input image
#@param text is the message
#@return image with message embedded
def encodingDCT(img, text):

	col = img.shape[1]

	row = img.shape[0]

	valueForce = 30

	blockW  = 8
	blockH = 8

	matrixW = col/blockW
	matrixH = row/blockH

	i = 0

	size = len(text)*8

	mess = [ord(ch) for ch in text]

	imgf = np.float32(img)

	planRGB = cv2.split(imgf)

	c1 = [0]
	c2 = [0]

	for x in range(1, matrixW):
		for y in range(1, matrixH):
			mx = (x-1)*blockW
			my = (y-1)*blockH

			block = planRGB[0][my:my + blockH, mx: mx + blockW]

			freq = dct(block, norm="ortho")

			c1[0] = freq[6,7]
			c2[0] = freq[5,1]

			#c1[1] = freq[3,1]
			#c2[1] = freq[1,2]

			#c1[2] = freq[0,1]
			#c2[2] = freq[1,3]

			#c1[3] = freq[0,7]
			#c2[3] = freq[7,0]

			if (i >= size):
				break

			for k in range(0, len(c1)):
				val = 0
				if (i < size):
					val = (mess[i/8] & 1 << i%8) >> i%8
					i = i + 1

				if (val == 0):
					if (c1[k] > c2[k]):
						c1[k], c2[k] = c2[k], c1[k]
				elif (c1[k] < c2[k]):
					c1[k], c2[k] = c2[k], c1[k]

				if (c1[k] > c2[k]):
					c1[k] = c1[k] + ((valueForce - (c1[k] - c2[k]))/2)
					c2[k] = c2[k] - ((valueForce - (c1[k] - c2[k]))/2)
				else:
					c1[k] = c1[k] - ((valueForce - (c2[k] - c1[k]))/2)
					c2[k] = c2[k] + ((valueForce - (c2[k] - c1[k]))/2)

			freq[6,7] = c1[k]
			freq[5,1] = c2[k]

			#freq[3,1] = c1[1]
			#freq[1,2] = c2[1]

			#freq[0,1] = c1[2]
			#freq[1,3] = c2[2]

			#freq[0,7] = c1[3]
			#freq[7,0] = c2[3]

			imS = idct(freq, norm="ortho")

			np.copyto(planRGB[0][my:my + blockH, mx: mx + blockW],imS)

	merF = cv2.merge(planRGB)

	mer = np.uint8(merF)

	return mer

#@Description this function extract the message embedded in the image
#
#@name decodingDCT
#
#@param img is the input image
#@return message embedded
def decodingDCT(img):

	col = img.shape[1]
	row = img.shape[0]

	blockW = 8
	blockH = 8
	matrixW = col/blockW
	matrixH = row/blockH

	i = 0
	bits = ''

	imgf = np.float32(img)

	planRGB = cv2.split(imgf)

	flag = 0

	extract = ''

	c1 = [0]
	c2 = [0]

	for x in range(1, matrixW):
		for y in range(1, matrixH):

			mx = (x - 1)*blockW
			my = (y - 1)*blockH

			block = planRGB[0][my:my + blockH, mx: mx + blockW]

			freq = dct(block, norm="ortho")

			c1[0] = freq[6,7]
			c2[0] = freq[5,1]

			#c1[1] = freq[3,1]
			#c2[1] = freq[1,2]

			#c1[2] = freq[0,1]
			#c2[2] = freq[1,3]

			#a[3] = trans[0,7]
			#b[3] = trans[7,0]

			for k in range(0,len(c1)):
				if (c1[k] > c2[k]):
					bits = bits + '1'
					flag = flag + 1
					i = 0
				else:
					bits = bits + '0'
					flag = flag + 1
					i = i + 1

				if (flag == 8):
					dec = bits[::-1]
					extract = extract + chr(int(dec, 2))
					bits = ''
					flag = 0

	final = ''
	for c in extract:
		if (ord(c) < 128):
			final = final + c
		else:
			break

	#best_string_ever = re.sub('[\0\200-\377]', '', extract)

	best_string_ever = filter(lambda x: x in string.printable, final)

	return best_string_ever

#@Description this function calulates the standard deviation of the prediction errors (RMSE)
#
#@name rmse
#
#@param img is the input image
#@param steg is the stego image
#@return rmse
def rmse(origin, steg):
    return np.sqrt(((origin - steg) ** 2).mean())

#@Description this function determine whether there is a significant difference between the expected frequencies
# and the observed frequencies.
#
#@name chiSquare
#
#@param im1 is the input image
#@param im2 is the stego image
#@return a text message indicating whether the image is suspect or not
def chiSquare(im1, im2):
    hist1 = [0]*256

    hist2 = [0]*256

    row=im1.shape[0]
    col=im1.shape[1]

    #hist of im1
    for x in range (0, row):
        for y in range (0, col):
            b, g, r = im1[x, y]
            gr = (r+g+b)/3.0
            hist1[int(gr)] = hist1[int(gr)] + 1

    #hist of im2
    for x in range (0, row):
        for y in range (0, col):
            b, g, r = im2[x, y]
            gr = (r+g+b)/3.0
            hist2[int(gr)] = hist2[int(gr)] + 1

    #chi
    chisqr = 0.0
    for i in range (0,256):
        if (hist2[i] != 0):
            diff = (hist1[i] - hist2[i])^2
            chisqr += abs(diff)/float(hist2[i])

	return ("Chi-squared: "+str(chisqr))

#@Description this function generates the histogram of an image.
#
#@name show_histogram
#
#@param img is an image
#@param str is the histogram title
def show_histogram(img, str):

    h = np.zeros((300, 256, 3))
    b, g, r = cv2.split(img)
    bins = np.arange(256).reshape(256, 1)
    color = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

    for item, col in zip([b, g, r], color):
        hist_item = cv2.calcHist([item], [0], None, [256], [0, 255])
        cv2.normalize(hist_item, hist_item, 0, 255, cv2.NORM_MINMAX)
        hist = np.int32(np.around(hist_item))
        pts = np.column_stack((bins, hist))
        cv2.polylines(h, [pts], False, col)

    h = np.flipud(h)

    cv2.imshow('Histogram '+str, h)
    cv2.waitKey(10)


################################################################################################
################################################################################################
cv2.imshow("Input", im)

stego = encodingDCT(im, text)

cv2.imwrite('test\stego.jpg', stego, [cv2.IMWRITE_JPEG_QUALITY, 100])

cv2.imwrite('test\stego.png', stego, [cv2.IMWRITE_PNG_COMPRESSION, 0])

img2 = cv2.imread("test\stego.png", 1)

cv2.imshow("Stego", stego)

mess = decodingDCT(img2)

print 'Message decoded: '+mess

chi = chiSquare(im,img2)

print chi

rmse_val = rmse(im, img2)

print("The RMSE is: " + str(rmse_val))

show_histogram(im, 'Input')
show_histogram(img2,'Stego')

cv2.waitKey(0)
cv2.destroyAllWindows()
################################################################################################
################################################################################################




















