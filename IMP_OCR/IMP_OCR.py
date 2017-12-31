from PIL import Image
import pytesseract
import cv2
import os

def IMP_OCR(images):

    """
    Args:
        images: A list of file paths to grayscaled images of interest

    Returns:
        alphanums: A corresponding list of alphanumerics for each image. "NULL" if no char
    """

    alphanums = list()

    for image in images:
        currImg = cv2.imread(image)

        #Perform thresholding to separate foreground and background
        currImg = cv2.threshold(currImg, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        #Perform median blurring to reduce discretized random noise
        currImg = cv2.medianBlur(currImg, 3)

        #Save image to disk as a temp file to use Pytesseract
        tempFile = "{}.jpeg".format(os.getpid())
        cv2.imwrite(tempFile, currImg)

        #Load tempFile, apply Pytesseract OCR and then delete tempFile
        char = pytesseract.image_to_string(Image.open(tempFile))
        os.remove(tempFile)

        """
        Add FP Detection functionality here
            Set char = 'NULL' if FP detected
        """

        if char != 'NULL':
            alphanums.append(char)

    return alphanums