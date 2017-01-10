import os

test_img = "sample1.jpg"
tesseract_command = "tesseract " + test_img + " out"
os.system(tesseract_command)


# os.remove(test_img)

# fileOpen = open("out.txt", "r")
#
# os.remove(textOutput)
