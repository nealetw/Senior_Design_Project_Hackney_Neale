from PIL import Image
import os

#Open CAT_GAN folder

#Directory of the original images should be /CAT_GAN/image_Resizing/originalImages
    #inside there should be /test/cat/ and /train/cat/
#This program will take those, and change them to the size below
#Can also change to be black and white, to make it a 1 channel image
#The new images will be going to /CAT_GAN/catimages/ and into the respective /test/ and /train/ folder


TARGET_SIZE = (128,128)
BLACK_AND_WHITE = False

os.chdir('image_Resizing/')

for x in os.listdir('../catimages/test/cat/'):
    os.remove('../catimages/test/cat/'+x)

for x in os.listdir('originalImages/test/cat/'):
    image = Image.open('originalImages/test/cat/'+x)
    newIm = image.resize(TARGET_SIZE)
    if(BLACK_AND_WHITE):
        newIm = newIm.convert("L") #to greyscale
    newName = '../catimages/test/cat/'+x+".png"
    newIm.save(newName)

for x in os.listdir('originalImages/train/cat/'):
    image = Image.open('originalImages/train/cat/'+x)
    newIm = image.resize(TARGET_SIZE)
    if(BLACK_AND_WHITE):
        newIm = newIm.convert("L") #to greyscale
    newName = '../catimages/train/cat/'+x+".png"
    newIm.save(newName)