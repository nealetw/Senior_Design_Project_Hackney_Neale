# ShenaniGANs User Manual

## Prerequisites

If you want to use this project, make sure you have the following installed:

- [Python 3.10](https://www.python.org/downloads/)
- [TensorFlow](https://www.tensorflow.org/install)
- [imageio library](https://imageio.readthedocs.io/en/stable/getting_started/installation.html)
- [matplotlib](https://matplotlib.org/stable/users/getting_started/index.html#installation-quick-start)

Download this project folder and open the python file /MNIST_GAN/mnist_GAN.py, or compile it by entering into your terminal,

```
python cat_gan.py
```

and the program will start running. You should verify that you have images (and properly formatted to be 128x128) in the directory CAT_GAN/catimages/test/cat/ and CAT_GAN/catimages/train/cat/. The direcorty structure can be seen here: ![image](https://user-images.githubusercontent.com/38301800/163854998-f48b9547-802b-404e-9701-62f238f49bbe.png)

The images used were derived from [this dataset from Kaggle](https://www.kaggle.com/datasets/andrewmvd/animal-faces), and a copy of the final dataset used [can be downloaded here](https://drive.google.com/file/d/14gdn4IYF_e6sjUpqKFrYRyUs3lE6CzF2/view?usp=sharing)

This program can take a long time to run, varying on the power of the computer it is run on. As it runs, the images it outputs will be in "/cat_gan/TrainingImages"

---

## FAQ

_My program won't run! It doesn't even say it started an epoch yet! What do I do?_
> Make sure you have images in the correct directory. You should put training images that are formatted to be 128x128 pixels into the directory /CAT_GAN/catimages/train/cat/ and /CAT_GAN/catimages/test/cat/ (*YOU MAY HAVE TO MAKE THESE FOLDERS YOURSELF*). Make sure they are different images. If your images aren't that size, you can use the program in /CAT_GAN/Image_Resizing, which is fairly simple to understand. 

_Why is it taking so long to get output?_

> It depends on the number of epochs you ran the program for (this variable can be changed at the top of the program) as well as the performance of your computer. Just make sure the Python process is running, and wait longer.

_Can I give the GAN more of my own pictures to train from?_

> In theory, yes, but it needs to be in the same resolution and format as the other training images for it to properly train.

_Can I use a different set of images to get a different kind of output?_

> In theory, yes you could, but it needs to be in the same resolution and format of the default images the GAN is configured to take, otherwise it will halt or output garbage. We would not recommend trying to, but you could reconfigure it to take different images.
