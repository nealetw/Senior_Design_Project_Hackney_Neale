# ShenaniGANs User Manual

## Prerequisites

If you want to use this project, make sure you have the following installed:

- [Python 3.10](https://www.python.org/downloads/)
- [TensorFlow](https://www.tensorflow.org/install)
- [imageio library](https://imageio.readthedocs.io/en/stable/getting_started/installation.html)
- [matplotlib](https://matplotlib.org/stable/users/getting_started/index.html#installation-quick-start)

Download this project folder and open the python file /MNIST_GAN/mnist_GAN.py, or compile it by entering into your terminal,

```
python mnist_GAN.py
```

and the program will start running.

This program can take a long time to run, varying on the power of the computer it is run on. As it runs, the images it outputs will be in "/MNIST_GAN/TrainingImages"

---

## FAQ

_Why is it taking so long to get output?_

> It depends on the number of epochs you ran the program for (this variable can be changed at the top of the program) as well as the performance of your computer. Just make sure the Python process is running, and wait longer.

_Can I give the GAN more of my own pictures to train from?_

> In theory, yes, but it needs to be in the same resolution and format as the other training images for it to properly train.

_Can I use a different set of images to get a different kind of output?_

> In theory, yes you could, but it needs to be in the same resolution and format of the default images the GAN is configured to take, otherwise it will halt or output garbage. We would not recommend trying to, but you could reconfigure it to take different images.
