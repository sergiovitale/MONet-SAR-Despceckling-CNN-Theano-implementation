# MONet - SAR Despceckling CNN - (python/theano implementation)

[MONet: CNN based SAR despeckling](https://arxiv.org/abs/2006.09050) is a CNN for single look SAR despeckling.

This the python (Theano) version of the code

# Team members
 Sergio Vitale    (contact person, sergio.vitale@uniparthenope.it);
 Giampaolo Ferraioli (giampaolo.ferraioli@uniparthenope.it);
 Vito Pascazio (vito.pascazio@uniparthenope.it)
 
# License
....
By downloading and/or using any of these files, you implicitly agree to all the
terms of the license, as specified in the document LICENSE.txt
(included in this directory)

# Prerequisits
This code is written for Python2.7 and uses Theano library.
The list of all requirements is in `requirements.txt`.

The command to install the requirements is: 

```
cat requirements.txt | xargs -n 1 -L 1 pip2 install
```

Optional requirements for using gpu:
* cuda = 8 
* cudnn = 5

# Usage
* **imgs** folder contains three samples images with simulated single look speckle in amplitude format;
the sample image are taken from [UC Merced LandUse Dataset](http://weegee.vision.ucmerced.edu/datasets/landuse.html)
* **model** folder contains the pre-trained network
* open **main.py**, select the image to filter and run
