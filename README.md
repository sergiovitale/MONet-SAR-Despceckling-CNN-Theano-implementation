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
This code is written on Ubuntu system for Python2.7 and uses Theano library.

The command to install the requirements is: 

```
cat requirements.txt | xargs -n 1 -L 1 pip2 install
```

### Anaconda (Optional)
If you use a python editor
* install [Anaconda](https://repo.anaconda.com/archive/)
* install requirements and **spyder** editor with **conda**
* edit **main.py** and run

### Optional requirements for using gpu:
* cuda = 8 
* cudnn = 5


# Usage 
* **imgs** folder contains three samples images with simulated single look speckle in amplitude format;
the sample image are taken from [UC Merced LandUse Dataset](http://weegee.vision.ucmerced.edu/datasets/landuse.html).
Three differente <AREA> can be tested:
     * baseballdiamond
     * golfcourse
     * storagetanks

* **model** folder contains the pre-trained network
* run test without GPU
```
python main.py -a <AREA>
```
* run with GPU
```
PATH=<CUDAPATH>:$PATH python main.py -g -a <AREA>
```
