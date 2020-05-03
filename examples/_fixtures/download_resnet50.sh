#!/bin/sh

MODEL=resnet50

mkdir $MODEL

wget http://s3.amazonaws.com/store.carml.org/models/caffe/resnet50/ResNet-50-deploy.prototxt -O $MODEL/resnet50.prototxt
wget http://s3.amazonaws.com/store.carml.org/models/caffe/resnet50/ResNet-50-model.caffemodel -O $MODEL/resnet50.caffemodel
wget http://s3.amazonaws.com/store.carml.org/synsets/imagenet/synset.txt -O $MODEL/synset.txt

