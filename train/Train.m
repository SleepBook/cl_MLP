clear;clc;
path(path,'Functions/');

%% ===============================================================
%load data
imageDim = 28;
numclasses = 10;
images = loadMNISTImages('DataSet/train-images.idx3-ubyte');
images = reshape(images,imageDim,imageDim,[]);
labels = loadMNISTLabels('DataSet/train-labels.idx1-ubyte');
labels(labels == 0) = 10;

%% initialparameters
mlp.layer_num = 4; %include the input glue
mlp.layers = {  
    struct('type','input')
    struct('type','real','input',784,'output',300)
    struct('type','real','input',300,'output',60)
    struct('type','real','input',60,'output',10)
};
mlp = initialNet(mlp);

%% train the mlp
opts.alpha = 0.1;
opts.batchsize = 25;
opts.numepochs = 20;
mlp = train(mlp,images,labels,opts);

%% test the trained mlp
testImages = loadMNISTImages('DataSet/t10k-images.idx3-ubyte');
testImages = reshape(testImages,imageDim,imageDim,[]);
testLabels = loadMNISTLabels('DataSet/t10k-labels.idx1-ubyte');
testLabels(testLabels == 0) = 10;

mlp = mlpff(mlp,testImages);

a = mlp.layers{4}.a;
[~,preds] = max(mlp.layers{4}.a,[],1);
preds = preds';

acc = sum(preds == testLabels) / length(preds);
fprintf('Accuracy is %f\n',acc);
