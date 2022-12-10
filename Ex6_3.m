%This my FNN for ex6.3; but currently I am failing to properly train; 
%If I am able to train it, I will upload NN file as well, so you don't have to train; just load and see how it works


clear all, close all
load ./DATA/VORTALL;

P = 0.8;
data = VORTALL;

train_size = round(151 * P);
test_size = 151 - train_size;

input = data(:, 1:train_size -1);
output = data(:, 2:train_size);

%FNN
net = feedforwardnet([20 20 20]);
net.layers{1}.transferFcn = 'logsig';
net.layers{2}.transferFcn = 'radbas';
net.layers{3}.transferFcn = 'purelin';
net = train(net,input',output');

test_data = data(:,train_size +1:end);
fhandle = plotCylinder(reshape(test_data(:,1),199,449));
