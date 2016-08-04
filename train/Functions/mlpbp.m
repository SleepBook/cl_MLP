function net = mlpbp(net,y)
n = numel(net.layers);
num = size(y,1);
groundTruth = zeros(10,num);
for j = 1 : num
    for i = 1 : 10
        if y(j) == i
            groundTruth(i,j) = 1;
        end
    end
end

net.L = -1 ./ num * groundTruth(:)' *  log(net.layers{n}.a(:)) ;
net.layers{n}.delta = -(groundTruth - net.layers{n}.a);
%fprintf('the delta in this iteration is %f\n',net.layers{n}.delta);

for l=n-1:-1:2   
    net.layers{l}.delta = (net.layers{l+1}.w'* net.layers{l+1}.delta).* sigmoidGrad(net.layers{l}.z);
end

for l = 2 : n
    net.layers{l}.dw = net.layers{l}.delta * net.layers{l-1}.a' / size(net.layers{l-1}.a,2);
    net.layers{l}.db = mean(net.layers{l}.delta ,2);    
end

function grad = sigmoidGrad(x)
    e_x = exp(-x);
    grad = e_x ./ ((1 + e_x).^2); 
end
end