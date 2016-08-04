function net = mlpff(net, x)
    num = size(x, 3);
    net.layers{1}.a = (reshape(x, 784, num));
    
    for l = 2:net.layer_num        
        net.layers{l}.z = net.layers{l}.w * net.layers{l-1}.a + repmat(net.layers{l}.b,1,num);
        net.layers{l}.a = sigm(net.layers{l}.z);        
    end
    
end