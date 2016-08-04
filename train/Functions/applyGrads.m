function net = applyGrads(net,opts)
for l = 2: numel(net.layers)    
%     w = net.layers{l}.w;
%     dw = opts.alpha * net.layers{l}.dw;
    net.layers{l}.w = net.layers{l}.w - opts.alpha * net.layers{l}.dw;
    %w1 = net.layers{l}.w;
    net.layers{l}.b = net.layers{l}.b - opts.alpha * net.layers{l}.db;
end
end