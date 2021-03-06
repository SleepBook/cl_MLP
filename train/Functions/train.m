function net = train(net,x,y,opts) %x represent image, y represent labels
    m = 10000;
    numbatches = m / opts.batchsize;     
    net.rl = [];
     
    for i = 1:opts.numepochs
        disp (['epoch' num2str(i) '/' num2str(opts.numepochs)]);
        kk = randperm(m);
        tic;       
        for l = 1 :numbatches
            batch_x = x(:,:, kk( (l - 1) * opts.batchsize + 1 : l * opts.batchsize) );
            batch_y = y(kk( (l - 1) * opts.batchsize + 1 : l * opts.batchsize),:);
             
            net = mlpff(net, batch_x);
            net = mlpbp(net, batch_y);
            %checkgrads(net,batch_x,batch_y);
            net = applyGrads(net, opts);
            if isempty(net.rl)
                net.rl(l) = net.L;
            end
            net.rl(end+1) = 0.99 * net.rl(end) + 0.01*net.L;
        end
        toc;
    end 
end