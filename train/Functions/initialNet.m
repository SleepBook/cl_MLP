function net = initialNet(net)
    for l = 2:net.layer_num
        fan_in = net.layers{l}.input;
        fan_out = net.layers{l}.output;
        net.layers{l}.w = (rand(fan_out, fan_in) - 0.5) * 2 * sqrt(6/(fan_in + fan_out));
        net.layers{l}.b = zeros(fan_out, 1);
    
 
%             net.layers{l}.w = (rand(120,256) - 0.5) * 2 * sqrt(6/(fan_in + fan_out));
%             net.layers{l}.b = zeros(120,1);
%             inputmaps = 120;
%         end
        
%         if strcmp(net.layers{l}.type, 'F6')
%             fan_in = 120;
%             fan_out = 84;
%             net.layers{l}.w = (rand(84,120)-0.5) * 2 * sqrt(6/(fan_in + fan_out));
%             net.layers{l}.b = zeros(84,1);
%             inputmaps = 84;
%         end
%         
%         if strcmp(net.layers{l}.type, 'Soft')
%             fan_in = 84;
%             fan_out = 10;
%             net.layers{l}.w = (rand(10,84)-0.5) * 2 * sqrt(6/(fan_in+fan_out));
%             net.layers{l}.b = zeros(10,1);
%         end
    end
end