%%This is a tool used to extract the net from the matlab model
%%which comply the the format I defined.
%written by oar, 05/23/2016


f = fopen('output/mnist_mlp.net','w');
fprintf(f,'%d\n',mlp.layer_num);
for i=2:mlp.layer_num
    fprintf(f, '%d ',mlp.layers{i}.input);
end
fprintf(f,'%d\n',mlp.layers{mlp.layer_num}.output);
for j = 2:mlp.layer_num
    for k = 1:mlp.layers{j}.input
        for m=1:mlp.layers{j}.output
            fprintf(f,'%f ',mlp.layers{j}.w(m,k));
        end
        fprintf(f,'\n');
    end
    for m=1:mlp.layers{j}.output
        fprintf(f,'%f ',mlp.layers{j}.b(m));
    end
    fprintf(f,'\n');
end
fclose(f);




    
    
            
