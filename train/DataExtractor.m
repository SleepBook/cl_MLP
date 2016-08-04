%%This is a tool used to extract test data from the MNIST dataset
%%either to extract single frame data or several frames(batch_mode)
%%define a switch at the very beginning
%%written by oar, 05/23/2016.

data_op = 1;
batch_mode = 0;
batch_size = 100;

if(data_op == 1)
    f = fopen('output/test.dat','w');
    fprintf(f,'1\n');
    fprintf(f,'784\n');
    for i=1:784
        fprintf(f,'%f ',mlp.layers{1}.a(i,1));
    end
    fprintf(f, '\n');
    fclose(f);
end

if(batch_mode == 1)
    f1 = fopen('output/batch_test.dat','w');
    fprintf(f1,'%d\n',batch_size);
    fprintf(f,'784\n');
    for i=1:batch_size
        for j=1:784
            fprintf(f1,'%f ',mlp.layers{1}.a(i,1));
        end
        fprintf('\n');
    end
    fclose(f1);
end
    

