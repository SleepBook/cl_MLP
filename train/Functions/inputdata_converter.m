%this convert the input data

id = fopen('batch_test.cdat','w');
fprintf(id,'1\n28\n28\n');
for i=1:10000
for m=1:28
    for n=1:28
        fprintf(id,'%f ',testImages(m,n,i));
    end
end
fprintf(id,'\n');
end
fclose(id);

lb = fopen('batch_test.lbl','w');
for i=1:10000
    if testLabels(i)==10
        fprintf(lb,'0\n');
    else   
        fprintf(lb,'%d\n',testLabels(i));
    end
end
fclose(lb);