clc; clear all; close all;

npts = 8;
nframe = 32;
nrk = 12;
frameall = 637;

dirname = './results';
datalist = dir(dirname);
datalist = datalist(3:end);
lengdata = length(datalist);

gts = zeros(3,npts,frameall);
infers = zeros(3,npts,frameall);
nr = 0;
for i=1:lengdata
    temp = load(sprintf('%s/%d.mat',dirname,i-1));

    gt = reshape(double(temp.gt),[],nframe,npts,3);
    gt = permute(squeeze(gt(:,:,:,:)),[3,2,1]);

    infer = reshape(double(temp.infer_rot),nframe,npts,3);
    infer = permute(infer,[3,2,1]);
    
    infer(1:2,:,:) = gt(1:2,:,:);

    f = size(infer, 3);
    for j=1:f
        if norm(infer(3,:,j)+gt(3,:,j)) <  norm(infer(3,:,j)-gt(3,:,j))
           infer(3,:,j) =  -infer(3,:,j);
        end
    end
    
    gts(:,:,nr+1:nr+f) = gt;
    infers(:,:,nr+1:nr+f) = infer;

    nr = nr+f;
end

infersn = infers(:,:,1:frameall);
gtsn = gts(:,:,1:frameall);
fperfs = compareStructs(reshape(permute(gtsn,[1,3,2]),[],npts),reshape(permute(infersn,[1,3,2]),[],npts));
fprintf('Performance: %.4f \n', mean(fperfs));
