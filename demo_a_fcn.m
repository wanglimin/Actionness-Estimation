% demo code of actionness estimation
rgb_video =  'example.avi';
flow_video = 'example/';
gpu_id = 0;


% Read video
vidObj = VideoReader(rgb_video);
video = read(vidObj);

% Data preparation
IMAGE_MEAN = load('VGG_mean.mat');
IMAGE_MEAN = IMAGE_MEAN.image_mean;
IMAGE_MEAN = imresize(IMAGE_MEAN,[240,320]);
test_image = single(video(:,:,[3,2,1],:));
test_image = bsxfun(@minus,test_image,IMAGE_MEAN);
test_image = permute(test_image,[2,1,3,4]);

batch_size = 50;
num_images = size(test_image,4);
num_batches = ceil(num_images/batch_size);

model_file = 'jhmdb_split1_actionness_a-fcn.caffemodel';

% Multi-scale test
scale = 1;
model_def_file = ['proto/actionness_a-fcn_scale_', num2str(scale), '_deploy.prototxt'];
caffe.reset_all();
caffe.set_mode_gpu();
caffe.set_device(gpu_id);
net = caffe.Net(model_def_file, model_file, 'test');

a_fcn_scale_1 = zeros(10, 13, 2, size(test_image,4));
images = zeros(214, 160, 3, batch_size, 'single');
for bb = 1 : num_batches
    range = 1 + batch_size*(bb-1): min(num_images,batch_size*bb);
    tmp = test_image(:,:,:,range);
    for i  =1:size(tmp,4)
        images(:,:,:,i) = imresize(tmp(:,:,:,i),[214, 160]);
    end
    net.blobs('data').set_data(images);
    net.forward_prefilled();
    prediction = permute(net.blobs('prob').get_data(), [2,1,3,4]);
    a_fcn_scale_1(:,:,:,range) = prediction(:,:,:,mod(range-1,batch_size)+1);
end

scale = 2;
model_def_file = ['proto/actionness_a-fcn_scale_', num2str(scale), '_deploy.prototxt'];
caffe.reset_all();
caffe.set_mode_gpu();
caffe.set_device(gpu_id);
net = caffe.Net(model_def_file, model_file, 'test');

a_fcn_scale_2 = zeros(15, 20, 2, size(test_image,4));
images = zeros(320, 240, 3, batch_size, 'single');
for bb = 1 : num_batches
    range = 1 + batch_size*(bb-1): min(num_images,batch_size*bb);
    tmp = test_image(:,:,:,range);
    images(:,:,:,1:size(tmp,4)) = tmp;
    net.blobs('data').set_data(images);
    net.forward_prefilled();
    prediction = permute(net.blobs('prob').get_data(), [2,1,3,4]);
    a_fcn_scale_2(:,:,:,range) = prediction(:,:,:,mod(range-1,batch_size)+1);
end

scale = 3;
model_def_file = ['proto/actionness_a-fcn_scale_', num2str(scale), '_deploy.prototxt'];
caffe.reset_all();
caffe.set_mode_gpu();
caffe.set_device(gpu_id);
net = caffe.Net(model_def_file, model_file, 'test');

a_fcn_scale_3 = zeros(22, 30, 2, size(test_image,4));
images = zeros(480, 360, 3, batch_size, 'single');
for bb = 1 : num_batches
    range = 1 + batch_size*(bb-1): min(num_images,batch_size*bb);
    tmp = test_image(:,:,:,range);
    for i  =1:size(tmp,4)
        images(:,:,:,i) = imresize(tmp(:,:,:,i),[480, 360]);
    end
    net.blobs('data').set_data(images);
    net.forward_prefilled();
    prediction = permute(net.blobs('prob').get_data(), [2,1,3,4]);
    a_fcn_scale_3(:,:,:,range) = prediction(:,:,:,mod(range-1,batch_size)+1);
end

scale = 4;
model_def_file = ['proto/actionness_a-fcn_scale_', num2str(scale), '_deploy.prototxt'];
caffe.reset_all();
caffe.set_mode_gpu();
caffe.set_device(gpu_id);
net = caffe.Net(model_def_file, model_file, 'test');

a_fcn_scale_4 = zeros(30, 40, 2, size(test_image,4));
images = zeros(640, 480, 3, batch_size, 'single');
for bb = 1 : num_batches
    range = 1 + batch_size*(bb-1): min(num_images,batch_size*bb);
    tmp = test_image(:,:,:,range);
    for i  =1:size(tmp,4)
        images(:,:,:,i) = imresize(tmp(:,:,:,i),[640,480]);
    end
    net.blobs('data').set_data(images);
    net.forward_prefilled();
    prediction = permute(net.blobs('prob').get_data(), [2,1,3,4]);
    a_fcn_scale_4(:,:,:,range) = prediction(:,:,:,mod(range-1,batch_size)+1);
end


for i = 1:size(video);
    subplot(1,2,1);
    imshow(video(:,:,:,i));
    subplot(1,2,2);
    result = (imresize(a_fcn_scale_1(:,:,2,i),[240,320]) ...
        +imresize(a_fcn_scale_2(:,:,2,i),[240,320])...
        +imresize(a_fcn_scale_3(:,:,2,i),[240,320])...
        +imresize(a_fcn_scale_4(:,:,2,i),[240,320]))/4;
    imagesc(result);
    axis image; axis off;
    pause(1);
end