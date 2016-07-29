% demo code of actionness estimation
path_flow = 'example/';
gpu_id = 0;
filelist = dir([path_flow, '*.jpg']);
duration = length(filelist)/2 + 1;


% Read optical flow
flow = zeros(240, 320, 4, duration);
pre_flow_x = []; cur_flow_x = [];
pre_flow_y = []; cur_flow_y = [];
for k = 1:duration
    if k < duration
        name_x = sprintf('flow_x_%04d.jpg',k);
        name_y = sprintf('flow_y_%04d.jpg',k);
        if isempty(pre_flow_x)
            pre_flow_x = imresize(imread([path_flow,'/',name_x]),[240,320]);
            pre_flow_y = imresize(imread([path_flow,'/',name_y]),[240,320]);
        end
        cur_flow_x = imresize(imread([path_flow,'/',name_x]),[240,320]);
        cur_flow_y = imresize(imread([path_flow,'/',name_y]),[240,320]);
    end
    flow(:,:,:,k) = cat(3,pre_flow_x,pre_flow_y,cur_flow_x,cur_flow_y);
    pre_flow_x = cur_flow_x;
    pre_flow_y = cur_flow_y;
end



% Data preparation
flow(:) = flow(:) -128;
test_image = permute(flow,[2,1,3,4]);

batch_size = 50;
num_images = size(test_image,4);
num_batches = ceil(num_images/batch_size);

model_file = 'jhmdb_split1_actionness_m-fcn.caffemodel';

% Multi-scale test
scale = 1;
model_def_file = ['proto/actionness_m-fcn_scale_', num2str(scale), '_deploy.prototxt'];
caffe.reset_all();
caffe.set_mode_gpu();
caffe.set_device(gpu_id);
net = caffe.Net(model_def_file, model_file, 'test');

m_fcn_scale_1 = zeros(10, 13, 2, size(test_image,4));
images = zeros(214, 160, 4, batch_size, 'single');
for bb = 1 : num_batches
    range = 1 + batch_size*(bb-1): min(num_images,batch_size*bb);
    tmp = test_image(:,:,:,range);
    for i  =1:size(tmp,4)
        images(:,:,:,i) = imresize(tmp(:,:,:,i),[214, 160]);
    end
    net.blobs('data').set_data(images);
    net.forward_prefilled();
    prediction = permute(net.blobs('prob').get_data(), [2,1,3,4]);
    m_fcn_scale_1(:,:,:,range) = prediction(:,:,:,mod(range-1,batch_size)+1);
end

scale = 2;
model_def_file = ['proto/actionness_m-fcn_scale_', num2str(scale), '_deploy.prototxt'];
caffe.reset_all();
caffe.set_mode_gpu();
caffe.set_device(gpu_id);
net = caffe.Net(model_def_file, model_file, 'test');

m_fcn_scale_2 = zeros(15, 20, 2, size(test_image,4));
images = zeros(320, 240, 4, batch_size, 'single');
for bb = 1 : num_batches
    range = 1 + batch_size*(bb-1): min(num_images,batch_size*bb);
    tmp = test_image(:,:,:,range);
    images(:,:,:,1:size(tmp,4)) = tmp;
    net.blobs('data').set_data(images);
    net.forward_prefilled();
    prediction = permute(net.blobs('prob').get_data(), [2,1,3,4]);
    m_fcn_scale_2(:,:,:,range) = prediction(:,:,:,mod(range-1,batch_size)+1);
end

scale = 3;
model_def_file = ['proto/actionness_m-fcn_scale_', num2str(scale), '_deploy.prototxt'];
caffe.reset_all();
caffe.set_mode_gpu();
caffe.set_device(gpu_id);
net = caffe.Net(model_def_file, model_file, 'test');

m_fcn_scale_3 = zeros(22, 30, 2, size(test_image,4));
images = zeros(480, 360, 4, batch_size, 'single');
for bb = 1 : num_batches
    range = 1 + batch_size*(bb-1): min(num_images,batch_size*bb);
    tmp = test_image(:,:,:,range);
    for i  =1:size(tmp,4)
        images(:,:,:,i) = imresize(tmp(:,:,:,i),[480, 360]);
    end
    net.blobs('data').set_data(images);
    net.forward_prefilled();
    prediction = permute(net.blobs('prob').get_data(), [2,1,3,4]);
    m_fcn_scale_3(:,:,:,range) = prediction(:,:,:,mod(range-1,batch_size)+1);
end

scale = 4;
model_def_file = ['proto/actionness_m-fcn_scale_', num2str(scale), '_deploy.prototxt'];
caffe.reset_all();
caffe.set_mode_gpu();
caffe.set_device(gpu_id);
net = caffe.Net(model_def_file, model_file, 'test');

m_fcn_scale_4 = zeros(30, 40, 2, size(test_image,4));
images = zeros(640, 480, 4, batch_size, 'single');
for bb = 1 : num_batches
    range = 1 + batch_size*(bb-1): min(num_images,batch_size*bb);
    tmp = test_image(:,:,:,range);
    for i  =1:size(tmp,4)
        images(:,:,:,i) = imresize(tmp(:,:,:,i),[640,480]);
    end
    net.blobs('data').set_data(images);
    net.forward_prefilled();
    prediction = permute(net.blobs('prob').get_data(), [2,1,3,4]);
    m_fcn_scale_4(:,:,:,range) = prediction(:,:,:,mod(range-1,batch_size)+1);
end


for i = 1:size(video,4);
    subplot(1,2,1);
    imshow(video(:,:,:,i));
    subplot(1,2,2);
    result_a = (imresize(a_fcn_scale_1(:,:,2,i),[240,320]) ...
        +imresize(a_fcn_scale_2(:,:,2,i),[240,320])...
        +imresize(a_fcn_scale_3(:,:,2,i),[240,320])...
        +imresize(a_fcn_scale_4(:,:,2,i),[240,320]))/4;
    result_m = (imresize(m_fcn_scale_1(:,:,2,i),[240,320]) ...
        +imresize(m_fcn_scale_2(:,:,2,i),[240,320])...
        +imresize(m_fcn_scale_3(:,:,2,i),[240,320])...
        +imresize(m_fcn_scale_4(:,:,2,i),[240,320]))/4;
    imagesc(result_a + result_m);
    axis image; axis off;
    pause(1);
end
