clear all
close all

peppers_truth = imread('peppers.png');
camera = imread('cameraman.tif');
imwrite(peppers_truth, 'peppers_truth.png');
imwrite(camera, 'camera_truth.png');

filters = {'salt & pepper', 'gaussian', 'speckle', 'poisson'};

for i = 1:4
    peppers_noisy = imnoise(peppers_truth, filters{i});
    imwrite(peppers_noisy, ['peps_noisy' num2str(i) '.png'])
end

for i = 1:4
    camera_noisy = imnoise(camera, filters{i});
    imwrite(camera_noisy, ['camera_noisy' num2str(i) '.png'])
end