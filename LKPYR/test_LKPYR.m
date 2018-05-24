clear

im1 = imread('frame1.png');
im2 = imread('frame2.png');

rangee = [-16 16];


[u, v, time] = LKPYR(im1, im2, 'numLevels', 3, 'winSize', 9);
time 

figure()
imshow(u, rangee)
colormap(gca, jet)


figure()
subplot(3,1,1)
imshow(im1)
subplot(3,1,2)
imshow(im2)
subplot(3,1,3)
quiver(u,v)
