%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Dense Pyramidal Lucas and Kanade Optical Flow
%Omid Rezai -  omid.srezai@uwaterloo.ca
function [u,v,varargout] = LKPYR (im1, im2, varargin)
%%% [u,v] = LKPYR (im1, im2)
%%% u and v are horizontal and vertical optical flow vectors.
%%% im1 and im2 are grayscale images with same size.
%%% 
%%% [u,v,runTime] = LKPYR (im1, im2) 
%%% You can have the execution time for each optical flow map calculation
%%% in milliseconds.
%%%
%%%  You can achieve additional control over LKPYR by using
%%%  parameter/value pairs. For example:
%%%
%%%  [u,v] = LKPYR(im1, im1, PARAM1, VALUE1, PARAM2, VALUE2, ...)
%%%
%%%  Parameters include:
%%%
%%%       'numLevels'     - A scalar specifying the
%%%                       number of levels in the gaussian pyramid.  
%%%                       Default is 3.
%%%                      
%%%       'winSize'       - A scalar specifying the
%%%                       size of the window in which all pixels have the same velocity.
%%%                       Default is 9 pixels.
%%% 

    persistent CudaLKer 
    
    [ImgHeight, ImgWidth] = size(im1);
    
    iter = 1;
    numLevels = 3;
    winSize = 9;
    hw = floor(winSize/2);    
    alpha = .001;
        
    if numel(varargin) == 2
        if strcmp(varargin{1},'numLevels')
            numLevels = varargin{2};
        elseif strcmp(varargin{1},'winSize')
            winSize = varargin{2};
            hw = floor(winSize/2);
        else    
            error(message('Wrong input argument(s) for LKPYR function'));
        end
        
    elseif numel(varargin) == 4 
        if strcmp(varargin{1},'numLevels') && strcmp(varargin{3},'winSize')
            numLevels = varargin{2};
            winSize = varargin{4}; 
            hw = floor(winSize/2);
        elseif strcmp(varargin{1},'winSize') && strcmp(varargin{3},'numLevels')
            winSize = varargin{2};
            hw = floor(winSize/2);
            numLevels = varargin{4}; 
        else    
            error(message('Wrong input argument(s) for LKPYR function'));
        end
        
    elseif (numel(varargin)  > 4 || numel(varargin) == 1 || numel(varargin) == 3)
        error(message('Wrong number of input arguments for LKPYR function'));
    end
    
    if (numLevels < 1 || ~isa(numLevels,'double') || mod(numLevels,1) ~= 0 )
        error(message('numLevels should be an integer larger than 1'));
    end
    
    if (winSize < 2 || ~isa(winSize,'double') || mod(winSize,1) ~= 0 )
        error(message('winSize should be an integer larger than 2'));
    end
    
    if mod(ImgHeight,2) ||  mod(ImgWidth,2)
       error(message('Image width and height should even'));
    end
    
     ratioHeight = ceil(ImgHeight / (2^(numLevels-1)));
     PaddedHeight = 2^(numLevels-1)*ratioHeight;
     padHeight = (PaddedHeight - ImgHeight)/2;
     
     im1 = padarray(im1,[padHeight 0],'both');
     im2 = padarray(im2,[padHeight 0],'both');   
     
     ratioWidth = ceil(ImgWidth / (2^(numLevels-1)));
     PaddedWidth  = 2^(numLevels-1)*ratioWidth;
     padWidth  = (PaddedWidth - ImgWidth)/2;
     
     
     im1 = padarray(im1,[0 padWidth],'both');
     im2 = padarray(im2,[0 padWidth],'both');  
     
     [ImgHeight, ImgWidth] = size(im1);
     
    if (isempty (CudaLKer)|| CudaLKer.GridSize(1)*CudaLKer.ThreadBlockSize(1) ~= ImgHeight || CudaLKer.GridSize(2)*CudaLKer.ThreadBlockSize(2)~= ImgWidth )
        t0=clock;
        CudaLKer = GPUInitialization(ImgHeight,ImgWidth); 
        GPUInstallationTime = etime(clock,t0)
    end
    
    
    
    [u,v,varargout{1}] = LKPYRComput(im1, im2, CudaLKer, iter, numLevels, winSize, hw, alpha); 
    u = u(padHeight+1:end-padHeight,padWidth+1:end-padWidth);
    v = v(padHeight+1:end-padHeight,padWidth+1:end-padWidth);
end
%--------------------------------------------------------------------------

%==========================================================================
function  [Vx,Vy,runTime] = LKPYRComput (im1, im2, CudaLKer, iter, numLevels, win, hw, alpha) 
     
    pyramid1 = im1;
    pyramid2 = im2;

    t0 = clock;
    for i=2:numLevels
        im1 = impyramid(im1, 'reduce');
        im2 = impyramid(im2, 'reduce');
        pyramid1(1:size(im1,1), 1:size(im1,2), i) = im1;
        pyramid2(1:size(im2,1), 1:size(im2,2), i) = im2;
    end;
    
    %tic;
    for p =1:numLevels
        im1 = pyramid1(1:(size(pyramid1,1)/(2^(numLevels - p))), 1:(size(pyramid1,2)/(2^(numLevels - p))), (numLevels - p)+1);
        im2 = pyramid2(1:(size(pyramid2,1)/(2^(numLevels - p))), 1:(size(pyramid2,2)/(2^(numLevels - p))), (numLevels - p)+1);
        
%          figure(),imshow(im1)
        
        if p==1
        Vxinl=zeros(size(im1));
        Vyinl=zeros(size(im1));
        
        
        else  
        %resizing
        Vxinl = 2 * imresize(Vx,size(Vx)*2,'bilinear');   
        Vyinl = 2 * imresize(Vy,size(Vy)*2,'bilinear');
        end
        Vx = single(gpuArray.zeros(size(im1)));
        Vy = single(gpuArray.zeros(size(im1)));

% %         %%
% %         winFoo = win./(2^(numLevels - p));
% %         %%

        [Vx, Vy] = feval(CudaLKer, Vx, Vy, Vxinl, Vyinl, im1, im2, win, iter, hw, alpha, size(im1,1), size(im1,2), size(im1,1));
        Vx = gather(Vx);
        Vy = gather(Vy);

    end

    runTime = 1000*etime(clock,t0);
end
%--------------------------------------------------------------------------

%==========================================================================
function Ker = GPUInitialization(H,W)
    filename = 'LKPYRCUDA';
    cuFile =[filename, '.cu'];
    ptxFile =[filename '.' 'ptx'];
    system (['nvcc -ptx ' cuFile ' -o ' ptxFile]);
    Ker = parallel.gpu.CUDAKernel(ptxFile, cuFile);
    MaxThrd = floor(sqrt(Ker.MaxThreadsPerBlock));
    if (H >MaxThrd || W>MaxThrd )
        for MaxThrdX = MaxThrd:-1:1
            if ~mod(H,MaxThrdX)
                GridSizX = (H/MaxThrdX);
                break;
            end
        end
        
        for MaxThrdY = MaxThrd:-1:1
            if ~mod(W,MaxThrdY)
                GridSizY = W/MaxThrdY;
                break;
            end
        end   
        
        Ker.GridSize = [GridSizX GridSizY];
        Ker.ThreadBlockSize = [MaxThrdX MaxThrdY]; 
    else     
        Ker.ThreadBlockSize = [MaxThrd MaxThrd];         
    end
end
%--------------------------------------------------------------------------
