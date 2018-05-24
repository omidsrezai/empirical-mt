////////Dense Pyramidal Lucas and Kanade Optical Flow////////
////////Omid Rezai -  omid.rezai@uwaterloo.ca///////////////

#include <math.h>

__global__ void LKPYRCUDA(float* wr_vx, float* wr_vy, 
              float* r_vx, float* r_vy, float* img1, float* img2,                
              int window,
			  int iter,
			  int hw,
			  float alpha,
			  size_t Pitch,
			  unsigned int Width,
			  unsigned int Height)
{
  // calculate normalized texture coordinates
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  
  if( (x <Height)&&(y< Width ))
  
  {
    //wr_vx[y*Pitch + x] = lrintf( r_vx[x+Pitch*y]);
    //wr_vy[y*Pitch + x] = lrintf( r_vy[x+Pitch*y]);
    float vxPrec = lrintf( r_vx[x+Pitch*y]); // tex2D(_levelTexture7,x,y);
	float vyPrec = lrintf( r_vy[x+Pitch*y]); //tex2D(_levelTexture8,x,y);

	for (int r=0 ; r<iter ; ++r)
	{
    int lr = x - hw + vyPrec;
    int hr = x + hw + vyPrec;

    int lc = y - hw + vxPrec;
    int hc = y + hw + vxPrec;

	if ( (lr<0)||(hr>=Height)||(lc<0)||(hc>=Width))
	{
	//if indices outside image, last value keeped
	}
	else
	{
    float Ix;
  	float Iy;
  	float It;
  	float G11=.0f;
  	float G22=.0f;
  	float G12=.0f;
  	float bx=.0f;
  	float by=.0f;

	//Computation of matrices A and b
		for(int i=0;i<window-2;++i)
			for(int j=0;j<window-2;++j)
		{
            if ((i + lr < Height ) && (j + lc < Width )){
		//interpolation by texture
		Ix = (img1[x-hw+i+(y-hw+j)*Pitch]- img1[x-hw+i+(y+1-hw+j)*Pitch] + img1[x+1-hw+i+(y-hw+j)*Pitch]- img1[x+1-hw+i+(y+1-hw+j)*Pitch])*0.25 + (img2[lr+i+(lc+j)*Pitch]- img2[lr+i+(lc+j+1)*Pitch]+ img2[lr+i+1+(lc+j)*Pitch] - img2[lr+i+1+(lc+j+1)*Pitch])*0.25;                  
        //Ix = Drx1[x-hw+i+(y-hw+j)*Pitch]; + Drx2[lr+i+(lc+j)*Pitch];

		Iy = (img1[x-hw+i+(y-hw+j)*Pitch] + img1[x-hw+i+(y+1-hw+j)*Pitch] - img1[x+1-hw+i+(y-hw+j)*Pitch] - img1[x+1-hw+i+(y+1-hw+j)*Pitch] )*0.25+ (img2[lr+i+(lc+j)*Pitch] + img2[lr+i+(lc+j+1)*Pitch] - img2[lr+i+1+(lc+j)*Pitch] - img2[lr+i+1+(lc+j+1)*Pitch])*0.25;
        //Iy = Dry1[x-hw+i+(y-hw+j)*Pitch] + Dry2[lr+i+(lc+j)*Pitch];                                                                                                                                                

		It = (img1[x-hw+i+(y-hw+j)*Pitch] + img1[x-hw+i+(y+1-hw+j)*Pitch] + img1[x+1-hw+i+(y-hw+j)*Pitch] + img1[x+1-hw+i+(y+1-hw+j)*Pitch])*0.25 + (img2[lr+i+(lc+j)*Pitch] + img2[lr+i+(lc+j+1)*Pitch] + img2[lr+i+1+(lc+j)*Pitch] + img2[lr+i+1+(lc+j+1)*Pitch])*-0.25;       
        //It = Drt1[x-hw+i+(y-hw+j)*Pitch] + Drt2[lr+i+(lc+j)*Pitch];
        //wr_vx [j*Pitch + i] = It;
		G11 += Ix*Ix; 
		G22 += Iy*Iy;
		G12 += Ix*Iy;

		bx -= (It*Ix);
		by -= (It*Iy);
		}
        }
		G11 += alpha;
		G22 += alpha;
        //wr_vy [8] = G11;
        //wr_vy [9] = G12;

        //wr_vy [488] = G12;
        //wr_vy [489] = G22;
	//determinant and inverse
	//variable Ix is    reused to store the determinant
	Ix = (float)(1./(G11 * G22 - G12 * G12));
    //wr_vy [11] = Ix;
    //wr_vy [12] = bx;
    //wr_vy [13] = by;
	//speed vectors update
	vxPrec += (Ix * (G22*bx - G12*by));
	vyPrec += (Ix * (G11*by - G12*bx));
	}
    
	}

	//speed vectors final update
	wr_vx[y*Pitch + x] = vxPrec;
	wr_vy[y*Pitch + x] = vyPrec;
  }
}

