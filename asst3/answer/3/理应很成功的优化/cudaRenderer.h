#ifndef __CUDA_RENDERER_H__
#define __CUDA_RENDERER_H__

#ifndef uint
#define uint unsigned int
#endif

#include "circleRenderer.h"


class CudaRenderer : public CircleRenderer {

private:

    Image* image;
    SceneName sceneName;

    int numCircles;
    float* position;
    float* velocity;
    float* color;
    float* radius;

    float* cudaDevicePosition;
    float* cudaDeviceVelocity;
    float* cudaDeviceColor;
    float* cudaDeviceRadius;
    float* cudaDeviceImageData;

    int* cudaDeviceElementInPiece;
    int* cudaDeviceElementInPieceCurrent;
    int* cudaDeviceNewMemoryToPiece;
    int numPieceWidth;
    int numPieceHeight;
    int widthPerPiece;
    int heightPerPiece;
public:

    CudaRenderer();
    virtual ~CudaRenderer();

    const Image* getImage();

    void setup();

    void loadScene(SceneName name);

    void allocOutputImage(int width, int height);

    void clearImage();

    void advanceAnimation();

    void render();

    void shadePixel(
        int circleIndex,
        float pixelCenterX, float pixelCenterY,
        float px, float py, float pz,
        float* pixelData);

    // void numCircleInPieceCPU(std::vector<int>&cmp_v1, std::vector<std::vector<int>>& cmp_v2);
};


#endif
