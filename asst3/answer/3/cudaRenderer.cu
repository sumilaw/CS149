#include <string>
#include <algorithm>
#include <math.h>
#include <stdio.h>
#include <vector>
#include <assert.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "cudaRenderer.h"
#include "image.h"
#include "noise.h"
#include "sceneLoader.h"
#include "util.h"
#include "CycleTimer.h"

#define NUM_PIECE_WIDTH 32
#define NUM_PIECE_HEIGHT 32

////////////////////////////////////////////////////////////////////////////////////////
// Putting all the cuda kernels here
///////////////////////////////////////////////////////////////////////////////////////

struct GlobalConstants {

    SceneName sceneName;

    int numCircles;
    float* position;
    float* velocity;
    float* color;
    float* radius;

    int imageWidth;
    int imageHeight;
    float* imageData;

    // 分块
    int* numElementInPiece;
    int* numElementInPieceCurrent;
    int** newMemoryToPiece;
    int numPieceWidth;
    int numPieceHeight;
    int widthPerPiece;
    int heightPerPiece;
};

// Global variable that is in scope, but read-only, for all cuda
// kernels.  The __constant__ modifier designates this variable will
// be stored in special "constant" memory on the GPU. (we didn't talk
// about this type of memory in class, but constant memory is a fast
// place to put read-only variables).
__constant__ GlobalConstants cuConstRendererParams;

// read-only lookup tables used to quickly compute noise (needed by
// advanceAnimation for the snowflake scene)
__constant__ int    cuConstNoiseYPermutationTable[256];
__constant__ int    cuConstNoiseXPermutationTable[256];
__constant__ float  cuConstNoise1DValueTable[256];

// color ramp table needed for the color ramp lookup shader
#define COLOR_MAP_SIZE 5
__constant__ float  cuConstColorRamp[COLOR_MAP_SIZE][3];


// including parts of the CUDA code from external files to keep this
// file simpler and to seperate code that should not be modified
#include "noiseCuda.cu_inl"
#include "lookupColor.cu_inl"


// kernelClearImageSnowflake -- (CUDA device code)
//
// Clear the image, setting the image to the white-gray gradation that
// is used in the snowflake image
__global__ void kernelClearImageSnowflake() {

    int imageX = blockIdx.x * blockDim.x + threadIdx.x;
    int imageY = blockIdx.y * blockDim.y + threadIdx.y;

    int width = cuConstRendererParams.imageWidth;
    int height = cuConstRendererParams.imageHeight;

    if (imageX >= width || imageY >= height)
        return;

    int offset = 4 * (imageY * width + imageX);
    float shade = .4f + .45f * static_cast<float>(height-imageY) / height;
    float4 value = make_float4(shade, shade, shade, 1.f);

    // write to global memory: As an optimization, I use a float4
    // store, that results in more efficient code than if I coded this
    // up as four seperate fp32 stores.
    *(float4*)(&cuConstRendererParams.imageData[offset]) = value;
}

// kernelClearImage --  (CUDA device code)
//
// Clear the image, setting all pixels to the specified color rgba
__global__ void kernelClearImage(float r, float g, float b, float a) {

    int imageX = blockIdx.x * blockDim.x + threadIdx.x;
    int imageY = blockIdx.y * blockDim.y + threadIdx.y;

    int width = cuConstRendererParams.imageWidth;
    int height = cuConstRendererParams.imageHeight;

    if (imageX >= width || imageY >= height)
        return;

    int offset = 4 * (imageY * width + imageX);
    float4 value = make_float4(r, g, b, a);

    // write to global memory: As an optimization, I use a float4
    // store, that results in more efficient code than if I coded this
    // up as four seperate fp32 stores.
    *(float4*)(&cuConstRendererParams.imageData[offset]) = value;
}

// kernelAdvanceFireWorks
// 
// Update the position of the fireworks (if circle is firework)
__global__ void kernelAdvanceFireWorks() {
    const float dt = 1.f / 60.f;
    const float pi = 3.14159;
    const float maxDist = 0.25f;

    float* velocity = cuConstRendererParams.velocity;
    float* position = cuConstRendererParams.position;
    float* radius = cuConstRendererParams.radius;

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= cuConstRendererParams.numCircles)
        return;

    if (0 <= index && index < NUM_FIREWORKS) { // firework center; no update 
        return;
    }

    // determine the fire-work center/spark indices
    int fIdx = (index - NUM_FIREWORKS) / NUM_SPARKS;
    int sfIdx = (index - NUM_FIREWORKS) % NUM_SPARKS;

    int index3i = 3 * fIdx;
    int sIdx = NUM_FIREWORKS + fIdx * NUM_SPARKS + sfIdx;
    int index3j = 3 * sIdx;

    float cx = position[index3i];
    float cy = position[index3i+1];

    // update position
    position[index3j] += velocity[index3j] * dt;
    position[index3j+1] += velocity[index3j+1] * dt;

    // fire-work sparks
    float sx = position[index3j];
    float sy = position[index3j+1];

    // compute vector from firework-spark
    float cxsx = sx - cx;
    float cysy = sy - cy;

    // compute distance from fire-work 
    float dist = sqrt(cxsx * cxsx + cysy * cysy);
    if (dist > maxDist) { // restore to starting position 
        // random starting position on fire-work's rim
        float angle = (sfIdx * 2 * pi)/NUM_SPARKS;
        float sinA = sin(angle);
        float cosA = cos(angle);
        float x = cosA * radius[fIdx];
        float y = sinA * radius[fIdx];

        position[index3j] = position[index3i] + x;
        position[index3j+1] = position[index3i+1] + y;
        position[index3j+2] = 0.0f;

        // travel scaled unit length 
        velocity[index3j] = cosA/5.0;
        velocity[index3j+1] = sinA/5.0;
        velocity[index3j+2] = 0.0f;
    }
}

// kernelAdvanceHypnosis   
//
// Update the radius/color of the circles
__global__ void kernelAdvanceHypnosis() { 
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= cuConstRendererParams.numCircles) 
        return; 

    float* radius = cuConstRendererParams.radius; 

    float cutOff = 0.5f;
    // place circle back in center after reaching threshold radisus 
    if (radius[index] > cutOff) { 
        radius[index] = 0.02f; 
    } else { 
        radius[index] += 0.01f; 
    }   
}   


// kernelAdvanceBouncingBalls
// 
// Update the positino of the balls
__global__ void kernelAdvanceBouncingBalls() { 
    const float dt = 1.f / 60.f;
    const float kGravity = -2.8f; // sorry Newton
    const float kDragCoeff = -0.8f;
    const float epsilon = 0.001f;

    int index = blockIdx.x * blockDim.x + threadIdx.x; 
   
    if (index >= cuConstRendererParams.numCircles) 
        return; 

    float* velocity = cuConstRendererParams.velocity; 
    float* position = cuConstRendererParams.position; 

    int index3 = 3 * index;
    // reverse velocity if center position < 0
    float oldVelocity = velocity[index3+1];
    float oldPosition = position[index3+1];

    if (oldVelocity == 0.f && oldPosition == 0.f) { // stop-condition 
        return;
    }

    if (position[index3+1] < 0 && oldVelocity < 0.f) { // bounce ball 
        velocity[index3+1] *= kDragCoeff;
    }

    // update velocity: v = u + at (only along y-axis)
    velocity[index3+1] += kGravity * dt;

    // update positions (only along y-axis)
    position[index3+1] += velocity[index3+1] * dt;

    if (fabsf(velocity[index3+1] - oldVelocity) < epsilon
        && oldPosition < 0.0f
        && fabsf(position[index3+1]-oldPosition) < epsilon) { // stop ball 
        velocity[index3+1] = 0.f;
        position[index3+1] = 0.f;
    }
}

// kernelAdvanceSnowflake -- (CUDA device code)
//
// move the snowflake animation forward one time step.  Updates circle
// positions and velocities.  Note how the position of the snowflake
// is reset if it moves off the left, right, or bottom of the screen.
__global__ void kernelAdvanceSnowflake() {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= cuConstRendererParams.numCircles)
        return;

    const float dt = 1.f / 60.f;
    const float kGravity = -1.8f; // sorry Newton
    const float kDragCoeff = 2.f;

    int index3 = 3 * index;

    float* positionPtr = &cuConstRendererParams.position[index3];
    float* velocityPtr = &cuConstRendererParams.velocity[index3];

    // loads from global memory
    float3 position = *((float3*)positionPtr);
    float3 velocity = *((float3*)velocityPtr);

    // hack to make farther circles move more slowly, giving the
    // illusion of parallax
    float forceScaling = fmin(fmax(1.f - position.z, .1f), 1.f); // clamp

    // add some noise to the motion to make the snow flutter
    float3 noiseInput;
    noiseInput.x = 10.f * position.x;
    noiseInput.y = 10.f * position.y;
    noiseInput.z = 255.f * position.z;
    float2 noiseForce = cudaVec2CellNoise(noiseInput, index);
    noiseForce.x *= 7.5f;
    noiseForce.y *= 5.f;

    // drag
    float2 dragForce;
    dragForce.x = -1.f * kDragCoeff * velocity.x;
    dragForce.y = -1.f * kDragCoeff * velocity.y;

    // update positions
    position.x += velocity.x * dt;
    position.y += velocity.y * dt;

    // update velocities
    velocity.x += forceScaling * (noiseForce.x + dragForce.y) * dt;
    velocity.y += forceScaling * (kGravity + noiseForce.y + dragForce.y) * dt;

    float radius = cuConstRendererParams.radius[index];

    // if the snowflake has moved off the left, right or bottom of
    // the screen, place it back at the top and give it a
    // pseudorandom x position and velocity.
    if ( (position.y + radius < 0.f) ||
         (position.x + radius) < -0.f ||
         (position.x - radius) > 1.f)
    {
        noiseInput.x = 255.f * position.x;
        noiseInput.y = 255.f * position.y;
        noiseInput.z = 255.f * position.z;
        noiseForce = cudaVec2CellNoise(noiseInput, index);

        position.x = .5f + .5f * noiseForce.x;
        position.y = 1.35f + radius;

        // restart from 0 vertical velocity.  Choose a
        // pseudo-random horizontal velocity.
        velocity.x = 2.f * noiseForce.y;
        velocity.y = 0.f;
    }

    // store updated positions and velocities to global memory
    *((float3*)positionPtr) = position;
    *((float3*)velocityPtr) = velocity;
}

// shadePixel -- (CUDA device code)
//
// given a pixel and a circle, determines the contribution to the
// pixel from the circle.  Update of the image is done in this
// function.  Called by kernelRenderCircles()
__device__ __inline__ void
shadePixel(int circleIndex, float2 pixelCenter, float3 p, float4* imagePtr) {

    float diffX = p.x - pixelCenter.x;
    float diffY = p.y - pixelCenter.y;
    float pixelDist = diffX * diffX + diffY * diffY;

    float rad = cuConstRendererParams.radius[circleIndex];;
    float maxDist = rad * rad;

    // circle does not contribute to the image
    if (pixelDist > maxDist)
        return;

    float3 rgb;
    float alpha;

    // there is a non-zero contribution.  Now compute the shading value

    // suggestion: This conditional is in the inner loop.  Although it
    // will evaluate the same for all threads, there is overhead in
    // setting up the lane masks etc to implement the conditional.  It
    // would be wise to perform this logic outside of the loop next in
    // kernelRenderCircles.  (If feeling good about yourself, you
    // could use some specialized template magic).
    if (cuConstRendererParams.sceneName == SNOWFLAKES || cuConstRendererParams.sceneName == SNOWFLAKES_SINGLE_FRAME) {

        const float kCircleMaxAlpha = .5f;
        const float falloffScale = 4.f;

        float normPixelDist = sqrt(pixelDist) / rad;
        rgb = lookupColor(normPixelDist);

        float maxAlpha = .6f + .4f * (1.f-p.z);
        maxAlpha = kCircleMaxAlpha * fmaxf(fminf(maxAlpha, 1.f), 0.f); // kCircleMaxAlpha * clamped value
        alpha = maxAlpha * exp(-1.f * falloffScale * normPixelDist * normPixelDist);

    } else {
        // simple: each circle has an assigned color
        int index3 = 3 * circleIndex;
        rgb = *(float3*)&(cuConstRendererParams.color[index3]);
        alpha = .5f;
    }

    float oneMinusAlpha = 1.f - alpha;

    // BEGIN SHOULD-BE-ATOMIC REGION
    // global memory read

    float4 existingColor = *imagePtr;
    float4 newColor;
    newColor.x = alpha * rgb.x + oneMinusAlpha * existingColor.x;
    newColor.y = alpha * rgb.y + oneMinusAlpha * existingColor.y;
    newColor.z = alpha * rgb.z + oneMinusAlpha * existingColor.z;
    newColor.w = alpha + existingColor.w;

    // global memory write
    *imagePtr = newColor;

    // END SHOULD-BE-ATOMIC REGION
}

// kernelRenderCircles -- (CUDA device code)
//
// Each thread renders a circle.  Since there is no protection to
// ensure order of update or mutual exclusion on the output image, the
// resulting image will be incorrect.
__global__ void kernelRenderCircles() {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= cuConstRendererParams.numCircles)
        return;

    int index3 = 3 * index;

    // read position and radius
    float3 p = *(float3*)(&cuConstRendererParams.position[index3]);
    float  rad = cuConstRendererParams.radius[index];

    // compute the bounding box of the circle. The bound is in integer
    // screen coordinates, so it's clamped to the edges of the screen.
    short imageWidth = cuConstRendererParams.imageWidth;
    short imageHeight = cuConstRendererParams.imageHeight;
    short minX = static_cast<short>(imageWidth * (p.x - rad));
    short maxX = static_cast<short>(imageWidth * (p.x + rad)) + 1;
    short minY = static_cast<short>(imageHeight * (p.y - rad));
    short maxY = static_cast<short>(imageHeight * (p.y + rad)) + 1;

    // a bunch of clamps.  Is there a CUDA built-in for this?
    short screenMinX = (minX > 0) ? ((minX < imageWidth) ? minX : imageWidth) : 0;
    short screenMaxX = (maxX > 0) ? ((maxX < imageWidth) ? maxX : imageWidth) : 0;
    short screenMinY = (minY > 0) ? ((minY < imageHeight) ? minY : imageHeight) : 0;
    short screenMaxY = (maxY > 0) ? ((maxY < imageHeight) ? maxY : imageHeight) : 0;

    float invWidth = 1.f / imageWidth;
    float invHeight = 1.f / imageHeight;

    // for all pixels in the bonding box
    for (int pixelY=screenMinY; pixelY<screenMaxY; pixelY++) {
        float4* imgPtr = (float4*)(&cuConstRendererParams.imageData[4 * (pixelY * imageWidth + screenMinX)]);
        for (int pixelX=screenMinX; pixelX<screenMaxX; pixelX++) {
            float2 pixelCenterNorm = make_float2(invWidth * (static_cast<float>(pixelX) + 0.5f),
                                                 invHeight * (static_cast<float>(pixelY) + 0.5f));
            shadePixel(index, pixelCenterNorm, p, imgPtr);
            imgPtr++;
        }
    }
}

__device__ __inline__ int
circleInBoxConservative(
    float circleX, float circleY, float circleRadius,
    float boxL, float boxR, float boxT, float boxB)
{

    // expand box by circle radius.  Test if circle center is in the
    // expanded box.

    if ( circleX >= (boxL - circleRadius) &&
         circleX <= (boxR + circleRadius) &&
         circleY >= (boxB - circleRadius) &&
         circleY <= (boxT + circleRadius) ) {
        return 1;
    } else {
        return 0;
    }
}

__device__ __inline__ int
circleInBox(
    float circleX, float circleY, float circleRadius,
    float boxL, float boxR, float boxT, float boxB)
{

    // clamp circle center to box (finds the closest point on the box)
    float closestX = (circleX > boxL) ? ((circleX < boxR) ? circleX : boxR) : boxL;
    float closestY = (circleY > boxB) ? ((circleY < boxT) ? circleY : boxT) : boxB;

    // is circle radius less than the distance to the closest point on
    // the box?
    float distX = closestX - circleX;
    float distY = closestY - circleY;

    if ( ((distX*distX) + (distY*distY)) <= (circleRadius*circleRadius) ) {
        return 1;
    } else {
        return 0;
    }
}

////////////////////////////////////////////////////////////////////////////////////////


CudaRenderer::CudaRenderer() {
    image = NULL;

    numCircles = 0;
    position = NULL;
    velocity = NULL;
    color = NULL;
    radius = NULL;

    cudaDevicePosition = NULL;
    cudaDeviceVelocity = NULL;
    cudaDeviceColor = NULL;
    cudaDeviceRadius = NULL;
    cudaDeviceImageData = NULL;
    cudaDeviceElementInPiece = NULL;
    cudaDeviceNewMemoryToPiece = NULL;
}

CudaRenderer::~CudaRenderer() {

    if (image) {
        delete image;
    }

    if (position) {
        delete [] position;
        delete [] velocity;
        delete [] color;
        delete [] radius;
    }

    if (cudaDevicePosition) {
        cudaFree(cudaDevicePosition);
        cudaFree(cudaDeviceVelocity);
        cudaFree(cudaDeviceColor);
        cudaFree(cudaDeviceRadius);
        cudaFree(cudaDeviceImageData);
        cudaFree(cudaDeviceElementInPiece);
        cudaFree(cudaDeviceElementInPieceCurrent);
        cudaFree(cudaDeviceNewMemoryToPiece);
    }
}

const Image*
CudaRenderer::getImage() {

    // need to copy contents of the rendered image from device memory
    // before we expose the Image object to the caller

    printf("Copying image data from device\n");

    cudaMemcpy(image->data,
               cudaDeviceImageData,
               sizeof(float) * 4 * image->width * image->height,
               cudaMemcpyDeviceToHost);

    return image;
}

void
CudaRenderer::loadScene(SceneName scene) {
    sceneName = scene;
    loadCircleScene(sceneName, numCircles, position, velocity, color, radius);
}

void
CudaRenderer::setup() {

    int deviceCount = 0;
    std::string name;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Initializing CUDA for CudaRenderer\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++) {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        name = deviceProps.name;

        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n", static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n");
    
    // By this time the scene should be loaded.  Now copy all the key
    // data structures into device memory so they are accessible to
    // CUDA kernels
    //
    // See the CUDA Programmer's Guide for descriptions of
    // cudaMalloc and cudaMemcpy

    cudaMalloc(&cudaDevicePosition, sizeof(float) * 3 * numCircles);
    cudaMalloc(&cudaDeviceVelocity, sizeof(float) * 3 * numCircles);
    cudaMalloc(&cudaDeviceColor, sizeof(float) * 3 * numCircles);
    cudaMalloc(&cudaDeviceRadius, sizeof(float) * numCircles);
    cudaMalloc(&cudaDeviceImageData, sizeof(float) * 4 * image->width * image->height);

    cudaMemcpy(cudaDevicePosition, position, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceVelocity, velocity, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceColor, color, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceRadius, radius, sizeof(float) * numCircles, cudaMemcpyHostToDevice);

    // Initialize parameters in constant memory.  We didn't talk about
    // constant memory in class, but the use of read-only constant
    // memory here is an optimization over just sticking these values
    // in device global memory.  NVIDIA GPUs have a few special tricks
    // for optimizing access to constant memory.  Using global memory
    // here would have worked just as well.  See the Programmer's
    // Guide for more information about constant memory.

    // 具体分块信息
    numPieceWidth = NUM_PIECE_WIDTH;
    numPieceHeight = NUM_PIECE_HEIGHT;
    widthPerPiece = (image->width + numPieceWidth - 1) / numPieceWidth;
    heightPerPiece = (image->height + numPieceHeight - 1) / numPieceHeight;
    printf("widthPerPiece: %d, heightPerPiece: %d\n", widthPerPiece, heightPerPiece);
    cudaError_t cudaStatus;
    cudaStatus = cudaMalloc((void**)&cudaDeviceElementInPiece, sizeof(int) * numPieceHeight * numPieceWidth);
    if (cudaStatus != cudaSuccess) 
        fprintf(stderr, "cudaMalloc failed\n");
    cudaStatus = cudaMalloc((void**)&cudaDeviceElementInPieceCurrent, sizeof(int) * numPieceHeight * numPieceWidth);
    if (cudaStatus != cudaSuccess) 
        fprintf(stderr, "cudaMalloc failed\n");
    cudaStatus = cudaMalloc((void**)&cudaDeviceNewMemoryToPiece, sizeof(int*) * numPieceHeight * numPieceWidth);
    if (cudaStatus != cudaSuccess) 
        fprintf(stderr, "cudaMalloc failed\n");
    cudaMemset(cudaDeviceElementInPiece, 0, sizeof(int) * numPieceHeight * numPieceWidth);
    if (cudaStatus != cudaSuccess) 
        fprintf(stderr, "cudaMemset failed\n");
    cudaMemset(cudaDeviceElementInPieceCurrent, 0, sizeof(int) * numPieceHeight * numPieceWidth);
    if (cudaStatus != cudaSuccess) 
        fprintf(stderr, "cudaMemset failed\n");

    cudaDeviceSynchronize();

    GlobalConstants params;
    params.sceneName = sceneName;
    params.numCircles = numCircles;
    params.imageWidth = image->width;
    params.imageHeight = image->height;
    params.position = cudaDevicePosition;
    params.velocity = cudaDeviceVelocity;
    params.color = cudaDeviceColor;
    params.radius = cudaDeviceRadius;
    params.imageData = cudaDeviceImageData;

    params.numPieceWidth = numPieceWidth;
    params.numPieceHeight = numPieceHeight;
    params.widthPerPiece = widthPerPiece;
    params.heightPerPiece = heightPerPiece;
    params.numElementInPiece = cudaDeviceElementInPiece;
    params.numElementInPieceCurrent = cudaDeviceElementInPieceCurrent;
    params.newMemoryToPiece = cudaDeviceNewMemoryToPiece;

    cudaMemcpyToSymbol(cuConstRendererParams, &params, sizeof(GlobalConstants));

    // also need to copy over the noise lookup tables, so we can
    // implement noise on the GPU
    int* permX;
    int* permY;
    float* value1D;
    getNoiseTables(&permX, &permY, &value1D);
    cudaMemcpyToSymbol(cuConstNoiseXPermutationTable, permX, sizeof(int) * 256);
    cudaMemcpyToSymbol(cuConstNoiseYPermutationTable, permY, sizeof(int) * 256);
    cudaMemcpyToSymbol(cuConstNoise1DValueTable, value1D, sizeof(float) * 256);

    // last, copy over the color table that's used by the shading
    // function for circles in the snowflake demo

    float lookupTable[COLOR_MAP_SIZE][3] = {
        {1.f, 1.f, 1.f},
        {1.f, 1.f, 1.f},
        {.8f, .9f, 1.f},
        {.8f, .9f, 1.f},
        {.8f, 0.8f, 1.f},
    };

    cudaMemcpyToSymbol(cuConstColorRamp, lookupTable, sizeof(float) * 3 * COLOR_MAP_SIZE);

}

// allocOutputImage --
//
// Allocate buffer the renderer will render into.  Check status of
// image first to avoid memory leak.
void
CudaRenderer::allocOutputImage(int width, int height) {

    if (image)
        delete image;
    image = new Image(width, height);
}

// clearImage --
//
// Clear's the renderer's target image.  The state of the image after
// the clear depends on the scene being rendered.
void
CudaRenderer::clearImage() {

    // 256 threads per block is a healthy number
    dim3 blockDim(16, 16, 1);
    dim3 gridDim(
        (image->width + blockDim.x - 1) / blockDim.x,
        (image->height + blockDim.y - 1) / blockDim.y);

    if (sceneName == SNOWFLAKES || sceneName == SNOWFLAKES_SINGLE_FRAME) {
        kernelClearImageSnowflake<<<gridDim, blockDim>>>();
    } else {
        kernelClearImage<<<gridDim, blockDim>>>(1.f, 1.f, 1.f, 1.f);
    }
    cudaDeviceSynchronize();
}

// advanceAnimation --
//
// Advance the simulation one time step.  Updates all circle positions
// and velocities
void
CudaRenderer::advanceAnimation() {
     // 256 threads per block is a healthy number
    dim3 blockDim(256, 1);
    dim3 gridDim((numCircles + blockDim.x - 1) / blockDim.x);

    // only the snowflake scene has animation
    if (sceneName == SNOWFLAKES) {
        kernelAdvanceSnowflake<<<gridDim, blockDim>>>();
    } else if (sceneName == BOUNCING_BALLS) {
        kernelAdvanceBouncingBalls<<<gridDim, blockDim>>>();
    } else if (sceneName == HYPNOSIS) {
        kernelAdvanceHypnosis<<<gridDim, blockDim>>>();
    } else if (sceneName == FIREWORKS) { 
        kernelAdvanceFireWorks<<<gridDim, blockDim>>>(); 
    }
    cudaDeviceSynchronize();                                                                     
}


__global__
void numCircleInPiece() {
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    if (index >= cuConstRendererParams.numCircles) {
        return;
    }
    short imageWidth = cuConstRendererParams.imageWidth;
    short imageHeight = cuConstRendererParams.imageHeight;
    short numPieceWidth = cuConstRendererParams.numPieceWidth;
    short numPieceHeight = cuConstRendererParams.numPieceHeight;
    short widthEachPiece = (imageWidth + numPieceWidth - 1) / numPieceWidth;
    short heightEachPiece = (imageHeight + numPieceHeight - 1) / numPieceHeight;
    // read position and radius
    float3 p = *(float3*)(&cuConstRendererParams.position[index * 3]);
    float  rad = cuConstRendererParams.radius[index];

    short minX = static_cast<short>(imageWidth * (p.x - rad));
    short maxX = static_cast<short>(imageWidth * (p.x + rad)) + 1;
    short minY = static_cast<short>(imageHeight * (p.y - rad));
    short maxY = static_cast<short>(imageHeight * (p.y + rad)) + 1;

    // a bunch of clamps.  Is there a CUDA built-in for this?
    short screenMinX = (minX > 0) ? ((minX < imageWidth) ? minX : imageWidth) : 0;
    short screenMaxX = (maxX > 0) ? ((maxX < imageWidth) ? maxX : imageWidth) : 0;
    short screenMinY = (minY > 0) ? ((minY < imageHeight) ? minY : imageHeight) : 0;
    short screenMaxY = (maxY > 0) ? ((maxY < imageHeight) ? maxY : imageHeight) : 0;

    short pieceMinX = screenMinX / widthEachPiece;
    short pieceMaxX = (screenMaxX - 1) / widthEachPiece + 1;
    short pieceMinY = screenMinY / heightEachPiece;
    short pieceMaxY = (screenMaxY - 1) / heightEachPiece + 1;

    float invWidth = 1.f / imageWidth * widthEachPiece;
    float invHeight = 1.f / imageHeight * heightEachPiece;

    // 遍历像素块
    float boxB = invHeight * pieceMinY;
    for (int pieceY = pieceMinY; pieceY < pieceMaxY; pieceY++, boxB += invHeight) {
        int pieceIndex = pieceY * numPieceWidth + pieceMinX;
        int* val = &cuConstRendererParams.numElementInPiece[pieceIndex];
        float boxL = invWidth * pieceMinX;
        for (int pieceX = pieceMinX; pieceX < pieceMaxX; pieceX++, val++, boxL += invWidth) {
            if (circleInBox(p.x, p.y, rad, boxL, boxL + invWidth, boxB + invHeight, boxB)) {
                atomicAdd(val, 1);
            }
        }
    }
}
// void CudaRenderer::numCircleInPieceCPU(std::vector<int>&cmp_v1, std::vector<std::vector<int>>& cmp_v2) {
//     int imageWidth = image->width;
//     int imageHeight = image->height;
//     int widthEachPiece = (imageWidth + numPieceWidth - 1) / numPieceWidth;
//     int heightEachPiece = (imageHeight + numPieceHeight - 1) / numPieceHeight;
//     // printf("%d, %d, %d, %d, %d, %d\n", imageWidth, imageHeight, widthEachPiece, heightEachPiece, numPieceWidth, numPieceHeight);
//     for (int index = 0;index < this->numCircles;index++) {
//         // read position and radius
//         float3 p = *(float3*)(&position[index * 3]);
//         float  rad = radius[index];

//         int minX = static_cast<short>(imageWidth * (p.x - rad));
//         int maxX = static_cast<short>(imageWidth * (p.x + rad)) + 1;
//         int minY = static_cast<short>(imageHeight * (p.y - rad));
//         int maxY = static_cast<short>(imageHeight * (p.y + rad)) + 1;

//         // a bunch of clamps.  Is there a CUDA built-in for this?
//         int screenMinX = (minX > 0) ? ((minX < imageWidth) ? minX : imageWidth) : 0;
//         int screenMaxX = (maxX > 0) ? ((maxX < imageWidth) ? maxX : imageWidth) : 0;
//         int screenMinY = (minY > 0) ? ((minY < imageHeight) ? minY : imageHeight) : 0;
//         int screenMaxY = (maxY > 0) ? ((maxY < imageHeight) ? maxY : imageHeight) : 0;

//         int pieceMinX = screenMinX / widthEachPiece;
//         int pieceMaxX = (screenMaxX - 1) / widthEachPiece + 1;
//         int pieceMinY = screenMinY / heightEachPiece;
//         int pieceMaxY = (screenMaxY - 1) / heightEachPiece + 1;
//         // printf("%d, %d, %d, %d\n", screenMinX, screenMaxX, screenMinY, screenMaxY);
//         // printf("%d, %d, %d, %d\n", pieceMinX, pieceMaxX, pieceMinY, pieceMaxY);

//         // 遍历像素块
//         for (int pieceY = pieceMinY; pieceY < pieceMaxY; pieceY++) {
//             int pieceIndex = pieceY * numPieceWidth + pieceMinX;
//             // printf("%d\n", pieceIndex);
//             for (int pieceX = pieceMinX; pieceX < pieceMaxX; pieceX++, pieceIndex++) {
//                 // 修改部分
//                 // cmp_v1[pieceIndex]++;
//                 // cmp_v2[pieceIndex].push_back(index);
//             }
//         }
//     }
// }

__global__
void addIndexToNewMemory() {
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    if (index >= cuConstRendererParams.numCircles) {
        return;
    }
    short imageWidth = cuConstRendererParams.imageWidth;
    short imageHeight = cuConstRendererParams.imageHeight;
    short numPieceWidth = cuConstRendererParams.numPieceWidth;
    short numPieceHeight = cuConstRendererParams.numPieceHeight;
    short widthEachPiece = (imageWidth + numPieceWidth - 1) / numPieceWidth;
    short heightEachPiece = (imageHeight + numPieceHeight - 1) / numPieceHeight;
    // read position and radius
    float3 p = *(float3*)(&cuConstRendererParams.position[index * 3]);
    float  rad = cuConstRendererParams.radius[index];

    short minX = static_cast<short>(imageWidth * (p.x - rad));
    short maxX = static_cast<short>(imageWidth * (p.x + rad)) + 1;
    short minY = static_cast<short>(imageHeight * (p.y - rad));
    short maxY = static_cast<short>(imageHeight * (p.y + rad)) + 1;

    // a bunch of clamps.  Is there a CUDA built-in for this?
    short screenMinX = (minX > 0) ? ((minX < imageWidth) ? minX : imageWidth) : 0;
    short screenMaxX = (maxX > 0) ? ((maxX < imageWidth) ? maxX : imageWidth) : 0;
    short screenMinY = (minY > 0) ? ((minY < imageHeight) ? minY : imageHeight) : 0;
    short screenMaxY = (maxY > 0) ? ((maxY < imageHeight) ? maxY : imageHeight) : 0;

    short pieceMinX = screenMinX / widthEachPiece;
    short pieceMaxX = (screenMaxX - 1) / widthEachPiece + 1;
    short pieceMinY = screenMinY / heightEachPiece;
    short pieceMaxY = (screenMaxY - 1) / heightEachPiece + 1;

    float invWidth = 1.f / imageWidth * widthEachPiece;
    float invHeight = 1.f / imageHeight * heightEachPiece;
    // 新的地址空间合集
    int** newMemoryToPiece = cuConstRendererParams.newMemoryToPiece;

    float boxB = invHeight * pieceMinY;
    // 遍历像素块
    for (int pieceY = pieceMinY; pieceY < pieceMaxY; pieceY++, boxB += invHeight) {
        int pieceIndex = pieceY * numPieceWidth + pieceMinX;
        float boxL = invWidth * pieceMinX;
        for (int pieceX = pieceMinX; pieceX < pieceMaxX; pieceX++, pieceIndex++, boxL += invWidth) {
            if (circleInBox(p.x, p.y, rad, boxL, boxL + invWidth, boxB + invHeight, boxB)) {
                int* val = &cuConstRendererParams.numElementInPieceCurrent[pieceIndex];
                int* newMemory = newMemoryToPiece[pieceIndex];
                // 修改部分
                int d = atomicAdd(val, 1);
                newMemory[d] = index;
            }
        }
    }
}
__global__
void sortPerPiece() {
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    int numPieceWidth = cuConstRendererParams.numPieceWidth;
    int numPieceHeight = cuConstRendererParams.numPieceHeight;
    if (index >= numPieceWidth * numPieceHeight) {
        return;
    }
    // 获取地址
    int total = cuConstRendererParams.numElementInPiece[index];
    int *arr = cuConstRendererParams.newMemoryToPiece[index];
    // 特判，会出现一个地方的空间为 0 的情况;
    if (total <= 0) {
        return;
    }
    // 堆排序
    // 变大根堆
    for (int i = total - 1;i >= 0;i--) {
        int root = i;
        while(true) {
            int sonL = (root << 1) + 1;
            int sonR = sonL + 1;
            if (sonL >= total) {
                break;
            }
            if (sonR >= total || arr[sonL] >= arr[sonR]) {
                if (arr[root] < arr[sonL]) {
                    sonR = arr[sonL];
                    arr[sonL] = arr[root];
                    arr[root] = sonR;
                    root = sonL;
                } else {
                    break;
                }
            } else {
                if (arr[root] < arr[sonR]) {
                    sonL = arr[sonR];
                    arr[sonR] = arr[root];
                    arr[root] = sonL;
                    root = sonR;
                } else {
                    break;
                }
            }
        }
    }

    while(--total) {
        int root = arr[0];
        arr[0] = arr[total];
        arr[total] = root;
        root = 0;
        while(true) {
            int sonL = (root << 1) + 1;
            int sonR = sonL + 1;
            if (sonL >= total) {
                break;
            }
            if (sonR >= total || arr[sonL] >= arr[sonR]) {
                if (arr[root] < arr[sonL]) {
                    sonR = arr[sonL];
                    arr[sonL] = arr[root];
                    arr[root] = sonR;
                    root = sonL;
                } else {
                    break;
                }
            } else {
                if (arr[root] < arr[sonR]) {
                    sonL = arr[sonR];
                    arr[sonR] = arr[root];
                    arr[root] = sonL;
                    root = sonR;
                } else {
                    break;
                }
            }
        }
    }
    // for (int i = 1;i < cuConstRendererParams.numElementInPiece[index];i++) {
    //     if (arr[i - 1] >= arr[i]) {
    //         printf("arr[%d - 1] >= arr[%d], arr[i - 1] = %d, arr[i] = %d\n", i, i, arr[i - 1], arr[i]);
    //     }
    //     assert(arr[i - 1] < arr[i]);
    // }
}
__global__ void kernelRenderPixel() {
    int pixelX = blockIdx.x * blockDim.x + threadIdx.x;
    int pixelY = blockIdx.y * blockDim.y + threadIdx.y;
    int imageWidth = cuConstRendererParams.imageWidth;
    int imageHeight = cuConstRendererParams.imageHeight;
    if (pixelX >= imageWidth || pixelY >= imageHeight)
        return;
    int pieceX = pixelX / cuConstRendererParams.widthPerPiece;
    int pieceY = pixelY / cuConstRendererParams.heightPerPiece;

    float invWidth = 1.f / imageWidth;
    float invHeight = 1.f / imageHeight;

    int pieceIndex = pieceX + pieceY * cuConstRendererParams.numPieceWidth;
    assert(pieceIndex < cuConstRendererParams.numPieceWidth * cuConstRendererParams.numPieceHeight);
    int numCircle = cuConstRendererParams.numElementInPiece[pieceIndex];
    int *circleIndexs = cuConstRendererParams.newMemoryToPiece[pieceIndex];

    for (int i = 0; i < numCircle; i++) {
        int index = circleIndexs[i];
        int index3 = 3 * index;
        // read position and radius
        float3 p = *(float3*)(&cuConstRendererParams.position[index3]);
        // float  rad = cuConstRendererParams.radius[index];
        // 像素信息
        float4* imgPtr = (float4*)(&cuConstRendererParams.imageData[4 * (pixelY * imageWidth + pixelX)]);
        float2 pixelCenterNorm = make_float2(invWidth * (static_cast<float>(pixelX) + 0.5f),
                                                 invHeight * (static_cast<float>(pixelY) + 0.5f));
        // 更新像素信息
        shadePixel(index, pixelCenterNorm, p, imgPtr);
    }
}

void
CudaRenderer::render() {
    cudaError_t cudaStatus;
    double startTime, endTime;
    // 将整个图片分块进行(块数不能太小，否则需要更改kernelRenderPixel函数相关的调用)
    printf("---------------------------------------------\n");
    printf("CudaRenderer::render start\n");
    printf("图片将分成 %d * %d 块\n", numPieceWidth, numPieceHeight);
    printf("每块包含 %d * %d 个像素点\n", widthPerPiece, heightPerPiece);

    dim3 blockDimCircle(256, 1);
    dim3 gridDimCircle((numCircles + blockDimCircle.x - 1) / blockDimCircle.x);
    // 圆并行
    printf("---------------------------------------------\n");
    printf("求每个块与之对应的圆的数量(numCircleInPiece)\n");
    // std::vector<int>cmp_v1(1024, 0);
    // std::vector<std::vector<int>> cmp_v2(1024, std::vector<int>());
    // numCircleInPieceCPU(cmp_v1, cmp_v2);
    // printf("numCirclePieceCPU final\n");
    printf("gridDimCircle: %d, blockDimCircle: %d\n", gridDimCircle.x, blockDimCircle.x);
    startTime = CycleTimer::currentSeconds();
    numCircleInPiece<<<gridDimCircle, blockDimCircle>>>();
    cudaDeviceSynchronize();
    endTime = CycleTimer::currentSeconds();
    printf("numCirclePiece final, run time: %.3f ms\n", 1000.f * (endTime - startTime));
    int numPiece = numPieceWidth * numPieceHeight;
    int* elementInPiece = new int[numPiece];
    cudaMemcpy(elementInPiece, cudaDeviceElementInPiece, sizeof(int) * numPiece, cudaMemcpyDeviceToHost);
    // for (int i = 0;i < numPiece;i++) {
    //     printf("第 %d 块: 有 %d 个圆\n", i, elementInPiece[i]);
    // }
    // for(int i = 0;i < numPiece;i++) {
    //     if(elementInPiece[i] != cmp_v1[i]) {
    //         printf("pieceIndex: %d, expect: %d, value: %d\n", i, cmp_v1[i], elementInPiece[i]);
    //         return;
    //     }
    // }
    // printf("result is correct\n\n");

    // 根据结果开辟空间
    printf("---------------------------------------------\n");
    printf("申请新空间(mallocNewMemory, 串行)\n");
    startTime = CycleTimer::currentSeconds();
    int **newMemory = new int*[numPiece];
    for (int i = 0;i < numPieceWidth;i++) {
        for (int j = 0;j < numPieceHeight;j++) {
            int index = i + j * numPieceWidth;
            cudaStatus = cudaMalloc((void**)&newMemory[index], sizeof(int) * elementInPiece[index]);
            // printf("第 %d 块: 新内存开辟完成\n", index);
            if (cudaStatus != cudaSuccess) 
                printf("mallocNewMemory cudaMalloc error: %d\n", cudaStatus);
        }
    }
    cudaStatus = cudaMemcpy(cudaDeviceNewMemoryToPiece, newMemory, sizeof(int*) * numPiece, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) 
        printf("mallocNewMemory cudaMalloc error: %d\n", cudaStatus);
    cudaDeviceSynchronize();
    endTime = CycleTimer::currentSeconds();
    printf("mallocNewMemory final, run time: %.3f ms\n", 1000.f * (endTime - startTime));

    // 二次圆并行
    printf("---------------------------------------------\n");
    printf("将圆的ID添加至新申请的空间(addIndexToNewMemory)\n");
    printf("gridDimCircle: %d, blockDimCircle: %d\n", gridDimCircle.x, blockDimCircle.x);
    startTime = CycleTimer::currentSeconds();
    addIndexToNewMemory<<<gridDimCircle, blockDimCircle>>>();
    cudaDeviceSynchronize();
    endTime = CycleTimer::currentSeconds();
    printf("addIndexToNewMemory final, run time: %.3f ms\n", 1000.f * (endTime - startTime));
    // 这里需要写一点验证函数
    // for (int i = 0;i < numPiece;i++) {
    //     int* arr = new int[elementInPiece[i]];
    //     cudaMemcpy(arr, newMemory[i], sizeof(int) * elementInPiece[i], cudaMemcpyDeviceToHost);
    //     printf("第 %d 块的所有圆 id: ", i);
    //     for (int j = 0;j < elementInPiece[i];j++) {
    //         printf("%d ", arr[j]);
    //     }
    //     printf("\n");
    //     delete[] arr;
    // }
    // printf("result is correct\n\n");

    // 块并行排序
    printf("---------------------------------------------\n");
    printf("每个块对圆id排序(sortPerPiece)\n");
    dim3 blockDimPiece(std::min(256, numPiece), 1);
    dim3 gridDimPiece((numPiece + blockDimPiece.x - 1) / blockDimPiece.x);
    printf("gridDimPiece: %d, blockDimPiece: %d\n", gridDimPiece.x, blockDimPiece.x);
    startTime = CycleTimer::currentSeconds();
    sortPerPiece<<<blockDimPiece, gridDimPiece>>>();
    cudaDeviceSynchronize();
    endTime = CycleTimer::currentSeconds();
    printf("sortPerPiece final, run time: %.3f ms\n", 1000.f * (endTime - startTime));
    // 验证部分
    // for (int i = 0;i < numPiece;i++) {
    //     int* arr = new int[elementInPiece[i]];
    //     cudaMemcpy(arr, newMemory[i], sizeof(int) * elementInPiece[i], cudaMemcpyDeviceToHost);
    //     printf("排序后第 %d 块的所有圆 id: ", i);
    //     for (int j = 0;j < elementInPiece[i];j++) {
    //         printf("%d ", arr[j]);
    //     }
    //     printf("\n");
    //     delete[] arr;
    // }
    // 像素并行计算
    printf("---------------------------------------------\n");
    printf("开始像素并行计算\n");
    dim3 blockDimPixel(std::min(widthPerPiece, 8), std::min(heightPerPiece, 8));
    dim3 gridDimPixel((image->width + blockDimPixel.x - 1) / blockDimPixel.x, 
                      (image->height + blockDimPixel.y - 1) / blockDimPixel.y);
    printf("gridDimPixel.x: %d, blockDimPixel.x: %d\n", gridDimPixel.x, blockDimPixel.x);
    printf("gridDimPixel.y: %d, blockDimPixel.y: %d\n", gridDimPixel.y, blockDimPixel.y);
    startTime = CycleTimer::currentSeconds();
    kernelRenderPixel<<<gridDimPixel, blockDimPixel>>>();
    cudaDeviceSynchronize();
    endTime = CycleTimer::currentSeconds();
    printf("kernelRenderPixel final, run time: %.3f ms\n", 1000.f * (endTime - startTime));
    printf("---------------------------------------------\n");
    // 记得释放空间
    delete[] elementInPiece;
    delete[] newMemory;
}

