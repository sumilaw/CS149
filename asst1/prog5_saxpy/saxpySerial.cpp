#include <stdio.h>
#include <thread>

typedef struct {
    float scale;
    float* X;
    float* Y;
    float* result;
    int start;
    int num;
} WorkerArgs;

void saxpySerial(int N,
                       float scale,
                       float X[],
                       float Y[],
                       float result[])
{

    for (int i=0; i<N; i++) {
        result[i] = scale * X[i] + Y[i];
    }
}

void saxpyThreadStart(WorkerArgs * const args)
{
    for (int i=args->start; i<args->start + args->num; i++) {
        args->result[i] = args->scale * args->X[i] + args->Y[i];
    }
}


void saxpyThread(int N,
                       float scale,
                       float X[],
                       float Y[],
                       float result[])
{
    static constexpr int NUM_THREADS = 16;

    // Creates thread objects that do not yet represent a thread.
    std::thread workers[NUM_THREADS];
    WorkerArgs args[NUM_THREADS];

    for (int i=0; i<NUM_THREADS; i++) {
      
        // TODO FOR CS149 STUDENTS: You may or may not wish to modify
        // the per-thread arguments here.  The code below copies the
        // same arguments for each thread
        args[i].scale = scale;
        args[i].X = X;
        args[i].Y = Y;
        args[i].num = N / NUM_THREADS;
        args[i].start = args[i].num * i;
        args[i].result = result;
    }

    for (int i = 0;i < NUM_THREADS; i++) {
        workers[i] = std::thread(saxpyThreadStart, &args[i]);
    }
    for (int i = 0;i < NUM_THREADS; i++) {
        workers[i].join();
    }
}