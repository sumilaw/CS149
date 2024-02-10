#include "tasksys.h"


IRunnable::~IRunnable() {}

ITaskSystem::ITaskSystem(int num_threads) {}
ITaskSystem::~ITaskSystem() {}

void WorkInfo::init(IRunnable* runnable, int num_total_tasks) {
    this->runnable_ = runnable;
    this->num_total_tasks_ = num_total_tasks;
    this->now_task_id_ = this->num_end_tasks_ = 0;
}

/*
 * ================================================================
 * Serial task system implementation
 * ================================================================
 */

const char* TaskSystemSerial::name() {
    return "Serial";
}

TaskSystemSerial::TaskSystemSerial(int num_threads): ITaskSystem(num_threads) {
}

TaskSystemSerial::~TaskSystemSerial() {}

void TaskSystemSerial::run(IRunnable* runnable, int num_total_tasks) {
    for (int i = 0; i < num_total_tasks; i++) {
        runnable->runTask(i, num_total_tasks);
    }
}

TaskID TaskSystemSerial::runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                          const std::vector<TaskID>& deps) {
    // You do not need to implement this method.
    return 0;
}

void TaskSystemSerial::sync() {
    // You do not need to implement this method.
    return;
}

/*
 * ================================================================
 * Parallel Task System Implementation
 * ================================================================
 */

const char* TaskSystemParallelSpawn::name() {
    return "Parallel + Always Spawn";
}

TaskSystemParallelSpawn::TaskSystemParallelSpawn(int num_threads): ITaskSystem(num_threads) {
    //
    // TODO: CS149 student implementations may decide to perform setup
    // operations (such as thread pool construction) here.
    // Implementations are free to add new class member variables
    // (requiring changes to tasksys.h).
    //
    num_threads_ = num_threads;
}

TaskSystemParallelSpawn::~TaskSystemParallelSpawn() {}

void TaskSystemParallelSpawn::threadRun(IRunnable* runnable, int num_total_tasks, int* task_id, std::mutex& mutex) {
    while(true) {
        mutex.lock();
        int num = (*task_id)++;
        mutex.unlock();
        if (num >= num_total_tasks) {
            break;
        }
        runnable->runTask(num, num_total_tasks);
    }
}

void TaskSystemParallelSpawn::run(IRunnable* runnable, int num_total_tasks) {


    //
    // TODO: CS149 students will modify the implementation of this
    // method in Part A.  The implementation provided below runs all
    // tasks sequentially on the calling thread.
    //
    std::vector<std::thread> thread_pool(num_threads_);
    int task_id = 0;
    std::mutex mutex;
    for (int i = 0; i < num_threads_; i++) {
        thread_pool[i] = std::thread(&TaskSystemParallelSpawn::threadRun, this, runnable, num_total_tasks, &task_id, std::ref(mutex));
    }
    for (int i = 0;i < num_threads_;i++) {
        thread_pool[i].join();
    }
}

TaskID TaskSystemParallelSpawn::runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                                 const std::vector<TaskID>& deps) {
    // You do not need to implement this method.
    return 0;
}

void TaskSystemParallelSpawn::sync() {
    // You do not need to implement this method.
    return;
}

/*
 * ================================================================
 * Parallel Thread Pool Spinning Task System Implementation
 * ================================================================
 */

const char* TaskSystemParallelThreadPoolSpinning::name() {
    return "Parallel + Thread Pool + Spin";
}

TaskSystemParallelThreadPoolSpinning::TaskSystemParallelThreadPoolSpinning(int num_threads): 
    ITaskSystem(num_threads), num_threads_(num_threads), thread_pool_(num_threads) {
    //
    // TODO: CS149 student implementations may decide to perform setup
    // operations (such as thread pool construction) here.
    // Implementations are free to add new class member variables
    // (requiring changes to tasksys.h).
    //

    // thread_pool_ = std::vector<std::thread>(num_threads);
    for (int i = 0; i < num_threads; i++) {
        (thread_pool_)[i] = std::thread(&TaskSystemParallelThreadPoolSpinning::threadRun, this);
    }
}

TaskSystemParallelThreadPoolSpinning::~TaskSystemParallelThreadPoolSpinning() {
    is_end_ = true;
    for (int i = 0;i < num_threads_;i++) {
        (thread_pool_)[i].join();
    }
}

void TaskSystemParallelThreadPoolSpinning::threadRun() {
    while(!is_end_) {
        this->mutex_.lock();
        if (this->work_info_.now_task_id_ == this->work_info_.num_total_tasks_) {
            this->mutex_.unlock();
            continue;
        }
        int num = work_info_.now_task_id_++;
        this->mutex_.unlock();
        // run
        this->runnable_->runTask(num, this->work_info_.num_total_tasks_);
        // add num_end_task_
        this->mutex_.lock();
        this->work_info_.num_end_tasks_ += 1;
        this->mutex_.unlock();
    }
}

void TaskSystemParallelThreadPoolSpinning::run(IRunnable* runnable, int num_total_tasks) {
    //
    // TODO: CS149 students will modify the implementation of this
    // method in Part A.  The implementation provided below runs all
    // tasks sequentially on the calling thread.
    //
    this->runnable_ = runnable;
    mutex_.lock();
    this->work_info_.init(runnable, num_total_tasks);
    mutex_.unlock();

    while(true) {
        mutex_.lock();
        int num_end = work_info_.num_end_tasks_;
        mutex_.unlock();
        if (num_end == num_total_tasks) {
            break;
        }
    }
}

TaskID TaskSystemParallelThreadPoolSpinning::runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                                              const std::vector<TaskID>& deps) {
    // You do not need to implement this method.
    return 0;
}

void TaskSystemParallelThreadPoolSpinning::sync() {
    // You do not need to implement this method.
    return;
}

/*
 * ================================================================
 * Parallel Thread Pool Sleeping Task System Implementation
 * ================================================================
 */

const char* TaskSystemParallelThreadPoolSleeping::name() {
    return "Parallel + Thread Pool + Sleep";
}

TaskSystemParallelThreadPoolSleeping::TaskSystemParallelThreadPoolSleeping(int num_threads): 
    ITaskSystem(num_threads), num_threads_(num_threads), thread_pool_(num_threads) {
    //
    // TODO: CS149 student implementations may decide to perform setup
    // operations (such as thread pool construction) here.
    // Implementations are free to add new class member variables
    // (requiring changes to tasksys.h).
    //
    for (int i = 0; i < num_threads; i++) {
        (thread_pool_)[i] = std::thread(&TaskSystemParallelThreadPoolSleeping::threadRun, this);
    }
}

TaskSystemParallelThreadPoolSleeping::~TaskSystemParallelThreadPoolSleeping() {
    //
    // TODO: CS149 student implementations may decide to perform cleanup
    // operations (such as thread pool shutdown construction) here.
    // Implementations are free to add new class member variables
    // (requiring changes to tasksys.h).
    //
    thread_mutex_.lock();
    is_end_ = true;
    thread_cv_.notify_all();
    thread_mutex_.unlock();
    for (int i = 0;i < num_threads_;i++) {
        thread_pool_[i].join();
    }
}

void TaskSystemParallelThreadPoolSleeping::threadRun() {
    int num;
    while (true) {
        std::unique_lock<std::mutex> lock(thread_mutex_);
        while (this->work_info_.now_task_id_ == this->work_info_.num_total_tasks_) {
            if (is_end_) {
                return;
            }
            thread_cv_.wait(lock);
        }
        num = work_info_.now_task_id_++;
        lock.unlock();
        // run
        this->work_info_.runnable_->runTask(num, this->work_info_.num_total_tasks_);
        // add num_end_task_
        run_mutex_.lock();
        this->work_info_.num_end_tasks_ += 1;
        if (this->work_info_.num_end_tasks_ == this->work_info_.num_total_tasks_) {
            run_cv_.notify_one();
        }
        run_mutex_.unlock();
    }
}

void TaskSystemParallelThreadPoolSleeping::run(IRunnable* runnable, int num_total_tasks) {
    //
    // TODO: CS149 students will modify the implementation of this
    // method in Parts A and B.  The implementation provided below runs all
    // tasks sequentially on the calling thread.
    //
    std::unique_lock<std::mutex> lock(thread_mutex_);
    this->work_info_.init(runnable, num_total_tasks);
    thread_cv_.notify_all();
    run_cv_.wait(lock);
}

TaskID TaskSystemParallelThreadPoolSleeping::runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                                    const std::vector<TaskID>& deps) {


    //
    // TODO: CS149 students will implement this method in Part B.
    //

    return 0;
}

void TaskSystemParallelThreadPoolSleeping::sync() {

    //
    // TODO: CS149 students will modify the implementation of this method in Part B.
    //

    return;
}
