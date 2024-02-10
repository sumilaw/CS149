#include "tasksys.h"


IRunnable::~IRunnable() {}

ITaskSystem::ITaskSystem(int num_threads) {}
ITaskSystem::~ITaskSystem() {}

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
    for (int i = 0; i < num_total_tasks; i++) {
        runnable->runTask(i, num_total_tasks);
    }

    return 0;
}

void TaskSystemSerial::sync() {
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
    // NOTE: CS149 students are not expected to implement TaskSystemParallelSpawn in Part B.
}

TaskSystemParallelSpawn::~TaskSystemParallelSpawn() {}

void TaskSystemParallelSpawn::run(IRunnable* runnable, int num_total_tasks) {
    // NOTE: CS149 students are not expected to implement TaskSystemParallelSpawn in Part B.
    for (int i = 0; i < num_total_tasks; i++) {
        runnable->runTask(i, num_total_tasks);
    }
}

TaskID TaskSystemParallelSpawn::runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                                 const std::vector<TaskID>& deps) {
    // NOTE: CS149 students are not expected to implement TaskSystemParallelSpawn in Part B.
    for (int i = 0; i < num_total_tasks; i++) {
        runnable->runTask(i, num_total_tasks);
    }

    return 0;
}

void TaskSystemParallelSpawn::sync() {
    // NOTE: CS149 students are not expected to implement TaskSystemParallelSpawn in Part B.
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

TaskSystemParallelThreadPoolSpinning::TaskSystemParallelThreadPoolSpinning(int num_threads): ITaskSystem(num_threads) {
    // NOTE: CS149 students are not expected to implement TaskSystemParallelThreadPoolSpinning in Part B.
}

TaskSystemParallelThreadPoolSpinning::~TaskSystemParallelThreadPoolSpinning() {}

void TaskSystemParallelThreadPoolSpinning::run(IRunnable* runnable, int num_total_tasks) {
    // NOTE: CS149 students are not expected to implement TaskSystemParallelThreadPoolSpinning in Part B.
    for (int i = 0; i < num_total_tasks; i++) {
        runnable->runTask(i, num_total_tasks);
    }
}

TaskID TaskSystemParallelThreadPoolSpinning::runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                                              const std::vector<TaskID>& deps) {
    // NOTE: CS149 students are not expected to implement TaskSystemParallelThreadPoolSpinning in Part B.
    for (int i = 0; i < num_total_tasks; i++) {
        runnable->runTask(i, num_total_tasks);
    }

    return 0;
}

void TaskSystemParallelThreadPoolSpinning::sync() {
    // NOTE: CS149 students are not expected to implement TaskSystemParallelThreadPoolSpinning in Part B.
    return;
}

/*
 * ================================================================
 * Parallel Thread Pool Sleeping Task System Implementation
 * ================================================================
 */

void WorkInfo::init(IRunnable* runnable, int num_total_tasks, int task_id_) {
    this->runnable_ = runnable;
    this->num_total_tasks_ = num_total_tasks;
    this->now_task_id_ = this->num_end_tasks_ = 0;
    this->task_id_ = task_id_;
}

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
    tasks_queue_mutex_.lock();
    is_end_ = true;
    tasks_queue_cv_.notify_all();
    tasks_queue_mutex_.unlock();
    for (int i = 0;i < num_threads_;i++) {
        thread_pool_[i].join();
    }
}

void TaskSystemParallelThreadPoolSleeping::threadRun() {
    while (true) {
        std::unique_lock<std::mutex> lock(tasks_queue_mutex_);  // 拿队列锁
        while (tasks_queue_.empty()) {
            // 队列为空就睡
            if (is_end_) {
                return;
            }
            num_sleeping_thread_ += 1;
            tasks_queue_cv_.wait(lock);
            num_sleeping_thread_ -= 1;
        }
        // 处理队列，确保队首还没到最大任务数，这点由之后的代码保证
        auto work_info = tasks_queue_.front();
        // work_info->mutex_.lock();  // TODO(nbh): 可能可以不加锁
        int now_task_id = work_info->now_task_id_++;
        // work_info->mutex_.unlock();
        assert(now_task_id < work_info->num_total_tasks_);
        // 如果当前 work_info 已经没有任务可以运行了，那么就将其从队列驱逐出去，
        if (now_task_id == work_info->num_total_tasks_ - 1) {
            tasks_queue_.pop();
        }
        // 解开队列锁
        lock.unlock();
        // 运行
        work_info->runnable_->runTask(now_task_id, work_info->num_total_tasks_);
        // 更改 work_info 的 num_end_tasks_
        work_info->mutex_.lock();
        work_info->num_end_tasks_ += 1;
        if (work_info->num_end_tasks_ == work_info->num_total_tasks_) {
            // 这里应该是只能到一次的，对于每个 work_info 而言
            // 更改 tasks_graph_，删去当前 work_info 信息;
            TaskID task_id = work_info->task_id_;
            // 图锁
            tasks_graph_mutex_.lock();
            auto iter = tasks_graph_.find(task_id);
            // 确保存在
            assert(iter != tasks_graph_.end());
            for (auto son : *iter->second.second) {
                auto son_iter = tasks_graph_.find(son);
                // 对于子节点也确保存在
                assert (son_iter != tasks_graph_.end());
                auto son_work_info = son_iter->second.first;
                // TODO(nbh): 考虑是否需要加锁(这里是否加锁与run中的实现相关)
                assert(son_work_info->num_parent_ > 0);
                son_work_info->num_parent_ -= 1;
                if (son_work_info->num_parent_ == 0) {
                    // 加入等待队列
                    lock.lock();
                    tasks_queue_.emplace(son_work_info);
                    if (num_sleeping_thread_) {
                        tasks_queue_cv_.notify_all();
                    }
                    lock.unlock();
                }
            }
            // 删去当前 work_info
            tasks_graph_.erase(iter);
            // 如果为空尝试唤醒
            if (tasks_graph_.empty()) {
                sync_cv_.notify_one();
            }
            tasks_graph_mutex_.unlock();
        }
        work_info->mutex_.unlock();
    }
}

void TaskSystemParallelThreadPoolSleeping::run(IRunnable* runnable, int num_total_tasks) {


    //
    // TODO: CS149 students will modify the implementation of this
    // method in Parts A and B.  The implementation provided below runs all
    // tasks sequentially on the calling thread.
    //
    auto work_info = std::make_shared<WorkInfo>();
    work_info->init(runnable, num_total_tasks, ++next_task_id_);
    tasks_graph_[next_task_id_] = {work_info, std::make_shared<std::vector<TaskID>>()};
    tasks_queue_mutex_.lock();
    // 把自己加入图中
    // 加入至队列
    tasks_queue_.emplace(work_info);
    if (num_sleeping_thread_) {
        tasks_queue_cv_.notify_all();
    }
    tasks_queue_mutex_.unlock();
    // 等待结束
    this->sync();
}

TaskID TaskSystemParallelThreadPoolSleeping::runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                                    const std::vector<TaskID>& deps) {


    //
    // TODO: CS149 students will implement this method in Part B.
    //
    auto work_info = std::make_shared<WorkInfo>();
    work_info->init(runnable, num_total_tasks, ++next_task_id_);
    tasks_graph_mutex_.lock();
    int num_parent = 0;
    for (auto v : deps) {
        // 确保依赖的任务是之前的任务
        assert(v < next_task_id_);
        // 在父节点处加上自己，更新自己的入度
        auto iter = tasks_graph_.find(v);
        if (iter != tasks_graph_.end()) {
            num_parent += 1;
            iter->second.second->emplace_back(next_task_id_);
        }
    }
    work_info->num_parent_ = num_parent;
    // 把自己加入图中
    tasks_graph_[next_task_id_] = {work_info, std::make_shared<std::vector<TaskID>>()};
    // std::lock_guard<std::mutex>(work_info->mutex_);  // TODO(nbh): 感觉可以不锁
    tasks_graph_mutex_.unlock();

    if (!num_parent) {
        // 没有还没处理的依赖节点，直接进入运行队列等待
        tasks_queue_mutex_.lock();
        if (num_sleeping_thread_) {
            tasks_queue_cv_.notify_all();
        }
        tasks_queue_.emplace(work_info);
        tasks_queue_mutex_.unlock();
    }

    return work_info->task_id_;
}

void TaskSystemParallelThreadPoolSleeping::sync() {

    //
    // TODO: CS149 students will modify the implementation of this method in Part B.
    //
    std::unique_lock<std::mutex> lock(tasks_graph_mutex_);
    while (!tasks_graph_.empty()) {
        sync_cv_.wait(lock);
    }
    return;

}
