#ifndef _TASKSYS_H
#define _TASKSYS_H
#include <thread>
#include <mutex>
#include <memory>
#include <condition_variable>
#include <unistd.h>
#include <queue>
#include <unordered_map>
#include <assert.h>

#include "itasksys.h"

/*
 * TaskSystemSerial: This class is the student's implementation of a
 * serial task execution engine.  See definition of ITaskSystem in
 * itasksys.h for documentation of the ITaskSystem interface.
 */
class TaskSystemSerial: public ITaskSystem {
    public:
        TaskSystemSerial(int num_threads);
        ~TaskSystemSerial();
        const char* name();
        void run(IRunnable* runnable, int num_total_tasks);
        TaskID runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                const std::vector<TaskID>& deps);
        void sync();
};

/*
 * TaskSystemParallelSpawn: This class is the student's implementation of a
 * parallel task execution engine that spawns threads in every run()
 * call.  See definition of ITaskSystem in itasksys.h for documentation
 * of the ITaskSystem interface.
 */
class TaskSystemParallelSpawn: public ITaskSystem {
    public:
        TaskSystemParallelSpawn(int num_threads);
        ~TaskSystemParallelSpawn();
        const char* name();
        void run(IRunnable* runnable, int num_total_tasks);
        TaskID runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                const std::vector<TaskID>& deps);
        void sync();
};

/*
 * TaskSystemParallelThreadPoolSpinning: This class is the student's
 * implementation of a parallel task execution engine that uses a
 * thread pool. See definition of ITaskSystem in itasksys.h for
 * documentation of the ITaskSystem interface.
 */
class TaskSystemParallelThreadPoolSpinning: public ITaskSystem {
    public:
        TaskSystemParallelThreadPoolSpinning(int num_threads);
        ~TaskSystemParallelThreadPoolSpinning();
        const char* name();
        void run(IRunnable* runnable, int num_total_tasks);
        TaskID runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                const std::vector<TaskID>& deps);
        void sync();
};

/*
 * TaskSystemParallelThreadPoolSleeping: This class is the student's
 * optimized implementation of a parallel task execution engine that uses
 * a thread pool. See definition of ITaskSystem in
 * itasksys.h for documentation of the ITaskSystem interface.
 */
struct WorkInfo {
    IRunnable* runnable_{nullptr};
    int num_total_tasks_{0};
    int now_task_id_{0};    // 这个用 tasks_queue_mutex_ 锁
    int num_end_tasks_{0};  // 这个用 mutex_ 锁

    TaskID task_id_{-1}; // 这个 task_id_ 是对于外部而言的，即对于当前这个 WorkInfo 而言
    size_t num_parent_{0};  // 这个用 tasks_graph_mutex_ 锁
    std::mutex mutex_;
    void init(IRunnable* runnable, int num_total_tasks, int task_id);
};

class TaskSystemParallelThreadPoolSleeping: public ITaskSystem {
    public:
        TaskSystemParallelThreadPoolSleeping(int num_threads);
        ~TaskSystemParallelThreadPoolSleeping();
        const char* name();
        void threadRun();
        void run(IRunnable* runnable, int num_total_tasks);
        TaskID runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                const std::vector<TaskID>& deps);
        void sync();
    private:
        // 线程控制
        int num_threads_;
        size_t num_sleeping_thread_{0};   // 处于休眠状态的线程，由 tasks_queue_mutex_ 控制
        std::vector<std::thread> thread_pool_;

        TaskID next_task_id_{-1};  // 不需要锁
        std::queue<std::shared_ptr<WorkInfo>> tasks_queue_;  // 可以运行的任务
        std::mutex tasks_queue_mutex_;
        std::condition_variable tasks_queue_cv_;

        std::unordered_map<TaskID, std::pair< std::shared_ptr<WorkInfo>, std::shared_ptr<std::vector<TaskID>> >> tasks_graph_;
        std::mutex tasks_graph_mutex_;

        // sync 用的
        std::condition_variable sync_cv_;  
        // 析构用
        bool is_end_{false};
};

#endif
