## Part_a

### 遇到的问题:

1、TaskSystemParallelThreadPoolSpinning 类中，变量 work_info_ 只能采用指针的形式，不然会段错误。
此外，如果使用 new WorkInfo 而不是 new WorkInfo() 也会段错误。

解决：

对于第二部分，错误原因是在类定义时有一个变量没初始化，当调用线程函数时会导致判断条件出现问题
new WorkInfo() 和 new WorkInfo 的区别可以理解为一个会把未定义初值的 int 初始化为 0 ，另一个不会。

然后发现对于第一部分也是同样的问题，初始化的值不同，但为什么第一种情况他在第二次创建才出现问题，
可以考虑是当第一次处理完毕后，两个变量本应该是相同的，但是第二次创建后分配的内存跟原来相同，
但是我将其中一个初始化了，另一个没有，导致两个值不相同了，从而产生段错误

所以类内的变量还是都把 {} 写上，防止出错

2、TaskSystemParallelThreadPoolSleeping 不够快，在单个数据计算量小，计算数据多时，速度是官方的 1.5 倍左右，不太理想
以上是 8 线程的情况，如果把线程数加到 32 线程(本机最大线程数)，原本速度慢的比官方的快了，原本快的变慢了，不太懂

解决：暂无


## Part_b

### 遇到的问题:
无

## Writeup Handin
1、Describe your task system implementation (1 page is fine). In additional to a general description of how it works, please make sure you address the following questions:

How did you decide to manage threads? (e.g., did you implement a thread pool?) 

How does your system assign tasks to worker threads? Did you use static or dynamic assignment? 

How did you track dependencies in Part B to ensure correct execution of task graphs? 

采用线程池


