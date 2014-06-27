/* The scheduler decides which task is next to get the CPU. Scheduling
 * is invoked by the dispatcher.
 * The struct scheduler_t provides an interface to create custom schedulers,
 * which can be set using setScheduler().
 * All callback functions have to be implemented.
 * All functions (return type int) return 0 on success. */

#pragma once

#include <stddef.h>
#include "tasklist.h"

#define TASK_IS_RUNNING(task) (task->state == RUNNING)
#define TASK_IS_READY(task) (task->state == READY)
#define TASK_IS_BLOCKING(task) (task->state == BLOCKING)

/* The init() callback is called during setScheduler(). Ready- and blockinglist should
 * be prepared, so the scheduler can work in a correct way.
 * getNextTask() is invoked to get the next task that will be running. If not task
 * can be found NULL should be returned.
 * taskStateChanged() is invoked if any task changes its state (running, blocking, ready).
 * The callback taskWokeUp() always runs in interrupt context (Timer interrupt). */
typedef struct  
{
	void (*init)();
	task_t* (*getNextTask)();
	void (*taskStateChanged)(task_t*);
	void (*taskWokeUp)(task_t*);
	taskList_t *readyList;
	taskList_t *blockingList;
} scheduler_t;

int initScheduler(scheduler_t *p_defaultScheduler);
task_t* schedule();

void yield();
void sleep(const int p_ms);
void setTaskState(task_t *p_task, taskState_t p_state);

void forceContextSwitch();

void setScheduler(scheduler_t *p_scheduler);

task_t *getRunningTask();
int hasRunningTask();



