/*
 * scheduler.h
 *
 * Created: 05.05.2014 09:41:31
 *  Author: Fabian
 */ 

#pragma once

#include "tasklist.h"
#include "stddef.h"

#define NO_TASK_SCHEDULED() (getRunningTask() == NULL)
#define RUNNING_TASK_IS_RUNNING() (getRunningTask() != NULL && getRunningTask()->state == RUNNING)
#define RUNNING_TASK_IS_READY() (getRunningTask() != NULL && getRunningTask()->state == READY)
#define RUNNING_TASK_IS_BLOCKING() (getRunningTask() != NULL && getRunningTask()->state == BLOCKING)

/* Function "taskWokeUp" always runs in interrupt context (Timer interrupt). */
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
taskState_t getTaskState(const task_t *p_task);

void forceContextSwitch();

void setScheduler(scheduler_t *p_scheduler);

task_t *getRunningTask();



