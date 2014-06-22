/*
 * scheduler.h
 *
 * Created: 05.05.2014 09:41:31
 *  Author: Fabian
 */ 

#pragma once

#include "tasklist.h"
#include "stddef.h"

#define TASK_IS_RUNNING(task) (task->state == RUNNING)
#define TASK_IS_READY(task) (task->state == READY)
#define TASK_IS_BLOCKING(task) (task->state == BLOCKING)

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

void forceContextSwitch();

void setScheduler(scheduler_t *p_scheduler);

task_t *getRunningTask();
int hasRunningTask();



