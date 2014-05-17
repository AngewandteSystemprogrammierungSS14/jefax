/*
 * scheduler.h
 *
 * Created: 05.05.2014 09:41:31
 *  Author: Fabian
 */ 

#pragma once

#include "tasklist.h"

typedef struct  
{
	void (*init)();
	task_t* (*getNextTask)();
	taskList_t *readyList;
	taskList_t *blockingList;
} scheduler_t;

int initScheduler(scheduler_t *p_defaultScheduler);
void setTaskState(task_t *p_task, taskState_t p_state);
taskState_t getTaskState(const task_t *p_task);
void setScheduler(scheduler_t *p_scheduler);



task_t *getRunningTask();
