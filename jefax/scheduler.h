/*
 * scheduler.h
 *
 * Created: 05.05.2014 09:41:31
 *  Author: Fabian
 */ 

#pragma once

#include "task.h"

typedef enum { RR_SCHEDULER } scheduler_t;

void initScheduler();
void setCurrentTaskState(taskState_t p_state);
void schedule();
task_t* getNextTask();
void setScheduler(scheduler_t p_scheduler);
