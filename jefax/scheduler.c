/*
 * scheduler.c
 *
 * Created: 05.05.2014 09:45:40
 *  Author: Fabian
 */ 

#include "scheduler.h"
#include "dispatcher.h"
#include <stdlib.h>
#include <util/atomic.h>

/* prototypes */
static int initSchedulerTasks();
static int initTaskLists();
static int schedule();
static task_t *selectNextTask();
static void forceContextSwitch();
static int idleTaskFunction();

extern task_t TASKS[];

static task_t *runningTask;
static taskList_t readyList;
static taskList_t blockingList;
static scheduler_t *scheduler;

static task_t idleTask;
static task_t schedulerTask;

int initScheduler(scheduler_t *p_defaultScheduler)
{
	int ret = initSchedulerTasks();
	if(ret)
		return ret;
	
	ret = initTaskLists();
	if(ret)
		return ret;
		
	setScheduler(p_defaultScheduler);
	initDispatcher(&schedulerTask);
	
	return 0;
}

static int initSchedulerTasks()
{
	schedulerTask.function = schedule;
	schedulerTask.priority = 255;
	schedulerTask.state = READY;
	schedulerTask.stackpointer = 0;
	
	idleTask.function = idleTaskFunction;
	idleTask.priority = 255;
	idleTask.state = READY;
	idleTask.stackpointer = 0;
	
	initTask(&schedulerTask);
	initTask(&idleTask);
	
	return 0;
}

static int initTaskLists()
{
	if(initTaskList(&readyList))
		return -1;
	if(initTaskList(&blockingList))
		return -1;
	
	int taskCount = countTasks();
	if(taskCount <= 0)
		return -1;
	
	runningTask = &(TASKS[0]);
	runningTask->state = RUNNING;
	
	int i;
	for(i = 1; i < taskCount; ++i)
	{
		TASKS[i].state = READY;
		pushTaskBack(&readyList, &TASKS[i]);
	}
	
	return 0;
}

static int schedule()
{
	task_t *task = selectNextTask();
	dispatch(task);
	return 0;
}

static task_t *selectNextTask()
{
	task_t *result;
	runningTask = scheduler->getNextTask();
	if(runningTask == NULL)
		result = &idleTask;
	else
		result = runningTask;
	result->state = RUNNING;
	return result;
}

void setTaskState(task_t *p_task, taskState_t p_state)
{
	ATOMIC_BLOCK(ATOMIC_RESTORESTATE)
	{
		p_task->state = p_state;
		//check if high prior task got ready
		if(p_task->state == READY && p_task->priority < runningTask->priority)
			runningTask->state = READY;
		if(runningTask->state != RUNNING)
			forceContextSwitch();
	}		
}

static void forceContextSwitch()
{
	uint8_t state = SREG & 0x80;
	// create interrupt
	sei();
	
	TCC0.CNT = TCC0.PER - 1;
	
	// wait to be exchanged
	while(getTaskState(runningTask) != RUNNING)
	{ }
	
	if(!state)
		cli();
}

taskState_t getTaskState(const task_t *p_task)
{
	taskState_t result;
	ATOMIC_BLOCK(ATOMIC_RESTORESTATE)
	{
		result = p_task->state;
	}
	return result;
}

void setScheduler(scheduler_t *p_scheduler)
{
	scheduler = p_scheduler;
	scheduler->readyList = &readyList;
	scheduler->blockingList = &blockingList;
	scheduler->init();
}

task_t *getRunningTask()
{
	return runningTask;
}

static int idleTaskFunction()
{
	uint8_t led = 0;
	
	while (1) {
		setLED(~(1 << led++));
		//_delay_ms(500);
		if (led == 8)
			led = 0;
	}
	
	return 0;
}

