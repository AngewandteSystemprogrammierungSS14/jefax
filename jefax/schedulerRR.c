/*
 * schedulerRR.c
 *
 * Created: 17.05.2014 11:13:59
 *  Author: Fabian
 */ 

#include "schedulerRR.h"
#include "stddef.h"
#include "utils.h"
#include <avr/interrupt.h>

static void initSchedulerRR();
static task_t* getNextTaskRR();
static void readyUpBlockingTasksRR();
static int insertTaskRR(taskList_t *p_tasks, task_t *p_task);
static int getInsertIndexRR(taskList_t *p_tasks, task_t *p_task);
static void taskStateChangedRR(task_t* p_task);
static void taskWokeUpRR(task_t* p_task);

static scheduler_t schedulerRR = { initSchedulerRR, getNextTaskRR, taskStateChangedRR, taskWokeUpRR, NULL, NULL };

scheduler_t *getRRScheduler()
{
	return &schedulerRR;
}

static void initSchedulerRR()
{
	sortPriority(schedulerRR.readyList);
}

static task_t* getNextTaskRR()
{
	task_t *result;
	
	readyUpBlockingTasksRR();
	
	//ready list is empty
	if(isEmpty(schedulerRR.readyList))
	{
		if(NO_TASK_SCHEDULED() || RUNNING_TASK_IS_BLOCKING())
			result = NULL;
		else
		{
			getRunningTask()->state = RUNNING;
			result = getRunningTask();
		}
	}
	else
	{
		// get next task with highest priority
		result = getLast(schedulerRR.readyList);
		
		//next task would have lower prio, keep running task
		if(getRunningTask()->state == RUNNING && result->priority > getRunningTask()->priority)
			result = getRunningTask();
		else
		{
			popTaskBack(schedulerRR.readyList);
			
			if(RUNNING_TASK_IS_RUNNING())
				getRunningTask()->state = READY;
		}
	}
		
	//put runningTask in correct List
	if(RUNNING_TASK_IS_READY())
		insertTaskRR(schedulerRR.readyList, getRunningTask());
	else if(RUNNING_TASK_IS_BLOCKING())
		pushTaskBack(schedulerRR.blockingList, getRunningTask());
	
	return result;
}

static void readyUpBlockingTasksRR()
{
	volatile int i;
	for(i = 0; i < schedulerRR.blockingList->count; ++i)
	{
		if(schedulerRR.blockingList->elements[i]->state != BLOCKING)
		{
			insertTaskRR(schedulerRR.readyList, schedulerRR.blockingList->elements[i]);
			removeTask(schedulerRR.blockingList, i);
		}
	}
}

static int insertTaskRR(taskList_t *p_tasks, task_t *p_task)
{	
	int index = getInsertIndexRR(p_tasks, p_task);
	return insertTask(p_tasks, p_task, index);
}

static int getInsertIndexRR(taskList_t *p_tasks, task_t *p_task)
{
	int result;
	
	if(p_tasks->count == 0)
		return 0;
		
	result = p_tasks->count / 2;
	if(p_tasks->elements[result]->priority < p_task->priority)
		result = 0;

	while(result < p_tasks->count && p_tasks->elements[result]->priority > p_task->priority)
		++result;
	
	while(p_tasks->elements[result]->priority == p_task->priority)
		--result;
	
	return result;
}

static void taskStateChangedRR(task_t* p_task)
{
	//check if high prior task got ready
	if(p_task->state == READY && p_task->priority < getRunningTask()->priority)
		getRunningTask()->state = READY;
	if(getRunningTask()->state != RUNNING)
		forceContextSwitch();
}

static void taskWokeUpRR(task_t* p_task)
{
	if(p_task->priority < getRunningTask()->priority)
		FORCE_INTERRUPT(TCC0);
}