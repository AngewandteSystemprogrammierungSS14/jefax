/*
 * schedulerRR.c
 *
 * Created: 17.05.2014 11:13:59
 *  Author: Fabian
 */ 

#include "schedulerRR.h"
#include "stddef.h"

static void initSchedulerRR();
static task_t* getNextTaskRR();
static void readyUpBlockingTasksRR();
static int insertTaskRR(taskList_t *p_tasks, task_t *p_task);
static int getInsertIndexRR(taskList_t *p_tasks, task_t *p_task);

static scheduler_t schedulerRR = { initSchedulerRR, getNextTaskRR, NULL, NULL };

scheduler_t *getRRScheduler()
{
	return &schedulerRR;
}

static void initSchedulerRR()
{
	//sortPriority(schedulerRR.readyList);
}

static task_t* getNextTaskRR()
{
	task_t *result;
	
	readyUpBlockingTasksRR();
	
	//ready list is empty
	if(schedulerRR.readyList->count <= 0)
	{
		if(getRunningTask()->state == BLOCKING)
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
		result = popTaskBack(schedulerRR.readyList);
		
		if(getRunningTask()->state == RUNNING)
			getRunningTask()->state = READY;
	}
		
	//put runningTask in correct List
	if(getRunningTask()->state == READY)
		insertTaskRR(schedulerRR.readyList, getRunningTask());
	else if(getRunningTask()->state == BLOCKING)
		pushTaskBack(schedulerRR.blockingList, getRunningTask());
	
	return result;
}

static void readyUpBlockingTasksRR()
{
	int i;
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
	if(p_tasks->elements[result]->priority <= p_task->priority)
		result = 0;
	result = 0;
	while(result < p_tasks->count && p_tasks->elements[result]->priority > p_task->priority)
		++result;
		
	return result;
}