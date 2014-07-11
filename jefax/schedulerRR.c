#include <avr/interrupt.h>
#include <stddef.h>
#include "schedulerRR.h"
#include "utils.h"

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
	// sorts tasks depending on their priority (ascending)
	sortPriority(schedulerRR.readyList);
}

static task_t* getNextTaskRR()
{
	task_t *result;
	
	readyUpBlockingTasksRR();
	
	//ready list is empty
	if(isEmpty(schedulerRR.readyList)) {
		// runningTask cannot keep running return NULL
		if(!hasRunningTask() || TASK_IS_BLOCKING(getRunningTask()))
			result = NULL;
		else {
			getRunningTask()->state = RUNNING;
			result = getRunningTask();
		}
	} else {
		// get next task with highest priority
		result = getLast(schedulerRR.readyList);
		
		//next task would have lower prio, keep running task
		if(hasRunningTask() && !TASK_IS_BLOCKING(getRunningTask()) && result->priority > getRunningTask()->priority) {
			getRunningTask()->state = RUNNING;
			result = getRunningTask();
		} else {
			popTaskBack(schedulerRR.readyList);
			
			if(hasRunningTask() && TASK_IS_RUNNING(getRunningTask()))
				getRunningTask()->state = READY;
		}
	}
		
	//put runningTask in correct List
	if(hasRunningTask() && TASK_IS_READY(getRunningTask()))
		insertTaskRR(schedulerRR.readyList, getRunningTask());
	else if(hasRunningTask() && TASK_IS_BLOCKING(getRunningTask()))
		pushTaskBack(schedulerRR.blockingList, getRunningTask());
	
	return result;
}

static void readyUpBlockingTasksRR()
{
	volatile int i;
	// for all blocking task check if their state changed to non blocking
	for(i = 0; i < schedulerRR.blockingList->count; ++i) {
		if(!TASK_IS_BLOCKING(schedulerRR.blockingList->elements[i])) {
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
			
	result = 0;
	// find correct index for corresponding priority
	while(result < p_tasks->count && p_tasks->elements[result]->priority > p_task->priority)
		++result;
	
	return result;
}

static void taskStateChangedRR(task_t* p_task)
{
	//check if there is any running task, check if high prior task got ready
	if(hasRunningTask() && TASK_IS_READY(p_task) && p_task->priority < getRunningTask()->priority)
		getRunningTask()->state = READY;
	// if running task is not in running state anymore switch task
	if(!hasRunningTask() || !TASK_IS_RUNNING(getRunningTask()))
		forceContextSwitch();
}

/* runs in interrupt context, so calling forceContextSwitch() is not possible */
static void taskWokeUpRR(task_t* p_task)
{
	if(!hasRunningTask() || (p_task->priority < getRunningTask()->priority))
		FORCE_INTERRUPT(TCC0);
}