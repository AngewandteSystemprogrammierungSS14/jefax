/*
 * scheduler.c
 *
 * Created: 05.05.2014 09:45:40
 *  Author: Fabian
 */ 

#include "scheduler.h"
#include <stdlib.h>
#include <assert.h>
#include <util/atomic.h>

#define DEF_TASK_LIST_SIZE 20

struct taskList
{
	task_t *elements[DEF_TASK_LIST_SIZE];
	int size;
	int count;
};

extern task_t TASKS[];
static task_t *runningTask;
static struct taskList readyTasks;
static struct taskList blockingTasks;
static scheduler_t scheduler;

/* prototypes */
static void setRRScheduler();
static task_t* getNextTaskRR();
static int getRRIndex(struct taskList *p_tasks, task_t *p_task);
static void updateBlockingTasksRR();
static void removeTask(struct taskList *p_tasks, int p_index);
static void insertRRTask(struct taskList *p_tasks, task_t *p_task);
static void insertTaskAt(struct taskList *p_tasks, task_t *p_task, int p_index);
static void addTask(struct taskList *p_tasks, task_t *p_task);

/* callbacks */
task_t* (*getNextTaskCB)();

void initScheduler()
{
	setScheduler(RR_SCHEDULER);
	readyTasks.size = DEF_TASK_LIST_SIZE;
	readyTasks.count = 0;
	
	blockingTasks.size = DEF_TASK_LIST_SIZE;
	blockingTasks.count = 0;
	
	//add tasks to scheduler
	int taskCount = countTasks();
	int i;
	for(i = 0; i < taskCount; ++i)
	{
		TASKS[i].state = READY;
		insertRRTask(&readyTasks, &TASKS[i]); 
	}
}

void setCurrentTaskState(taskState_t p_state)
{
	setTaskState(runningTask, p_state);
}

void setTaskState(task_t *p_task, taskState_t p_state)
{
	ATOMIC_BLOCK(ATOMIC_RESTORESTATE)
	{
		p_task->state = p_state;
		//check if high prior task got ready
		if(p_task->state == READY && p_task->priority < runningTask->priority)
			runningTask->state = READY;
	}
	
	if(runningTask->state != RUNNING)
		schedule();
}

void schedule()
{
	
	// create interrupt
	TCC0.CNT = TCC0.PER - 1;
	// wait to be exchanged
	while(runningTask->state != RUNNING)
	{ }
}

task_t* getNextTask()
{
	runningTask = getNextTaskCB();
	runningTask->state = RUNNING;
	
	return runningTask;
}

void setScheduler(scheduler_t p_scheduler)
{
	scheduler = p_scheduler;
	switch(scheduler)
	{
		case RR_SCHEDULER:
		setRRScheduler();
		break;
	}
}

static void setRRScheduler()
{
	getNextTaskCB = getNextTaskRR;
}

static task_t* getNextTaskRR()
{
	task_t *result;
	
	updateBlockingTasksRR();
	
	//ready list is empty
	if(readyTasks.count <= 0)
	{
		if(runningTask->state == BLOCKING)
			result = getIdleTask();
		else
			result = runningTask;
	}
	else
	{
		// get next task with highest priority
		result = readyTasks.elements[readyTasks.count - 1];
		--readyTasks.count;
		
		if(runningTask->state == RUNNING)
			runningTask->state = READY;
	}
		
	//put runningTask in correct List
	if(runningTask->state == READY)
		insertRRTask(&readyTasks, runningTask);
	else if(runningTask->state == BLOCKING)
		addTask(&blockingTasks, runningTask);
	
	return result;
}

static void updateBlockingTasksRR()
{
	int i;
	for(i = 0; i < blockingTasks.count; ++i)
	{
		if(blockingTasks.elements[i]->state != BLOCKING)
		{
			insertRRTask(&readyTasks, blockingTasks.elements[i]);
			removeTask(&blockingTasks, i);
		}
	}
}

static void removeTask(struct taskList *p_tasks, int p_index)
{
	assert(p_index < p_tasks->count);
	
	int i;
	
	for(i = 0; i < p_tasks->count - 1; ++i)
		p_tasks->elements[i] = p_tasks->elements[i + 1];
		
	--p_tasks->count;
}


static void insertRRTask(struct taskList *p_tasks, task_t *p_task)
{
	int index;
	
	assert(p_tasks->count < p_tasks->size);
	
	index = getRRIndex(p_tasks, p_task);
	
	insertTaskAt(p_tasks, p_task, index);
}

static int getRRIndex(struct taskList *p_tasks, task_t *p_task)
{
	int result;
	
	if(p_tasks->count == 0)
		return 0;
		
	result = p_tasks->count / 2;
	if(p_tasks->elements[result]->priority <= p_task->priority)
		result = 0;
	
	while(p_tasks->elements[result]->priority > p_task->priority)
		++result;
		
	return result;
}

static void insertTaskAt(struct taskList *p_tasks, task_t *p_task, int p_index)
{
	assert(p_tasks->count < p_tasks->size);
	
	for(int i = p_tasks->count; i > p_index; --i)
	p_tasks->elements[i] = p_tasks->elements[i - 1];
	
	p_tasks->elements[p_index] = p_task;
	++p_tasks->count;
}

static void addTask(struct taskList *p_tasks, task_t *p_task)
{
	assert(p_tasks->count < p_tasks->size);
	
	p_tasks->elements[p_tasks->count] = p_task;
	++p_tasks->count;
}

