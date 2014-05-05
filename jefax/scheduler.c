/*
 * scheduler.c
 *
 * Created: 05.05.2014 09:45:40
 *  Author: Fabian
 */ 

#include "scheduler.h"
#include <stdlib.h>
#include <assert.h>

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

static void setTaskState(task_t *p_task, taskState_t p_state);
static task_t* getNextTaskRR();
static void updateBlockingTasksRR();
static void removeTask(struct taskList *p_tasks, int p_index);
static void addRRTask(struct taskList *p_tasks, task_t *p_task);
static void addTaskAt(struct taskList *p_tasks, task_t *p_task, int p_index);
static void addTask(struct taskList *p_tasks, task_t *p_task);

void initScheduler()
{
	scheduler = RR_SCHEDULER;
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
		addRRTask(&readyTasks, &TASKS[i]); 
	}
}

void setCurrentTaskState(taskState_t p_state)
{
	setTaskState(runningTask, p_state);
}

void setTaskState(task_t *p_task, taskState_t p_state)
{
	p_task->state = p_state;
}

void schedule()
{
	// wait to be exchanged
	// TODO create Interrupt
	while(runningTask->state != RUNNING)
	{ }
}

task_t* getNextTask()
{
	task_t *result = runningTask;
	switch(scheduler)
	{
		case RR_SCHEDULER:
			result = getNextTaskRR();
			break;
	}
	
	return result;
}

void setScheduler(scheduler_t p_scheduler)
{
	scheduler = p_scheduler;
}

task_t* getNextTaskRR()
{
	task_t *result;
	
	updateBlockingTasksRR();
	
	//ready list is empty
	if(readyTasks.count <= 0)
		result = runningTask;
	else
	{
		// get next task with highest prio
		result = readyTasks.elements[readyTasks.count - 1];
		--readyTasks.count;
		
		if(runningTask->state == RUNNING)
			runningTask->state = READY;
	}
		
	//put runningTask in correct List
	if(runningTask->state == READY)
		addRRTask(&readyTasks, runningTask);
	else if(runningTask->state == BLOCKING)
		addTask(&blockingTasks, runningTask);
	
	return result;
}

void updateBlockingTasksRR()
{
	int i;
	for(i = 0; i < blockingTasks.count; ++i)
	{
		if(blockingTasks.elements[i]->state != BLOCKING)
		{
			addRRTask(&readyTasks, blockingTasks.elements[i]);
			removeTask(&blockingTasks, i);
		}
	}
}

void removeTask(struct taskList *p_tasks, int p_index)
{
	assert(p_index < p_tasks->count);
	
	int i;
	
	for(i = 0; i < p_tasks->count - 1; ++i)
		p_tasks->elements[i] = p_tasks->elements[i + 1];
		
	--p_tasks->count;
}


void addRRTask(struct taskList *p_tasks, task_t *p_task)
{
	int index;
	int left = 0;
	int right = p_tasks->count - 1;
	
	assert(p_tasks->count < p_tasks->size);
	
	//find sorted index
	while(left <= right)
	{
		index = (left + right) / 2;
		if(p_tasks->elements[index]->priority == p_task->priority)
		{
			while(p_tasks->elements[index]->priority == p_task->priority)
				--index;
			break;
		}
		else if(p_tasks->elements[index]->priority >  p_task->priority)
			left = index - 1;
		else
			right = index + 1;
	}
	
	addTaskAt(p_tasks, p_task, index);
}

void addTaskAt(struct taskList *p_tasks, task_t *p_task, int p_index)
{
	assert(p_tasks->count < p_tasks->size);
	
	for(int i = p_tasks->count; i > p_index; --i)
	p_tasks->elements[i] = p_tasks->elements[i - 1];
	
	p_tasks->elements[p_index] = p_task;
	++p_tasks->count;
}

void addTask(struct taskList *p_tasks, task_t *p_task)
{
	assert(p_tasks->count < p_tasks->size);
	
	p_tasks->elements[p_tasks->count] = p_task;
	++p_tasks->count;
}

