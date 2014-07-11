/* The tasklist provides several functions to manage task_t structs in a list.
 * This is used by the scheduler and locking mechanisms. */

#pragma once

#include "task.h"

#ifndef TASK_LIST_SIZE
	#define TASK_LIST_SIZE 10
#endif

typedef struct
{
	task_t *elements[TASK_LIST_SIZE];
	int size;
	int count;
} taskList_t;

int initTaskList(taskList_t *p_list);

int pushTaskBack(taskList_t *p_list, task_t *p_task);
int pushTaskFront(taskList_t *p_list, task_t *p_task);
task_t* popTaskBack(taskList_t *p_list);
task_t* popTaskFront(taskList_t *p_list);

task_t* getLast(taskList_t *p_list);
task_t* getFirst(taskList_t *p_list);

int insertTask(taskList_t *p_list, task_t *p_task, const int p_index);
task_t* removeTask(taskList_t *p_list, const int p_index);
int containsTask(taskList_t *p_list, task_t *p_task);
int isEmpty(taskList_t *p_list);

/* Sorts the tasks in the given tasklist with bubble sort.
   Ascending means highest priority (lowest value) has highest index.
   Descending means highest priority (lowest value) has lowest index. */
void sortPriorityAsc(taskList_t *p_list);
void sortPriorityDesc(taskList_t *p_list);
int insertTaskPriorityAsc(taskList_t *p_list, task_t *p_task);
int insertTaskPriorityDesc(taskList_t *p_list, task_t *p_task);