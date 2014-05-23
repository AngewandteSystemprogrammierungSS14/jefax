/*
 * tasklist.h
 *
 * Created: 17.05.2014 10:37:36
 *  Author: Fabian
 */ 

#pragma once

#include "task.h"

#define DEF_TASK_LIST_SIZE 20

typedef struct
{
	task_t *elements[DEF_TASK_LIST_SIZE];
	int size;
	int count;
} taskList_t;

int initTaskList(taskList_t *p_list);

int pushTaskBack(taskList_t *p_list, task_t *p_task);
int pushTaskFront(taskList_t *p_list, task_t *p_task);
task_t* popTaskBack(taskList_t *p_list);
task_t* popTaskFront(taskList_t *p_list);

int insertTask(taskList_t *p_list, task_t *p_task, const int p_index);
task_t* removeTask(taskList_t *p_list, const int p_index);
int containsTask(taskList_t *p_list, task_t *p_task);

/*Sorts the tasks in the given tasklist with bubble sort.
  Lowest priority (highest value) comes first. */
void sortPriority(taskList_t *p_list);