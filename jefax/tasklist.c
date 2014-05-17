/*
 * tasklist.c
 *
 * Created: 17.05.2014 10:37:50
 *  Author: Fabian
 */ 

#include "tasklist.h"
#include "stddef.h"

static void sortPriorityR(taskList_t *p_list, const int p_front, const int p_back, taskList_t *p_helperList);
static void mergePriority(taskList_t *p_a, const int a_front, const int a_back,
						  taskList_t *p_b, const int b_front, const int b_back,
						  taskList_t *p_c, const int c_front);

int initTaskList(taskList_t *p_list)
{
	p_list->count = 0;
	p_list->size = DEF_TASK_LIST_SIZE;
	
	return 0;
}

int pushTaskBack(taskList_t *p_list, task_t *p_task)
{
	if(p_list->count >= p_list->size)
		return -1;
	
	p_list->elements[p_list->count] = p_task;
	p_list->count += 1;
	
	return p_list->count - 1;
}
int pushTaskFront(taskList_t *p_list, task_t *p_task)
{
	return insertTask(p_list, p_task, 0);
}

task_t* popTaskBack(taskList_t *p_list)
{
	if(p_list->count <= 0)
		return NULL;
	
	task_t* result = p_list->elements[p_list->count - 1];
	p_list->count -= 1;
	
	return result;
}

task_t* popTaskFront(taskList_t *p_list)
{
	return removeTask(p_list, 0);
}

int insertTask(taskList_t *p_list, task_t *p_task, const int p_index)
{
	if(p_list->count >= p_list->size)
		return -1;
	if(p_index > p_list->count)
		return -2;
	if(p_index < 0)
		return -3;
	
	int i;
	for(i = p_list->count; i > p_index; --i)
		p_list->elements[i] = p_list->elements[i - 1];
		
	p_list->elements[p_index] = p_task;
	p_list->count += 1;
	
	return p_index;
}

task_t* removeTask(taskList_t *p_list, const int p_index)
{
	if(p_list->count <= 0 || p_index >= p_list->count || p_index < 0)
		return NULL;
	
	int i;
	task_t* result = p_list->elements[p_index];
	for(i = p_index; i < p_list->count - 1; ++i)
		p_list->elements[i] = p_list->elements[i + 1];
	
	p_list->count -= 1;
	return result;
}

int containsTask(taskList_t *p_list, task_t *p_task)
{
	int i;
	for(i = 0; i < p_list->count; ++i)
	{
		if(p_list->elements[i] == p_task)
			break;
	}
	
	return i < p_list->count;
}

void sortPriority(taskList_t *p_list)
{
	taskList_t helperList;
	sortPriorityR(p_list, 0, p_list->count - 1, &helperList);
}

static void sortPriorityR(taskList_t *p_list, const int p_front, const int p_back, taskList_t *p_helperList)
{
	if(p_back > p_front)
	{
		int i;
		int mid = (p_back + p_front) / 2;
		
		sortPriorityR(p_list, p_front, mid, p_helperList);
		sortPriorityR(p_list, mid + 1, p_back, p_helperList);
		
		mergePriority(p_list, p_front, mid, p_list, mid + 1, p_back, p_helperList, p_front);
		
		for(i = 0; i < p_list->count; ++i)
			p_list->elements[i] = p_helperList->elements[i];
	}
}

static void mergePriority(taskList_t *p_a, const int a_front, const int a_back,
						  taskList_t *p_b, const int b_front, const int b_back,
						  taskList_t *p_c, const int c_front)
{
	int i = a_front;
	int j = b_front;
	int k = c_front;
	
	while (i <= a_back && j <= b_back)
	{
		if (p_a->elements[i]->priority >= p_b->elements[j]->priority)
		{
			p_c->elements[k] = p_a->elements[i];
			++k;
			++i;
		}
		else
		{
			p_c->elements[k] = p_b->elements[j];
			++k;
			++j;
		}
	}
			
	if (j == b_back + 1)
	{
		while (i <= a_back)
		{
			p_c->elements[k] = p_a->elements[i];
			++k;
			++i;
		}
	}
	else 
	{
		while (j <= b_back)
		{
			p_c->elements[k] = p_b->elements[j];
			++j;
			++k;
		}
	}
}