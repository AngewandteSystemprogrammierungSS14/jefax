#include <stddef.h>
#include "tasklist.h"

static void sortPriority(taskList_t *p_list, int p_asc);
static int getTaskPriorityInsertIndex(taskList_t *p_list, task_t *p_task, int p_asc);

int initTaskList(taskList_t *p_list)
{
	p_list->count = 0;
	p_list->size = TASK_LIST_SIZE;
	
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

task_t* getLast(taskList_t *p_list)
{
	if(p_list->count <= 0)
		return NULL;
		
	return p_list->elements[p_list->count - 1];
}

task_t* getFirst(taskList_t *p_list)
{
	if(p_list->count <= 0)
		return NULL;
	
	return p_list->elements[0];
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
	for(i = 0; i < p_list->count; ++i) {
		if(p_list->elements[i] == p_task)
			break;
	}
	
	return i < p_list->count;
}

int isEmpty(taskList_t *p_list)
{
	return p_list->count == 0;
}

void sortPriorityAsc(taskList_t *p_list)
{
	sortPriority(p_list, 1);
}

void sortPriorityDesc(taskList_t *p_list)
{
	sortPriority(p_list, 0);
}

#define PRIORITY_ASC(t1, t2) (CMP_PRIORITY(t1, t2) > 0)
#define PRIORITY_DESC(t1, t2) (CMP_PRIORITY(t1, t2) < 0)
#define SWITCH_POSITION(t1, t2, asc) ((PRIORITY_ASC(t1, t2) && asc) || \
									  (PRIORITY_DESC(t1, t2) && !asc))

static void sortPriority(taskList_t *p_list, int p_asc)
{
	int i, j, changed;
	task_t *tmp;
	
	for(i = p_list->count - 1; i >= 0; --i) {
		changed = 0;
		for(j = 0; j < i; ++j) {
			if (SWITCH_POSITION(p_list->elements[j], p_list->elements[j+1], p_asc)) {
				tmp = p_list->elements[j];
				p_list->elements[j] = p_list->elements[j+1];
				p_list->elements[j+1] = tmp;
				changed = 1;
			}

		}
		if(!changed)
			break;
	}
}

int insertTaskPriorityAsc(taskList_t *p_list, task_t *p_task)
{
	int index = getTaskPriorityInsertIndex(p_list, p_task, 1);
	return insertTask(p_list, p_task, index);
}

int insertTaskPriorityDesc(taskList_t *p_list, task_t *p_task)
{
	int index = getTaskPriorityInsertIndex(p_list, p_task, 0);
	return insertTask(p_list, p_task, index);
}

static int getTaskPriorityInsertIndex(taskList_t *p_list, task_t *p_task, int p_asc)
{
	int result;
	
	if(p_list->count == 0)
		return 0;
	
	result = 0;
	// find correct index for corresponding priority
	while(result < p_list->count && SWITCH_POSITION(p_task, p_list->elements[result], p_asc))
		++result;
	
	return result;
}
