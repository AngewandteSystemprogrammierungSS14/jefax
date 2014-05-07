/**
 * @file dispatcher.h
 */

#pragma once

#include <avr/io.h>
#include <util/delay.h>
#include "task.h"
#include "jefax_xmega128.h"

/**
 * Starts the dispatchers idle task and sets the timer.
 */
void initDispatcher(task_t *p_defaultTask);
void setInterruptTime(unsigned int p_msec);

void dispatch(task_t *p_task);
