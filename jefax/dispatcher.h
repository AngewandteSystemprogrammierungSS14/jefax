/**
 * @file dispatcher.h
 */

#pragma once

/**
 * Starts the dispatchers idle task and sets the timer.
 */
void initDispatcher();
void setInterruptTime(unsigned int p_msec);
