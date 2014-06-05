/**
 * @file shell.h
 *
 * This file provides the shellTask() function which can be used for a very
 * basic shell.
 */

#pragma once

#include "task.h"

#define PRINT_HEADER "jefax> "
#define HEADER_LIMIT 0x38 // ASCII: Column '8'

#define SHELL_TASK {shellTask, 3, READY, 0, {0}}

/**
 * This task can be scheduled.
 */
int shellTask();