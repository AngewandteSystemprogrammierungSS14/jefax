/* The system component of jefax contains functions and members
 * regarding the whole system (all tasks, ISR). */

#pragma once

#include <stdint.h>

extern uint8_t *main_stackpointer;
#define ENTER_SYSTEM_STACK() SP = (uint16_t) main_stackpointer

#define RET() asm volatile("ret");