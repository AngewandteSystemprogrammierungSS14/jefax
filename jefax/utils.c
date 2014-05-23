/*
 * utils.c
 *
 * Created: 23.05.2014 10:48:04
 *  Author: Fabian
 */ 

#include "utils.h"
#include <avr/interrupt.h>

void enableInterrupts()
{
	PMIC.CTRL |= PMIC_LOLVLEN_bm;
	sei();
}