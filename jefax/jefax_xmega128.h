/**
 * @file jefax_xmega128.h
 *
 * This file contains helper functions to use with the avr xmeag128 xplained board.
 */

#pragma once

#include <stdint.h>

/**
 * Initializes the LEDs on the PORTE.
 */
void initLED();

/**
 * Sets the PORTE status to the given value.
 *
 * @param status The new status of the leds
 */
void setLED(uint8_t status);