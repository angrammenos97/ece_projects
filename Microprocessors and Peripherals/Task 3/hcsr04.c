#include "hcsr04.h"
#include "platform.h"
#include "gpio.h"
#include "delay.h"
#include "stopwatch.h"

// Anything over 400 cm is "out of range"
#define MAX_DIST 400.0f

Pin trigger_pin, echo_pin;
Stopwatch watch;

void hcsr04_init(Pin tr, Pin ec)
{
	trigger_pin = tr;
	gpio_set_mode(trigger_pin, Output);
	gpio_set(trigger_pin, 0);

	echo_pin = ec;
	gpio_set_mode(echo_pin, PullDown);
}

float hcsr04_get_distance(void)
{
	// Hold the trigger pin high for at least 10 us
	gpio_set(trigger_pin, 1);
	delay_us(10);
    gpio_set(trigger_pin, 0);
	// Wait for pulse on echo pin
	while(!gpio_get(echo_pin))
		__NOP();
	// Measure how long the echo pin was held high (pulse width)
	// Note: the Cycles Counter Register will overflow after ~4.5 min
	start_watch(&watch);
	while(gpio_get(echo_pin))
		__NOP();
	stop_watch(&watch);
	// Calculate distance in centimeters. The constants
    // are found in the datasheet, and calculated from the assumed speed
    // of sound in air at sea level (~340 m/s).
	float distance = get_watch_time(&watch) / 58.31f;
//    if (distance > MAX_DIST)
//		distance = -1.0f;
	return distance;
}
