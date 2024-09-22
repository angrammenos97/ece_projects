#include "stopwatch.h"
#include "platform.h"

void start_watch(Stopwatch *watch)
{
	watch->start = DWT->CYCCNT;
}

void stop_watch(Stopwatch *watch)
{
	watch->stop = DWT->CYCCNT - 1;
}

float get_watch_time(Stopwatch *watch)
{
	if (watch->stop < watch->start)
		return ( ((float)((UINT32_MAX - watch->start) + watch->stop)) * (float)1e6 ) / SystemCoreClock;	//Cycle Counter Register overflowed
	return ( ((float)(watch->stop - watch->start)) * (float)1e6 ) / SystemCoreClock;	// return time in us
}
