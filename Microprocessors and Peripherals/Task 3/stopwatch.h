#ifndef STOPWATCH_H
#define STOPWATCH_H

/*! Structure to hold start and stop time. */
typedef struct {
	unsigned int start;
	unsigned int stop;
} Stopwatch;

/*! \brief Starts watch to count time.
 *  \param watch.
 */
void start_watch(Stopwatch *watch);

/*! \brief Stops watch from counting time.
 *  \param watch.
 */
void stop_watch(Stopwatch *watch);

/*! \brief Return the watch's time interval.
 *  \param watch.
 *	\return time interval in microseconds.
*/
float get_watch_time(Stopwatch *watch);

#endif
