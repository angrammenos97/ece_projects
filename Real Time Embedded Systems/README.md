# Real Time Embedded Systems
This repository contains the source code for *"Real Time Embedded Systems"* course assignments.

## 1. Producer-Consumer Problem

### Goal
The main objective of this assignment was to modify the classic producer-consumer example to support multiple producers and consumers operating on a shared queue, exploring the concepts of concurrent programming, thread synchronization, and real-time system design.

### Key Features
- Implementation of a multi-producer, multi-consumer system
- Use of POSIX Threads for thread creation and management
- Mutex locks and conditional variables for synchronization
- Performance analysis of queue wait times and execution times


## 2. Advanced Real-Time Scheduling

### Goal
This assignment focused on developing a more complex real-time scheduling system with the following key objectives:
- Implement a flexible job scheduling system with customizable periodic tasks
- Create a robust logging system for performance analysis
- Explore CPU utilization and timing constraints in a real-time environment

### Key Features
- Customizable periodic job execution using timer-based producers
- Flexible queue implementation supporting various data types
- Advanced logging system for capturing detailed timing information
- JSON-based configuration for easy experiment setup
- Performance analysis, including CPU usage, queue lag, and execution times

## Implementation Details
- Programming Language: C
- Libraries: POSIX Threads, custom queue implementation
- Target Platform: Raspberry Pi 4 (for final testing)
- Build System: Make

## Notes
- These assignments explore fundamental concepts in real-time systems, including task scheduling, resource sharing, and performance analysis.
- The code demonstrates practical applications of concurrent programming techniques in embedded systems.