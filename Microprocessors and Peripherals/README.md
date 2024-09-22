# Microprocessors and Peripherals
This repository contains the source code for three assignments completed as part of the *"Microprocessors and Peripherals"* course.

## Assignments

### 1. Palindrome Checker in ARM Assembly
Implement a palindrome-checking routine in ARM assembly language.
- Develop a main routine in C to initialize a test string
- Create an ARM assembly routine to:
    - Check if the given string is a palindrome
    - Store the result (1 for palindrome, 0 for non-palindrome) in memory
    - Return the result to the main function

### 2. Reaction Time Measurement
Create an embedded device to measure human reaction time using the NUCLEO M4 board with interrupts caused by button presses.
- Implement a C program to:
    - Measure the time between LED activation and button press
    - Calculate the average reaction time over five trials
    - Store the result in memory
    - Provide an option to measure reaction time to LED deactivation

### 3. Smart Thermostat
Develop a smart thermostat system using the NUCLEO F401 board and various sensors.

Key features:
- Temperature measurement every 5 seconds
- Calculation and display of 2-minute average temperature
- LED indicators for high and low temperature thresholds
- Relay control for activating a cooling device (e.g., fan)
- Proximity sensor to display current and average temperature on demand