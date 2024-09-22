# Software Hardware Co-design
This repository contains the source files for three laboratory projects completed for the *"Software Hardware Co-design"* course.

## Tasks

### 1. Hardware Accelerator Design with Vivado HLS
Design and optimize a hardware accelerator for matrix multiplication using Vivado High-Level Synthesis (HLS).

Key tasks:
- Implement matrix multiplication in C/C++
- Create a testbench to verify functionality
- Synthesize the design and analyze performance metrics
- Apply HLS directives to optimize the accelerator
- Measure speedup compared to the initial design

### 2. Accelerator Implementation with Vitis
Port the matrix multiplication accelerator to the Vitis unified software platform and run it on an Alveo U200 FPGA card.

Key tasks:
- Convert host code to use OpenCL API
- Adapt kernel code for Vitis
- Perform software and hardware emulation
- Analyze performance in emulated environments

### 3. Optimized Data Transfer Methods
Optimize data transfer between the x86 host and FPGA accelerator using advanced Vitis techniques.

Key tasks:
- Implement wide data transfers using 512-bit vectors
- Utilize multiple DDR memory banks on the Alveo card
- Configure kernel interfaces for parallel data access
- Compare performance improvements from optimizations