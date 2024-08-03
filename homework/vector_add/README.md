# Programming Assignment 1: Vector Addition

## Objective

The purpose of this lab is to introduce the student to the OpenCL API by implementing vector addition. The student will implement vector addition by writing the OpenCL kernel code as well as the associated host code.

## Instructions

Edit the code in the code tab to perform the following:

- Create memory buffers on device
- Copy host memory to device
- Initialize global size (total number of work-itmes in the grid) and local size (number of work-items per work-group)
- Invoke OpenCL kernel
- Copy results from device to host
- Free device memory
- Write the OpenCL kernel

Instructions about where to place each part of the code is demarcated by the `//@@` comment lines.

## How to Compile

The `main.c` and `kernel.cl` files contain the code for the programming assignment. There is a Makefile included which compiles it. It can be run by typing `make` from the PA2 folder. It generates a `solution` output file. During development, make sure to run the `make clean` command before running `make`. 

## How to test

Use the `make run` command to test your program. There are a total of 10 tests on which your program will be evaluated.

## Dataset Generation (optional)

The dataset required to test the program is already generated. If you are interested in how the dataset is generated please refer to the `dataset_generator.py` file in the `Dataset` directory. You may run this script using `python dataset_generator.py`.

## Different Data Types

Adapt your code to test both 32-bit integers and doubles.

## Submission

Submit the `main.c` and `kernel.cl` files on Gradescope with the original `float` datatypes.  This will be graded by Gradescope and will expect at least 32-bit floating point precision.  You may lose points if you do not revert to the `float` datatype. Preserve the file name while uploading to Gradescope.