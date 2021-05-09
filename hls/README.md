# HLS section

## Requirements

```txt
Vivado HLS == 2019.2
Vivado SDSoC == 2018.3
```

## File format

In HLS, put files in this format:

```txt
Source
	|___hw.cpp
Test bench
	|___hw.hpp
	|___main_proposed.cpp (or main.cpp)
	|___sw.hpp
```

`main.cpp` is a test bench for the floating-point model and `main_proposed.cpp` is the test bench for our purposed **FTTTF** model.

To evaluate the latency on ARM Cortex A53, please go to `test-timing` directory and put `model.hpp`, `model.cpp`, and one of `main_l1.cpp`, `main_l2.cpp`, `main_l3.cpp`, `main_l4.cpp`, or `main_l5.cpp` into Vivado SDSoC. Then, please use Vivado SDSoC to generate Linux image and elf files and put to Xilinx ZCU-102 to measure the latency.
