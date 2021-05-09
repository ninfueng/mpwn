#ifndef __HW_HPP__
#define __HW_HPP__

#include <iostream>
#include <stdlib.h>
#include <assert.h>
#include "hls_half.h"
#include "ap_int.h"
#include "ap_fixed.h"
#include "hls_math.h"

typedef ap_int<1> int1;
typedef ap_int<2> int2;
typedef ap_int<15> int15;
typedef ap_uint<15> uint15;
typedef ap_int<16> int16;
typedef ap_uint<1> uint1;
typedef float data_t;
typedef unsigned int uint;
typedef int coor_t;
typedef half batch_t;
typedef int sp_t;

typedef half w1_t;
typedef half a1_t;
typedef int1 w2_t;
typedef half a2_t;
typedef ap_int<1> w3_t;
typedef half a3_t;
typedef int1 w4_t;
typedef half a4_t;
typedef half w5_t;
typedef half a5_t;

const int L1_KERNELSIZE = 5;
const int L1_CHL = 1;
const int L1_NUMKERNEL = 6;
const int L1_INSIZE = 28;
const int L1_OUTSIZE = 24;

const int L2_KERNELSIZE = 5;
const int L2_CHL = 6;
const int L2_NUMKERNEL = 16;
const int L2_INSIZE = 12;
const int L2_OUTSIZE = 8;

const int L3_ROW = 4*4*16;
const int L3_COL = 120;
const int L3_NONZEROS = 15440;
const int L3_MAX_NON_ZEROS_COL = 155;
const int L3_STAGE = 16;

const int L4_ROW = 120;
const int L4_COL = 84;

const int L5_ROW = 84;
const int L5_COL = 10;

const int PARTITION_CONV_FACTOR = 8;
const int PARTITION_DENSE_FACTOR = 16;

void xor_signed_bit(int2 w, half x, half &out);
void ternary_bitwise(int2 w, half x, half &out);
void kernel_sum(
		half w[L1_NUMKERNEL][L1_CHL][L1_KERNELSIZE*L1_KERNELSIZE],
		half x[L1_CHL][L1_INSIZE*L1_INSIZE],
		int out_x_idx, int out_y_idx, int ker_idx, half &a);
void conv(
		w1_t w[L1_NUMKERNEL][L1_CHL][L1_KERNELSIZE*L1_KERNELSIZE],
		a1_t x[L1_CHL][L1_INSIZE*L1_INSIZE],
		a1_t a[L1_NUMKERNEL][L1_OUTSIZE*L1_OUTSIZE]);
void conv_bin(
		w2_t w[L2_NUMKERNEL][L2_CHL][L2_KERNELSIZE*L2_KERNELSIZE],
		a2_t x[L2_CHL][L2_INSIZE*L2_INSIZE],
		a2_t a[L2_NUMKERNEL][L2_OUTSIZE*L2_OUTSIZE]);
void matmul(w5_t w[L5_ROW][L5_COL], a5_t x[L5_ROW], a5_t y[L5_COL]);
void bin_half_matmul(w4_t w[L4_ROW][L4_COL], a4_t x[L4_ROW], a4_t y[L4_COL]);

void dir_float_lenet5(
		float w1[L1_NUMKERNEL][L1_CHL][L1_KERNELSIZE*L1_KERNELSIZE],
		float w2[L2_NUMKERNEL][L2_CHL][L2_KERNELSIZE*L2_KERNELSIZE],
		float w3[L3_ROW][L3_COL],
		float w4[L4_ROW][L4_COL],
		float w5[L5_ROW][L5_COL],
		float x[L1_CHL][L1_INSIZE*L1_INSIZE],
		float mean1[L1_NUMKERNEL], float std1[L1_NUMKERNEL],
		float gamma1[L1_NUMKERNEL], float beta1[L1_NUMKERNEL],
		float mean2[L2_NUMKERNEL], float std2[L2_NUMKERNEL],
		float gamma2[L2_NUMKERNEL], float beta2[L2_NUMKERNEL],
		float mean3[L3_COL], float std3[L3_COL],
		float gamma3[L3_COL], float beta3[L3_COL],
		float mean4[L4_COL], float std4[L4_COL],
		float gamma4[L4_COL], float beta4[L4_COL],
		float out[L5_COL]);

void float_lenet5(
		float w1[L1_NUMKERNEL][L1_CHL][L1_KERNELSIZE*L1_KERNELSIZE],
		float w2[L2_NUMKERNEL][L2_CHL][L2_KERNELSIZE*L2_KERNELSIZE],
		float w3[L3_ROW][L3_COL],
		float w4[L4_ROW][L4_COL],
		float w5[L5_ROW][L5_COL],
		float x[L1_CHL][L1_INSIZE*L1_INSIZE],
		float mean1[L1_NUMKERNEL], float std1[L1_NUMKERNEL],
		float gamma1[L1_NUMKERNEL], float beta1[L1_NUMKERNEL],
		float mean2[L2_NUMKERNEL], float std2[L2_NUMKERNEL],
		float gamma2[L2_NUMKERNEL], float beta2[L2_NUMKERNEL],
		float mean3[L3_COL], float std3[L3_COL],
		float gamma3[L3_COL], float beta3[L3_COL],
		float mean4[L4_COL], float std4[L4_COL],
		float gamma4[L4_COL], float beta4[L4_COL],
		float out[L5_COL]);

void dir_float_ftttf(
		half w1[L1_NUMKERNEL][L1_CHL][L1_KERNELSIZE*L1_KERNELSIZE],
		int2 w2[L2_NUMKERNEL][L2_CHL][L2_KERNELSIZE*L2_KERNELSIZE],
		int2 w3[L3_ROW][L3_COL],
		int2 w4[L4_ROW][L4_COL],
		half w5[L5_ROW][L5_COL],
		half x[L1_CHL][L1_INSIZE*L1_INSIZE],
		half mean1[L1_NUMKERNEL], half std1[L1_NUMKERNEL],
		half gamma1[L1_NUMKERNEL], half beta1[L1_NUMKERNEL],
		half mean2[L2_NUMKERNEL], half std2[L2_NUMKERNEL],
		half gamma2[L2_NUMKERNEL], half beta2[L2_NUMKERNEL],
		half mean3[L3_COL], half std3[L3_COL],
		half gamma3[L3_COL], half beta3[L3_COL],
		half mean4[L4_COL], half std4[L4_COL],
		half gamma4[L4_COL], half beta4[L4_COL],
		half out[L5_COL]);

#endif



