#ifndef __MODEL_HPP__
#define __MODEL_HPP__

#include "constants.hpp"

void float_seq_lenet5(
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


void float_conv1(float w[L1_NUMKERNEL][L1_CHL][L1_KERNELSIZE*L1_KERNELSIZE],
		float x[L1_CHL][L1_INSIZE*L1_INSIZE],
		float a[L1_NUMKERNEL][L1_OUTSIZE*L1_OUTSIZE]);
void float_conv2(float w[L2_NUMKERNEL][L2_CHL][L2_KERNELSIZE*L2_KERNELSIZE],
		float x[L2_CHL][L2_INSIZE*L2_INSIZE],
		float a[L2_NUMKERNEL][L2_OUTSIZE*L2_OUTSIZE]);

void float_matmul3(float w[L3_ROW][L3_COL], float x[L3_ROW], float y[L3_COL]);
void float_matmul4(float w[L4_ROW][L4_COL], float x[L4_ROW], float y[L4_COL]);
void float_matmul5(float w[L5_ROW][L5_COL], float x[L5_ROW], float y[L5_COL]);
#endif
