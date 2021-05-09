#include "constants.hpp"
#include "math.h"

template <typename T, int D1, int D2>
void max_pool2x2(T x[D1][D2], T out[D1][D2/(2*2)])
{
	T tmp1x1 = 0, tmp1x2 = 0, tmp2x1 = 0, tmp2x2 = 0;
	const int STRIDE = 2;
	const int OUT_SIZE = floor(sqrt((float)D2));

	for(int i=0; i<D1; i++)
	{
		int count = 0;
		for(int j=0; j<OUT_SIZE; j+=STRIDE)
		{
			for(int k=0; k<OUT_SIZE; k+=STRIDE)
			{
				tmp1x1 = x[i][j*OUT_SIZE + k];
				tmp1x2 = x[i][j*OUT_SIZE + k + 1];
				tmp2x1 = x[i][j*OUT_SIZE + k + OUT_SIZE];
				tmp2x2 = x[i][j*OUT_SIZE + k + 1 + OUT_SIZE];
				out[i][count] = fmax(fmax(fmax(tmp1x1, tmp1x2), tmp2x1), tmp2x2);

//					log(i << "|" << j*OUT_SIZE + k << "|" << tmp1x1);
//					log(i << "|" << j*OUT_SIZE + k + 1 << "|" << tmp1x2);
//					log(i << "|" << j*OUT_SIZE + k + OUT_SIZE << "|" << tmp2x1);
//					log(i << "|" << j*OUT_SIZE + k + 1 + OUT_SIZE << "|" << tmp2x2);
//					log(i << "|" << count << "|" << out[i][count]);
//					log("#############################");
				count++;
			}
		}
	}
}

template <typename T, int D1, int D2>
void batch_nor(T x[D1][D2],  T mean[D1], T std[D1], T gamma[D1], T beta[D1], T out[D1][D2])
{
	for(int i=0; i<D1; i++)
		for(int j=0; j<D2; j++)
			out[i][j] = (gamma[i]*(x[i][j]-mean[i])/(std[i])) + beta[i];
}

template <typename T, int D1>
void batch_nor(T x[D1],  T mean[D1], T std[D1], T gamma[D1], T beta[D1], T out[D1])
{
	for(int i=0; i<D1; i++)
		out[i] = (gamma[i]*(x[i]-mean[i])/(std[i])) + beta[i];
}

template <typename T, int D1, int D2>
void relu(T x[D1][D2], T out[D1][D2])
{
	const static T tmp_zero = 0;
	for(int i=0; i<D1; i++)
		for(int j=0; j<D2; j++)
			out[i][j] = fmax(x[i][j], tmp_zero);
}

template <typename T, int D1>
void relu(T x[D1], T out[D1])
{
	const static T tmp_zero = 0;
	for(int i=0; i<D1; i++)
		out[i] = fmax(x[i], tmp_zero);
}

template <typename T, int D1>
void softmax(T x[D1], T out[D1])
{
	static T tmp = 0;
	T tmp_out[D1];
	for(int i=0; i<D1; i++)
	{
		//log("x[i]: " << x[i]);
		// If overflow, check for dividing the pixel values with 255 or not.
		tmp_out[i] = exp(x[i]);
		tmp += tmp_out[i];
	}
	for(int i=0; i<D1; i++)
	{
		out[i] = tmp_out[i]/tmp;
		//log("soft[" << i << "]: " << out[i]);
	}
}

template <typename T, int D1, int D2>
void flatten(T x[D1][D2], T out[D1*D2])
{
	int count = 0;
	for(int j=0; j<D1; j++)
	{
		for(int i=0; i<D2; i++)
		{
			out[count] = x[j][i];
			count++;
		}
	}
}

template <typename T1, typename T2, int D1, int D2>
void matmul(T1 w[D1][D2], T2 x[D1], T2 out[D2])
{
	for(int i=0; i<D2; i++)
	{
		T2 tmp_y = 0;
		for(int j=0; j<D1; j++)
			tmp_y += x[j]*w[j][i];
		out[i] = tmp_y;
	}
}

template <typename T1, typename T2, int D_NK, int D_C, int D_K, int D_I, int D_O>
void kernel_sum(T1 w[D_NK][D_C][D_K*D_K], T2 x[D_C][D_I*D_I], int out_x_idx, int out_y_idx, int ker_idx, T2 &out)
{
	T2 sum = 0;
	for(int k=0; k<D_C; k++)
		for(int i=0; i<D_K; i++)
			for(int j=0; j<D_K; j++)
				sum += x[k][(i+out_y_idx)*D_I + j + out_x_idx]*w[ker_idx][k][i*D_K+j];
	out = sum;
}

template <typename T1, typename T2, int D_NK, int D_C, int D_K, int D_I, int D_O>
void conv(T1 w[D_NK][D_C][D_K*D_K], T2 x[D_C][D_I*D_I], T2 out[D_NK][D_O*D_O])
{
	for(int i=0; i<D_NK; i++)
		for(int k=0; k<D_O; k++)
			for(int l=0; l<D_O; l++)
				kernel_sum<T1, T2, D_NK, D_C, D_K, D_I, D_O>(w, x, l, k, i, out[i][k*D_O + l]);
}

void float_conv1(float w[L1_NUMKERNEL][L1_CHL][L1_KERNELSIZE*L1_KERNELSIZE],
		float x[L1_CHL][L1_INSIZE*L1_INSIZE],
		float a[L1_NUMKERNEL][L1_OUTSIZE*L1_OUTSIZE])
{
	conv<float, float, L1_NUMKERNEL, L1_CHL, L1_KERNELSIZE, L1_INSIZE, L1_OUTSIZE>(w, x, a);
}

void float_conv2(float w[L2_NUMKERNEL][L2_CHL][L2_KERNELSIZE*L2_KERNELSIZE],
		float x[L2_CHL][L2_INSIZE*L2_INSIZE],
		float a[L2_NUMKERNEL][L2_OUTSIZE*L2_OUTSIZE])
{
	conv<float, float, L2_NUMKERNEL, L2_CHL, L2_KERNELSIZE, L2_INSIZE, L2_OUTSIZE>(w, x, a);
}

void float_matmul3(float w[L3_ROW][L3_COL], float x[L3_ROW], float y[L3_COL])
{
	matmul<float, float, L3_ROW, L3_COL>(w, x, y);
}
void float_matmul4(float w[L4_ROW][L4_COL], float x[L4_ROW], float y[L4_COL])
{
	matmul<float, float, L4_ROW, L4_COL>(w, x, y);
}
void float_matmul5(float w[L5_ROW][L5_COL], float x[L5_ROW], float y[L5_COL])
{
	matmul<float, float, L5_ROW, L5_COL>(w, x, y);
}

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
		float out[L5_COL])
{
	float o_c1[L1_NUMKERNEL][L1_OUTSIZE*L1_OUTSIZE];
	float o_b1[L1_NUMKERNEL][L1_OUTSIZE*L1_OUTSIZE];
	float o_r1[L1_NUMKERNEL][L1_OUTSIZE*L1_OUTSIZE];
	float o_m1[L1_NUMKERNEL][L1_OUTSIZE*L1_OUTSIZE/4];

	float o_c2[L2_NUMKERNEL][L2_OUTSIZE*L2_OUTSIZE];
	float o_b2[L2_NUMKERNEL][L2_OUTSIZE*L2_OUTSIZE];
	float o_r2[L2_NUMKERNEL][L2_OUTSIZE*L2_OUTSIZE];
	float o_m2[L2_NUMKERNEL][L2_OUTSIZE*L2_OUTSIZE/4];
	float o_f2[L2_NUMKERNEL*L2_OUTSIZE*L2_OUTSIZE/4];

	float o_c3[L3_COL];
	float o_b3[L3_COL];
	float o_r3[L3_COL];

	float o_c4[L4_COL];
	float o_b4[L4_COL];
	float o_r4[L4_COL];

	float o_c5[L5_COL];
	//float o_r5[L5_COL];

	float_conv1(w1, x, o_c1);
	batch_nor<float, L1_NUMKERNEL, L1_OUTSIZE*L1_OUTSIZE>(
		o_c1, mean1, std1, gamma1, beta1, o_b1);
	relu<float, L1_NUMKERNEL, L1_OUTSIZE*L1_OUTSIZE>(o_b1, o_r1);
	max_pool2x2<float, L1_NUMKERNEL, L1_OUTSIZE*L1_OUTSIZE>(o_r1, o_m1);

	float_conv2(w2, o_m1, o_c2);
	batch_nor<float, L2_NUMKERNEL, L2_OUTSIZE*L2_OUTSIZE>(
			o_c2, mean2, std2, gamma2, beta2, o_b2);
	relu<float, L2_NUMKERNEL, L2_OUTSIZE*L2_OUTSIZE>(o_b2, o_r2);
	max_pool2x2<float, L2_NUMKERNEL, L2_OUTSIZE*L2_OUTSIZE>(o_r2, o_m2);
	flatten<float, L2_NUMKERNEL, L2_OUTSIZE*L2_OUTSIZE/4>(o_m2, o_f2);

	float_matmul3(w3, o_f2, o_c3);
	batch_nor<float, L3_COL>(o_c3, mean3, std3, gamma3, beta3, o_b3);
	relu<float, L3_COL>(o_b3, o_r3);

	float_matmul4(w4, o_r3, o_c4);
	batch_nor<float, L4_COL>(o_c4, mean4, std4, gamma4, beta4, o_b4);
	relu<float, L4_COL>(o_b4, o_r4);

	float_matmul5(w5, o_r4, o_c5);
	softmax<float, L5_COL>(o_c5, out);
}


