#ifndef __SW_HPP__
#define __SW_HPP__

#include "hls_math.h"
#include "hw.hpp"

namespace sw
{
	template <typename T, int D1, int D2>
	void max_pool2x2(T x[D1][D2], T out[D1][D2/(2*2)])
	{
	#pragma HLS INLINE OFF
		T tmp1x1 = 0, tmp1x2 = 0, tmp2x1 = 0, tmp2x2 = 0;
		const int STRIDE = 2;
		const int OUT_SIZE = hls::floor(hls::sqrt((T)D2));

		max1:for(int i=0; i<D1; i++)
		{
			int count = 0;
			max2:for(int j=0; j<OUT_SIZE; j+=STRIDE)
			{
				max3:for(int k=0; k<OUT_SIZE; k+=STRIDE)
				{
					tmp1x1 = x[i][j*OUT_SIZE + k];
					tmp1x2 = x[i][j*OUT_SIZE + k + 1];
					tmp2x1 = x[i][j*OUT_SIZE + k + OUT_SIZE];
					tmp2x2 = x[i][j*OUT_SIZE + k + 1 + OUT_SIZE];
					out[i][count] = hls::fmax(hls::fmax(hls::fmax(tmp1x1, tmp1x2), tmp2x1), tmp2x2);
					count++;
				}
			}
		}
	}

	template <typename T, int D1, int D2>
	void batch_nor(T x[D1][D2],  T mean[D1], T std[D1], T gamma[D1], T beta[D1], T out[D1][D2])
	{
	#pragma HLS INLINE OFF
		batch1:for(int i=0; i<D1; i++)
			batch2:for(int j=0; j<D2; j++)
				out[i][j] = (gamma[i]*(x[i][j]-mean[i])/(std[i])) + beta[i];
	}

	template <typename T, int D1>
	void batch_nor(T x[D1],  T mean[D1], T std[D1], T gamma[D1], T beta[D1], T out[D1])
	{
	#pragma HLS INLINE OFF
		batch1:for(int i=0; i<D1; i++)
			out[i] = (gamma[i]*(x[i]-mean[i])/(std[i])) + beta[i];
	}

	template <typename T, int D1, int D2>
	void relu(T x[D1][D2], T out[D1][D2])
	{
	#pragma HLS INLINE OFF
		const static T tmp_zero = 0;
		relu1:for(int i=0; i<D1; i++)
			relu2:for(int j=0; j<D2; j++)
				out[i][j] = hls::fmax(x[i][j], tmp_zero);
	}

	template <typename T, int D1>
	void relu(T x[D1], T out[D1])
	{
	#pragma HLS INLINE OFF
		const static T tmp_zero = 0;
		relu1:for(int i=0; i<D1; i++)
			out[i] = hls::fmax(x[i], tmp_zero);
	}

	template <typename T, int D1>
	void softmax(T x[D1], T out[D1])
	{
	#pragma HLS INLINE OFF
		static T tmp = 0;
		T tmp_out[D1];
		for(int i=0; i<D1; i++)
		{
			tmp_out[i] = hls::exp(x[i]);
			tmp += tmp_out[i];
		}
		for(int i=0; i<D1; i++)
		{
			out[i] = tmp_out[i]/tmp;
		}
	}

	template <typename T, int D1, int D2>
	void flatten(T x[D1][D2], T out[D1*D2])
	{
	#pragma HLS INLINE OFF
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
	#pragma HLS INLINE OFF
		L1:for(int i=0; i<D2; i++)
		{
			T2 tmp_y = 0;
			L2:for(int j=0; j<D1; j++)
				tmp_y += x[j]*w[j][i];
			out[i] = tmp_y;
		}
	}

	template <typename T1, typename T2, int D_NK, int D_C, int D_K, int D_I, int D_O>
	void kernel_sum(T1 w[D_NK][D_C][D_K*D_K], T2 x[D_C][D_I*D_I], int out_x_idx, int out_y_idx, int ker_idx, T2 &out)
	{
	#pragma HLS INLINE OFF
		T2 sum = 0;
		L0:for(int k=0; k<D_C; k++)
			L1:for(int i=0; i<D_K; i++)
				L2:for(int j=0; j<D_K; j++)
					sum += x[k][(i+out_y_idx)*D_I + j + out_x_idx]*w[ker_idx][k][i*D_K+j];
		out = sum;
	}

	template <typename T1, typename T2, int D_NK, int D_C, int D_K, int D_I, int D_O>
	void conv(T1 w[D_NK][D_C][D_K*D_K], T2 x[D_C][D_I*D_I], T2 out[D_NK][D_O*D_O])
	{
	#pragma HLS INLINE OFF
		L1:for(int i=0; i<D_NK; i++)
			L2:for(int k=0; k<D_O; k++)
				L3:for(int l=0; l<D_O; l++)
					kernel_sum<T1, T2, D_NK, D_C, D_K, D_I, D_O>(w, x, l, k, i, out[i][k*D_O + l]);
	}
}

#endif
