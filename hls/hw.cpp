#include "hw.hpp"
#include "sw.hpp"

void xor_signed_bit(int1 w, half x, half &out)
{
#pragma HLS INLINE OFF
	int16 tmp_x;
	int16 tmp_out;
	tmp_x = *reinterpret_cast<int16*>(&x);
	uint1 sign = tmp_x.sign()^w.sign();
	int15 notsign = tmp_x.range(14, 0);
	tmp_out = sign.concat(notsign);
	out = *reinterpret_cast<half*>(&tmp_out);
}

void ternary_bitwise(int2 w, half x, half &out)
{
#pragma HLS INLINE OFF
	int16 tmp_x;
	int16 tmp_o;
	tmp_x = *reinterpret_cast<int16*>(&x);
	uint1 b15 = tmp_x.sign()^w.sign();

	uint1 w0 = w.range(0, 0);
	uint1 b0 = w0 && tmp_x.range(0, 0);
	uint1 b1 = w0 && tmp_x.range(1, 1);
	uint1 b2 = w0 && tmp_x.range(2, 2);
	uint1 b3 = w0 && tmp_x.range(3, 3);
	uint1 b4 = w0 && tmp_x.range(4, 4);
	uint1 b5 = w0 && tmp_x.range(5, 5);
	uint1 b6 = w0 && tmp_x.range(6, 6);
	uint1 b7 = w0 && tmp_x.range(7, 7);
	uint1 b8 = w0 && tmp_x.range(8, 8);
	uint1 b9 = w0 && tmp_x.range(9, 9);
	uint1 b10 = w0 && tmp_x.range(10, 10);
	uint1 b11 = w0 && tmp_x.range(11, 11);
	uint1 b12 = w0 && tmp_x.range(12, 12);
	uint1 b13 = w0 && tmp_x.range(13, 13);
	uint1 b14 = w0 && tmp_x.range(14, 14);
	uint15 b_con = (b14, b13, b12, b11, b10, b9, b8, b7, b6, b5, b4, b3, b2, b1, b0);

	tmp_o = b15.concat(b_con);
	out = *reinterpret_cast<half*>(&tmp_o);
}


void kernel_sum(
		half w[L1_NUMKERNEL][L1_CHL][L1_KERNELSIZE*L1_KERNELSIZE],
		half x[L1_CHL][L1_INSIZE*L1_INSIZE],
		int out_x_idx, int out_y_idx, int ker_idx, half &a)
{
#pragma HLS INLINE
	half sum = 0;
	kernel_sum1:for(int k=0; k<L1_CHL; k++)
	{
		kernel_sum2:for(int i=0; i<L1_KERNELSIZE; i++)
		{
			kernel_sum3:for(int j=0; j<L1_KERNELSIZE; j++)
			{
				sum += x[k][(i+out_y_idx)*L1_INSIZE + j + out_x_idx]*w[ker_idx][k][i*L1_KERNELSIZE + j];
			}
		}
	}
	a = sum;
}

void dir_kernel_sum(
		half w[L1_NUMKERNEL][L1_CHL][L1_KERNELSIZE*L1_KERNELSIZE],
		half x[L1_CHL][L1_INSIZE*L1_INSIZE],
		int out_x_idx, int out_y_idx, int ker_idx, half &a)
{
#pragma HLS INLINE
	half sum = 0;
	kernel_sum1:for(int k=0; k<L1_CHL; k++)
	{
		kernel_sum2:for(int i=0; i<L1_KERNELSIZE; i++)
		{
			kernel_sum3:for(int j=0; j<L1_KERNELSIZE; j++)
			{
				sum += x[k][(i+out_y_idx)*L1_INSIZE + j + out_x_idx]*w[ker_idx][k][i*L1_KERNELSIZE + j];
			}
		}
	}
	a = sum;
}

void bin_kernel_sum(
		int1 w[L2_NUMKERNEL][L2_CHL][L2_KERNELSIZE*L2_KERNELSIZE],
		half x[L2_CHL][L2_INSIZE*L2_INSIZE],
		int out_x_idx, int out_y_idx, int ker_idx, half &a)
{
#pragma HLS INLINE
	half sum = 0;
	bin_kernel_sum1:for(int k=0; k<L2_CHL; k++)
	{
		bin_kernel_sum2:for(int i=0; i<L2_KERNELSIZE; i++)
		{
			bin_kernel_sum3:for(int j=0; j<L2_KERNELSIZE; j++)
			{
				half tmp_sum = 0;
				int1 tmp_w = w[ker_idx][k][i*L2_KERNELSIZE + j];
				half tmp_x = x[k][(i + out_y_idx)*L2_INSIZE + j + out_x_idx];
				xor_signed_bit(tmp_w, tmp_x, tmp_sum);
				sum += tmp_sum;
			}
		}
	}
	a = sum;
}

void dir_bin_kernel_sum(
		int1 w[L2_NUMKERNEL][L2_CHL][L2_KERNELSIZE*L2_KERNELSIZE],
		half x[L2_CHL][L2_INSIZE*L2_INSIZE],
		int out_x_idx, int out_y_idx, int ker_idx, half &a)
{
#pragma HLS INLINE
	half sum = 0;
	bin_kernel_sum1:for(int k=0; k<L2_CHL; k++)
	{
		bin_kernel_sum2:for(int i=0; i<L2_KERNELSIZE; i++)
		{
			bin_kernel_sum3:for(int j=0; j<L2_KERNELSIZE; j++)
			{
				half tmp_sum = 0;
				int1 tmp_w = w[ker_idx][k][i*L2_KERNELSIZE + j];
				half tmp_x = x[k][(i + out_y_idx)*L2_INSIZE + j + out_x_idx];
				xor_signed_bit(tmp_w, tmp_x, tmp_sum);
				sum += tmp_sum;
			}
		}
	}
	a = sum;
}

void dir_float_kernel_sum(
		float w[L1_NUMKERNEL][L1_CHL][L1_KERNELSIZE*L1_KERNELSIZE],
		float x[L1_CHL][L1_INSIZE*L1_INSIZE],
		int out_x_idx, int out_y_idx, int ker_idx, float &a)
{
#pragma HLS INLINE
	float sum = 0;

	kernel_sum1:for(int k=0; k<L1_CHL; k++)
	{
		kernel_sum2:for(int i=0; i<L1_KERNELSIZE; i++)
		{
			kernel_sum3:for(int j=0; j<L1_KERNELSIZE; j++)
			{
				sum += x[k][(i+out_y_idx)*L1_INSIZE + j + out_x_idx]*w[ker_idx][k][i*L1_KERNELSIZE + j];
			}
		}
	}
	a = sum;
}

void dir_float_conv(
		float w[L1_NUMKERNEL][L1_CHL][L1_KERNELSIZE*L1_KERNELSIZE],
		float x[L1_CHL][L1_INSIZE*L1_INSIZE],
		float a[L1_NUMKERNEL][L1_OUTSIZE*L1_OUTSIZE])
{
#pragma HLS INLINE OFF
#pragma HLS ARRAY_PARTITION variable=w cyclic factor=PARTITION_CONV_FACTOR dim=3
#pragma HLS ARRAY_PARTITION variable=x cyclic factor=PARTITION_CONV_FACTOR dim=2
	conv1:for(int i=0; i<L1_NUMKERNEL; i++)
	{
		conv2:for(int k=0; k<L1_OUTSIZE; k++)
		{
#pragma HLS PIPELINE II=1
			conv3:for(int l=0; l<L1_OUTSIZE; l++)
			{
#pragma HLS UNROLL
				dir_float_kernel_sum(w, x, l, k, i, a[i][k*L1_OUTSIZE + l]);
			}
		}
	}
}


void dir_float_kernel_sum2(
		float w[L2_NUMKERNEL][L2_CHL][L2_KERNELSIZE*L2_KERNELSIZE],
		float x[L2_CHL][L2_INSIZE*L2_INSIZE],
		int out_x_idx, int out_y_idx, int ker_idx, float &a)
{
#pragma HLS INLINE
	float sum = 0;
	kernel_sum1:for(int k=0; k<L2_CHL; k++)
	{
		kernel_sum2:for(int i=0; i<L2_KERNELSIZE; i++)
		{
			kernel_sum3:for(int j=0; j<L2_KERNELSIZE; j++)
			{
				sum += x[k][(i+out_y_idx)*L2_INSIZE + j + out_x_idx]*w[ker_idx][k][i*L2_KERNELSIZE + j];
			}
		}
	}
	a = sum;
}


void dir_float_conv2(
		float w[L2_NUMKERNEL][L2_CHL][L2_KERNELSIZE*L2_KERNELSIZE],
		float x[L2_CHL][L2_INSIZE*L2_INSIZE],
		float a[L2_NUMKERNEL][L2_OUTSIZE*L2_OUTSIZE])
{
#pragma HLS INLINE OFF
#pragma HLS ARRAY_PARTITION variable=w cyclic factor=PARTITION_CONV_FACTOR dim=3
#pragma HLS ARRAY_PARTITION variable=x cyclic factor=PARTITION_CONV_FACTOR dim=2
	conv1:for(int i=0; i<L2_NUMKERNEL; i++)
	{
		conv2:for(int k=0; k<L2_OUTSIZE; k++)
		{
#pragma HLS PIPELINE II=1
			conv3:for(int l=0; l<L2_OUTSIZE; l++)
			{
#pragma HLS UNROLL
				dir_float_kernel_sum2(w, x, l, k, i, a[i][k*L2_OUTSIZE + l]);
			}
		}
	}
}

void dir_matmul3(float w[L3_ROW][L3_COL], float x[L3_ROW], float y[L3_COL])
{
#pragma HLS INLINE OFF
#pragma HLS ARRAY_PARTITION variable=w cyclic factor=PARTITION_DENSE_FACTOR dim=1
#pragma HLS ARRAY_PARTITION variable=x cyclic factor=PARTITION_DENSE_FACTOR dim=1
	matmul1:for(int i=0; i<L3_COL; i++)
	{
#pragma HLS PIPELINE II=1
		float tmp_y = 0;
		matmul2:for(int j=0; j<L3_ROW; j++)
		{
			tmp_y += w[j][i]*x[j];
		}
		y[i] = tmp_y;
	}
}

void dir_matmul4(float w[L4_ROW][L4_COL], float x[L4_ROW], float y[L4_COL])
{
#pragma HLS INLINE OFF
#pragma HLS ARRAY_PARTITION variable=w cyclic factor=PARTITION_DENSE_FACTOR dim=1
#pragma HLS ARRAY_PARTITION variable=x cyclic factor=PARTITION_DENSE_FACTOR dim=1
	matmul1:for(int i=0; i<L4_COL; i++)
	{
#pragma HLS PIPELINE II=1
		a5_t tmp_y = 0;
		matmul2:for(int j=0; j<L4_ROW; j++)
		{
			tmp_y += w[j][i]*x[j];
		}
		y[i] = tmp_y;
	}
}

void dir_matmul5(float w[L5_ROW][L5_COL], float x[L5_ROW], float y[L5_COL])
{
#pragma HLS INLINE OFF
#pragma HLS ARRAY_PARTITION variable=w cyclic factor=PARTITION_DENSE_FACTOR dim=1
#pragma HLS ARRAY_PARTITION variable=x cyclic factor=PARTITION_DENSE_FACTOR dim=1
	matmul1:for(int i=0; i<L5_COL; i++)
	{
#pragma HLS PIPELINE II=1
		float tmp_y = 0;
		matmul2:for(int j=0; j<L5_ROW; j++)
			tmp_y += w[j][i]*x[j];
		y[i] = tmp_y;
	}
}

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
	float o_r5[L5_COL];
	dir_float_conv(w1, x, o_c1);
	sw::batch_nor<float, L1_NUMKERNEL, L1_OUTSIZE*L1_OUTSIZE>(o_c1, mean1, std1, gamma1, beta1, o_b1);
	sw::relu<float, L1_NUMKERNEL, L1_OUTSIZE*L1_OUTSIZE>(o_b1, o_r1);
	sw::max_pool2x2<float, L1_NUMKERNEL, L1_OUTSIZE*L1_OUTSIZE>(o_r1, o_m1);

	dir_float_conv2(w2, o_m1, o_c2);
	sw::batch_nor<float, L2_NUMKERNEL, L2_OUTSIZE*L2_OUTSIZE>(o_c2, mean2, std2, gamma2, beta2, o_b2);
	sw::relu<float, L2_NUMKERNEL, L2_OUTSIZE*L2_OUTSIZE>(o_b2, o_r2);
	sw::max_pool2x2<float, L2_NUMKERNEL, L2_OUTSIZE*L2_OUTSIZE>(o_r2, o_m2);
	sw::flatten<float, L2_NUMKERNEL, L2_OUTSIZE*L2_OUTSIZE/4>(o_m2, o_f2);

	dir_matmul3(w3, o_f2, o_c3);
	sw::batch_nor<float, L3_COL>(o_c3, mean3, std3, gamma3, beta3, o_b3);
	sw::relu<float, L3_COL>(o_b3, o_r3);

	dir_matmul4(w4, o_r3, o_c4);
	sw::batch_nor<float, L4_COL>(o_c4, mean4, std4, gamma4, beta4, o_b4);
	sw::relu<float, L4_COL>(o_b4, o_r4);

	dir_matmul5(w5, o_r4, o_c5);
	sw::softmax<float, L5_COL>(o_c5, out);
}


void conv(
		w1_t w[L1_NUMKERNEL][L1_CHL][L1_KERNELSIZE*L1_KERNELSIZE],
		a1_t x[L1_CHL][L1_INSIZE*L1_INSIZE],
		a1_t a[L1_NUMKERNEL][L1_OUTSIZE*L1_OUTSIZE])
{
#pragma HLS INLINE OFF
	conv1:for(int i=0; i<L1_NUMKERNEL; i++)
	{
		conv2:for(int k=0; k<L1_OUTSIZE; k++)
		{
			conv3:for(int l=0; l<L1_OUTSIZE; l++)
			{
				kernel_sum(w, x, l, k, i, a[i][k*L1_OUTSIZE + l]);
			}
		}
	}
}

void dir_conv(
		w1_t w[L1_NUMKERNEL][L1_CHL][L1_KERNELSIZE*L1_KERNELSIZE],
		a1_t x[L1_CHL][L1_INSIZE*L1_INSIZE],
		a1_t a[L1_NUMKERNEL][L1_OUTSIZE*L1_OUTSIZE])
{
#pragma HLS INLINE OFF
#pragma HLS ARRAY_PARTITION variable=w cyclic factor=PARTITION_CONV_FACTOR dim=3
#pragma HLS ARRAY_PARTITION variable=x cyclic factor=PARTITION_CONV_FACTOR dim=2
	conv1:for(int i=0; i<L1_NUMKERNEL; i++)
	{
		conv2:for(int k=0; k<L1_OUTSIZE; k++)
		{
#pragma HLS PIPELINE II=1
			conv3:for(int l=0; l<L1_OUTSIZE; l++)
			{
#pragma HLS UNROLL
				dir_kernel_sum(w, x, l, k, i, a[i][k*L1_OUTSIZE + l]);
			}
		}
	}
}

void conv_bin(
		w2_t w[L2_NUMKERNEL][L2_CHL][L2_KERNELSIZE*L2_KERNELSIZE],
		a2_t x[L2_CHL][L2_INSIZE*L2_INSIZE],
		a2_t a[L2_NUMKERNEL][L2_OUTSIZE*L2_OUTSIZE])
{
#pragma HLS INLINE OFF
	conv_bin1:for(int i=0; i<L2_NUMKERNEL; i++)
	{
		conv_bin2:for(int k=0; k<L2_OUTSIZE; k++)
		{
			conv_bin3:for(int l=0; l<L2_OUTSIZE; l++)
			{
				bin_kernel_sum(w, x, l, k, i, a[i][k*L2_OUTSIZE + l]);
			}
		}
	}
}

void dir_conv_bin(
		w2_t w[L2_NUMKERNEL][L2_CHL][L2_KERNELSIZE*L2_KERNELSIZE],
		a2_t x[L2_CHL][L2_INSIZE*L2_INSIZE],
		a2_t a[L2_NUMKERNEL][L2_OUTSIZE*L2_OUTSIZE])
{
#pragma HLS ARRAY_PARTITION variable=w cyclic factor=PARTITION_CONV_FACTOR dim=3
#pragma HLS ARRAY_PARTITION variable=x cyclic factor=PARTITION_CONV_FACTOR dim=2
#pragma HLS INLINE OFF
	conv_bin1:for(int i=0; i<L2_NUMKERNEL; i++)
	{
		conv_bin2:for(int k=0; k<L2_OUTSIZE; k++)
		{
#pragma HLS PIPELINE II=1
			conv_bin3:for(int l=0; l<L2_OUTSIZE; l++)
			{
#pragma HLS UNROLL
				dir_bin_kernel_sum(w, x, l, k, i, a[i][k*L2_OUTSIZE + l]);
			}
		}
	}
}

void bin_half_matmul(w4_t w[L4_ROW][L4_COL], a4_t x[L4_ROW], a4_t y[L4_COL])
{
#pragma HLS INLINE OFF
	a4_t tmp_wx = 0;
	bin_half1:for(int i=0; i<L4_COL; i++)
	{
		a4_t tmp_y = 0;
		bin_half2:for(int j=0; j<L4_ROW; j++)
		{
			xor_signed_bit(w[j][i], x[j], tmp_wx);
			tmp_y += tmp_wx;
		}
		y[i] = tmp_y;
	}
}

void dir_bin_half_matmul(w4_t w[L4_ROW][L4_COL], a4_t x[L4_ROW], a4_t y[L4_COL])
{
#pragma HLS INLINE OFF
#pragma HLS ARRAY_PARTITION variable=w cyclic factor=PARTITION_DENSE_FACTOR dim=1
#pragma HLS ARRAY_PARTITION variable=x cyclic factor=PARTITION_DENSE_FACTOR dim=1
	a4_t tmp_wx = 0;
	bin_half1:for(int i=0; i<L4_COL; i++)
	{
#pragma HLS PIPELINE II=1
		a4_t tmp_y = 0;
		bin_half2:for(int j=0; j<L4_ROW; j++)
		{
			xor_signed_bit(w[j][i], x[j], tmp_wx);
			tmp_y += tmp_wx;
		}
		y[i] = tmp_y;
	}
}

void matmul(w5_t w[L5_ROW][L5_COL], a5_t x[L5_ROW], a5_t y[L5_COL])
{
#pragma HLS INLINE OFF
	matmul1:for(int i=0; i<L5_COL; i++)
	{
		a5_t tmp_y = 0;
		matmul2:for(int j=0; j<L5_ROW; j++)
			tmp_y += w[j][i]*x[j];
		y[i] = tmp_y;
	}
}

void dir_matmul(w5_t w[L5_ROW][L5_COL], a5_t x[L5_ROW], a5_t y[L5_COL])
{
#pragma HLS INLINE OFF
#pragma HLS ARRAY_PARTITION variable=w cyclic factor=PARTITION_DENSE_FACTOR dim=1
#pragma HLS ARRAY_PARTITION variable=x cyclic factor=PARTITION_DENSE_FACTOR dim=1
	matmul1:for(int i=0; i<L5_COL; i++)
	{
#pragma HLS PIPELINE II=1
		a5_t tmp_y = 0;
		matmul2:for(int j=0; j<L5_ROW; j++)
			tmp_y += w[j][i]*x[j];
		y[i] = tmp_y;
	}
}


void dir_conv1(
		w1_t w[L1_NUMKERNEL][L1_CHL][L1_KERNELSIZE*L1_KERNELSIZE],
		a1_t x[L1_CHL][L1_INSIZE*L1_INSIZE],
		a1_t a[L1_NUMKERNEL][L1_OUTSIZE*L1_OUTSIZE])
{
#pragma HLS INLINE OFF
#pragma HLS ARRAY_PARTITION variable=w cyclic factor=PARTITION_CONV_FACTOR dim=3
#pragma HLS ARRAY_PARTITION variable=x cyclic factor=PARTITION_CONV_FACTOR dim=2
	conv1:for(int i=0; i<L1_NUMKERNEL; i++)
	{
		conv2:for(int k=0; k<L1_OUTSIZE; k++)
		{
#pragma HLS PIPELINE II=1
			conv3:for(int l=0; l<L1_OUTSIZE; l++)
			{
#pragma HLS UNROLL
				dir_kernel_sum(w, x, l, k, i, a[i][k*L1_OUTSIZE + l]);
			}
		}
	}
}

void dir_matmul3_ternary(int2 w[L3_ROW][L3_COL], half x[L3_ROW], half y[L3_COL])
{
#pragma HLS INLINE OFF
#pragma HLS ARRAY_PARTITION variable=w cyclic factor=PARTITION_DENSE_FACTOR dim=1
#pragma HLS ARRAY_PARTITION variable=x cyclic factor=PARTITION_DENSE_FACTOR dim=1
	half tmp_wx = 0;
	matmul1:for(int i=0; i<L3_COL; i++)
	{
#pragma HLS PIPELINE II=1
		half tmp_y = 0;
		matmul2:for(int j=0; j<L3_ROW; j++)
		{
			half tmp_sum = 0;
			ternary_bitwise(w[j][i], x[j], tmp_wx);
			tmp_y += tmp_wx;
		}
		y[i] = tmp_y;
	}
}

void dir_matmul4_ternary(int2 w[L4_ROW][L4_COL], half x[L4_ROW], half y[L4_COL])
{
#pragma HLS INLINE OFF
#pragma HLS ARRAY_PARTITION variable=w cyclic factor=PARTITION_DENSE_FACTOR dim=1
#pragma HLS ARRAY_PARTITION variable=x cyclic factor=PARTITION_DENSE_FACTOR dim=1
	half tmp_wx = 0;
	matmul1:for(int i=0; i<L4_COL; i++)
	{
#pragma HLS PIPELINE II=1
		half tmp_y = 0;
		matmul2:for(int j=0; j<L4_ROW; j++)
		{
			half tmp_sum = 0;
			ternary_bitwise(w[j][i], x[j], tmp_wx);
			tmp_y += tmp_wx;
		}
		y[i] = tmp_y;
	}
}

void dir_matmul5(w5_t w[L5_ROW][L5_COL], a5_t x[L5_ROW], a5_t y[L5_COL])
{
#pragma HLS INLINE OFF
#pragma HLS ARRAY_PARTITION variable=w cyclic factor=PARTITION_DENSE_FACTOR dim=1
#pragma HLS ARRAY_PARTITION variable=x cyclic factor=PARTITION_DENSE_FACTOR dim=1
	matmul1:for(int i=0; i<L5_COL; i++)
	{
#pragma HLS PIPELINE II=1
		a5_t tmp_y = 0;
		matmul2:for(int j=0; j<L5_ROW; j++)
			tmp_y += w[j][i]*x[j];
		y[i] = tmp_y;
	}
}

void dir_ternary_kernel_sum(
		int2 w[L2_NUMKERNEL][L2_CHL][L2_KERNELSIZE*L2_KERNELSIZE],
		half x[L2_CHL][L2_INSIZE*L2_INSIZE],
		int out_x_idx, int out_y_idx, int ker_idx, half &a)
{
#pragma HLS INLINE
	half sum = 0;
	bin_kernel_sum1:for(int k=0; k<L2_CHL; k++)
	{
		bin_kernel_sum2:for(int i=0; i<L2_KERNELSIZE; i++)
		{
			bin_kernel_sum3:for(int j=0; j<L2_KERNELSIZE; j++)
			{
				half tmp_sum = 0;
				int2 tmp_w = w[ker_idx][k][i*L2_KERNELSIZE + j];
				half tmp_x = x[k][(i + out_y_idx)*L2_INSIZE + j + out_x_idx];
				ternary_bitwise(tmp_w, tmp_x, tmp_sum);
				sum += tmp_sum;
			}
		}
	}
	a = sum;
}

void dir_conv2_ternary(
		int2 w[L2_NUMKERNEL][L2_CHL][L2_KERNELSIZE*L2_KERNELSIZE],
		a2_t x[L2_CHL][L2_INSIZE*L2_INSIZE],
		a2_t a[L2_NUMKERNEL][L2_OUTSIZE*L2_OUTSIZE])
{
#pragma HLS ARRAY_PARTITION variable=w cyclic factor=PARTITION_CONV_FACTOR dim=3
#pragma HLS ARRAY_PARTITION variable=x cyclic factor=PARTITION_CONV_FACTOR dim=2
#pragma HLS INLINE OFF
	conv_bin1:for(int i=0; i<L2_NUMKERNEL; i++)
	{
		conv_bin2:for(int k=0; k<L2_OUTSIZE; k++)
		{
#pragma HLS PIPELINE II=1
			conv_bin3:for(int l=0; l<L2_OUTSIZE; l++)
			{
#pragma HLS UNROLL
				dir_ternary_kernel_sum(w, x, l, k, i, a[i][k*L2_OUTSIZE + l]);
			}
		}
	}
}

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
		half out[L5_COL])
{
	half o_c1[L1_NUMKERNEL][L1_OUTSIZE*L1_OUTSIZE];
	half o_b1[L1_NUMKERNEL][L1_OUTSIZE*L1_OUTSIZE];
	half o_r1[L1_NUMKERNEL][L1_OUTSIZE*L1_OUTSIZE];
	half o_m1[L1_NUMKERNEL][L1_OUTSIZE*L1_OUTSIZE/4];

	half o_c2[L2_NUMKERNEL][L2_OUTSIZE*L2_OUTSIZE];
	half o_b2[L2_NUMKERNEL][L2_OUTSIZE*L2_OUTSIZE];
	half o_r2[L2_NUMKERNEL][L2_OUTSIZE*L2_OUTSIZE];
	half o_m2[L2_NUMKERNEL][L2_OUTSIZE*L2_OUTSIZE/4];
	half o_f2[L2_NUMKERNEL*L2_OUTSIZE*L2_OUTSIZE/4];

	half o_c3[L3_COL];
	half o_b3[L3_COL];
	half o_r3[L3_COL];

	half o_c4[L4_COL];
	half o_b4[L4_COL];
	half o_r4[L4_COL];

	half o_c5[L5_COL];
	half o_r5[L5_COL];

	dir_conv1(w1, x, o_c1);
	sw::batch_nor<half, L1_NUMKERNEL, L1_OUTSIZE*L1_OUTSIZE>(o_c1, mean1, std1, gamma1, beta1, o_b1);
	sw::relu<half, L1_NUMKERNEL, L1_OUTSIZE*L1_OUTSIZE>(o_b1, o_r1);
	sw::max_pool2x2<half, L1_NUMKERNEL, L1_OUTSIZE*L1_OUTSIZE>(o_r1, o_m1);

	dir_conv2_ternary(w2, o_m1, o_c2);
	sw::batch_nor<half, L2_NUMKERNEL, L2_OUTSIZE*L2_OUTSIZE>(o_c2, mean2, std2, gamma2, beta2, o_b2);
	sw::relu<half, L2_NUMKERNEL, L2_OUTSIZE*L2_OUTSIZE>(o_b2, o_r2);
	sw::max_pool2x2<half, L2_NUMKERNEL, L2_OUTSIZE*L2_OUTSIZE>(o_r2, o_m2);
	sw::flatten<half, L2_NUMKERNEL, L2_OUTSIZE*L2_OUTSIZE/4>(o_m2, o_f2);

	dir_matmul3_ternary(w3, o_f2, o_c3);
	sw::batch_nor<half, L3_COL>(o_c3, mean3, std3, gamma3, beta3, o_b3);
	sw::relu<half, L3_COL>(o_b3, o_r3);

	dir_matmul4_ternary(w4, o_r3, o_c4);
	sw::batch_nor<half, L4_COL>(o_c4, mean4, std4, gamma4, beta4, o_b4);
	sw::relu<half, L4_COL>(o_b4, o_r4);

	dir_matmul5(w5, o_r4, o_c5);
	sw::softmax<half, L5_COL>(o_c5, out);
}
