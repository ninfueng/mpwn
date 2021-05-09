#include <iostream>

#include "hw.hpp"
#include "ap_int.h"
#include "ap_fixed.h"
#include "hls_half.h"
#include "utils/timer.hpp"
#include "utils/array.hpp"

#include "params/w0.hpp"
#include "params/beta0.hpp"
#include "params/gamma0.hpp"
#include "params/mean0.hpp"
#include "params/std0.hpp"

#include "params/w1.hpp"
#include "params/beta1.hpp"
#include "params/gamma1.hpp"
#include "params/mean1.hpp"
#include "params/std1.hpp"

#include "params/w2.hpp"
#include "params/beta2.hpp"
#include "params/gamma2.hpp"
#include "params/mean2.hpp"
#include "params/std2.hpp"

#include "params/w3.hpp"
#include "params/beta3.hpp"
#include "params/gamma3.hpp"
#include "params/mean3.hpp"
#include "params/std3.hpp"

#include "params/w4.hpp"
#include "params/b4.hpp"

#include "params/img0.hpp"
#include "params/img1.hpp"
#include "params/img2.hpp"


int main()
{
	//Label of img0: 3, img1: 2 (Model produces a wrong prediction), img2: 7.
	TimerMicro timer;
	half out[L4_COL];

	dir_float_ftttf(
		w0, w1, w2, w3, w4, img0,
		mean0, std0, gamma0, beta0,
		mean1, std1, gamma1, beta1,
		mean2, std2, gamma2, beta2,
		mean3, std3, gamma3, beta3, out);

	array::print<half, L5_COL>(out);

	return 0;
}
