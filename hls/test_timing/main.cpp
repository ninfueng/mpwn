#include <iostream>
#include "utils/timer.hpp"
#include "utils/array.hpp"

#include "float_params/w0.hpp"
#include "float_params/beta0.hpp"
#include "float_params/gamma0.hpp"
#include "float_params/mean0.hpp"
#include "float_params/std0.hpp"

#include "float_params/w1.hpp"
#include "float_params/beta1.hpp"
#include "float_params/gamma1.hpp"
#include "float_params/mean1.hpp"
#include "float_params/std1.hpp"

#include "float_params/w2.hpp"
#include "float_params/beta2.hpp"
#include "float_params/gamma2.hpp"
#include "float_params/mean2.hpp"
#include "float_params/std2.hpp"

#include "float_params/w3.hpp"
#include "float_params/beta3.hpp"
#include "float_params/gamma3.hpp"
#include "float_params/mean3.hpp"
#include "float_params/std3.hpp"

#include "float_params/w4.hpp"
#include "float_params/b4.hpp"

#include "float_params/img0.hpp"
#include "float_params/img1.hpp"
#include "float_params/img2.hpp"

#include "constants.hpp"
#include "model.hpp"

int main()
{
	//Label of img0: 3, img1: 2 (Model produces wrong prediction), img2: 7.
      TimerMicro timer;
      float out[L5_COL];
      float_seq_lenet5(
		  w0, w1, w2, w3, w4, img0,
		  mean0, std0, gamma0, beta0,
		  mean1, std1, gamma1, beta1,
		  mean2, std2, gamma2, beta2,
		  mean3, std3, gamma3, beta3, out);
      return 0;
}
