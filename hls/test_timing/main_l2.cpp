#include <iostream>
#include "utils/timer.hpp"
#include "utils/array.hpp"
#include "constants.hpp"
#include "model.hpp"

int main()
{
      //Label of img0: 3, img1: 2 (Model produces wrong prediction), img2: 7.
      TimerMicro timer;
      float in[6][12*12];
      float out[16][8*8];
      float w1[16][6][5*5];
      float_conv2(w1, in, out);
      return 0;
}
