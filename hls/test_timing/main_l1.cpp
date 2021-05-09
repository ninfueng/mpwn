#include <iostream>
#include "utils/timer.hpp"
#include "utils/array.hpp"
#include "constants.hpp"
#include "model.hpp"

int main()
{
      //Label of img0: 3, img1: 2 (Model produces wrong prediction), img2: 7.
      TimerMicro timer;
      float in[1][28*28];
      float out[6][24*24];
      float w1[6][1][5*5];
      float_conv1(w1, in, out);
      return 0;
}
