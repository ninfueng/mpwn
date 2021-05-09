#include <iostream>
#include "utils/timer.hpp"
#include "utils/array.hpp"
#include "constants.hpp"
#include "model.hpp"

int main()
{
      TimerMicro timer;
      float w5[84][10];
      float in[84];
      float out[10];
      float_matmul5(w5, in, out);
      return 0;
}
