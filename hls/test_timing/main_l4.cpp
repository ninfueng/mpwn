#include <iostream>
#include "utils/timer.hpp"
#include "utils/array.hpp"
#include "constants.hpp"
#include "model.hpp"

int main()
{
      TimerMicro timer;
      float w4[120][84];
      float in[120];
      float out[84];
      float_matmul4(w4, in, out);
      return 0;
}
