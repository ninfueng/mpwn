#include <iostream>
#include "utils/timer.hpp"
#include "utils/array.hpp"
#include "constants.hpp"
#include "model.hpp"

int main()
{
      TimerMicro timer;
      float w3[256][120];
      float in[256];
      float out[120];
      float_matmul3(w3, in, out);
      return 0;
}
