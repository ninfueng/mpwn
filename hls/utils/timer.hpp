#ifndef __TIMER_H__ 
#define __TIMER_H__ 

#include <chrono> 
#include <iostream>

struct TimerMicro
{
	std::chrono::time_point<std::chrono::system_clock> start, end;
	std::chrono::duration<double> duration;
	TimerMicro()
	{
		start =  std::chrono::system_clock::now();
	}
	~TimerMicro()
	{
		end =  std::chrono::system_clock::now();
		duration = start - end;
		std::cout << "Timer: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " us\n";
	}
};

#endif
