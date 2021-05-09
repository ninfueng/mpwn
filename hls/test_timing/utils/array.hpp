#ifndef __ARRAY_H__
#define __ARRAY_H__

#include <iostream>
#include <fstream>
#define log(x) std::cout << x << '\n'

static const int MAX_RAND = 100;

namespace array
{
	template <typename T, int D1>
	void print(T var[D1])
	{
		for(int i=0; i<D1; i++)
		{
			log('|' << i  << "|= " << var[i]);
		}
		log(' ');
	}

	template <typename T, int D1, int D2>
	void print(T var[D1][D2])
	{
		for(int i=0; i<D1; i++)
		{			for(int j=0; j<D2; j++)
			{
				log('|' << i << '|' << j  << "|= " << var[i][j]);
			}
		}
		log(' ');
	}

	template <typename T, int D1, int D2, int D3>
	void print(T var[D1][D2][D3])
	{
		for(int i=0; i<D1; i++)
		{
			for(int j=0; j<D2; j++)
			{
				for(int k=0; k<D3; k++)
				{
					log('|' << i << '|' << j  << '|' << k << "|= " << var[i][j][k]);
				}
			}
		}
		log(' ');
	}
	
	template <typename T, int D1, int D2, int D3, int D4>
	void print(T var[D1][D2][D3][D4])
	{
		for(int i=0; i<D1; i++)
		{
			for(int j=0; j<D2; j++)
			{
				for(int k=0; k<D3; k++)
				{
					for(int l=0; l<D4; l++)
					{
						log('|' << i << '|' << j  << '|' << k << '|' << l << "|= " << var[i][j][k][l]);	
					}
				}
			}
		}
		log(' ');
	}
	

	template <typename T>
	void load_rand_uint(T var, int max=MAX_RAND)
	{
		var = rand()%max;
	}

	template <typename T, int D1>
	void load_rand_uint(T var[D1], int max=MAX_RAND)
	{
		L1:for(int i=0; i<D1; i++)
			var[i] = rand()%max;
	}

	template <typename T, int D1, int D2>
	void load_rand_uint(T var[D1][D2], int max=MAX_RAND)
	{
		L1:for(int i=0; i<D1; i++)
			L2:for(int j=0; j<D2; j++)
				var[i][j] = rand()%max;
	}

	template <typename T, int D1, int D2, int CHL>
	void load_rand_uint(T var[D1][D2][CHL], int max=MAX_RAND)
	{
		L1:for(int i=0; i<D1; i++)
			L2:for(int j=0; j<D2; j++)
				L3:for(int k=0; k<CHL; k++)
					var[i][j][k] = rand()%max;
	}

	template <typename T>
	void load_const(T &var, T const_=0)
	{
		var = const_;
	}

	template <typename T, int D1>
	void load_const(T var[D1], T constant=0)
	{
		L1:for(int i=0; i<D1; i++)
			var[i] = constant;
	}

	template <typename T, int D1, int D2>
	void load_const(T var[D1][D2], T constant=0)
	{
		L1:for(int i=0; i<D1; i++)
			L2:for(int j=0; j<D2; j++)
				var[i][j] = constant;
	}

	template <typename T, int D1, int D2, int D3>
	void load_const(T var[D1][D2][D3], T constant=0)
	{
		L1:for(int i=0; i<D1; i++)
			L2:for(int j=0; j<D2; j++)
				L3:for(int k=0; k<D3; k++)
					var[i][j][k] = constant;
	}

	template <typename T, int D1>
	void measure_sparse(T var[D1])
	{
		int zero = 0;
		int nonzero = 0;
		L1:for(int i=0; i<D1; i++)
		{
			if(var[i] == 0)
				zero++;
			else
				nonzero++;
		}
		log("Sparsity: " <<  zero/(nonzero+zero));
	}

	template <typename T, int D1, int D2>
	void measure_sparse(T var[D1][D2])
	{
		int zero = 0;
		int nonzero = 0;
		L1:for(int i=0; i<D1; i++)
		{
			L2:for(int j=0; j<D2; j++)
			{
				if(var[i][j] == 0)
					zero++;
				else
					nonzero++;

			}
		}
		log("Sparsity: " <<  zero/(nonzero+zero));
	}

	template <typename T, int D1, int D2, int D3>
	void measure_sparse(T var[D1][D2][D3])
	{
		int zero = 0;
		int nonzero = 0;
		L1:for(int i=0; i<D1; i++)
		{
			L2:for(int j=0; j<D2; j++)
			{
				L3:for(int k=0; k<D3; k++)
				{
					if(var[i][j][k] == 0)
						zero++;
					else
						nonzero++;
				}
			}
		}
		log("Sparsity: " <<  zero/(nonzero+zero));
	}

	template <typename T, int D1>
	void load_txt(std::string txt_locat, T array[D1])
	{
		std::ifstream FILE;
		FILE.open(txt_locat);
		if(FILE.fail())
		{
			log("ERROR cannot open the file: " << txt_locat);
			exit(1);
		}
		T var = 0;
		int d1_count = 0;
		while(FILE >> var)
		{
			array[d1_count] = var;
			//printf("Loading 1D [%i]: %f\n", d1_count, array1d[d1_count]);
			d1_count += 1;
		}
		FILE.close();
		log("Finishing loading!!!");
	}

	template <typename T, int D1, int D2>
	void load_txt(std::string txt_locat, T array[D1][D2])
	{
		std::ifstream FILE;
		FILE.open(txt_locat);
		if(FILE.fail())
		{
			log("ERROR cannot open the file: " << txt_locat);
			exit(1);
		}
		T var = 0;
		int d1_count = 0;
		int d2_count = 0;

		while(FILE >> var)
		{
			array[d1_count][d2_count]= var;
			//printf("Loading 2D [%i][%d]: %f\n", d1_count, d2_count, array[d1_count][d2_count]);

			d2_count += 1;
			if(d2_count > D2-1)
			{
				d1_count += 1;
				d2_count = 0;
			}
		}
		FILE.close();
		log("Finishing loading!!!");
	}

	template <typename T, int D1, int D2, int D3>
	void load_txt(std::string txt_locat, T array[D1][D2][D3] )
	{
		std::ifstream FILE;
		FILE.open(txt_locat);
		if(FILE.fail())
		{
			log("ERROR cannot open the file: " << txt_locat);
			exit(1);
		}
		T var = 0;
		int d1_count = 0;
		int d2_count = 0;
		int d3_count = 0;

		while(FILE >> var)
		{
			array[d1_count][d2_count][d3_count] = var;
			//printf("Loading 3D [%i][%d][%i]: %f\n", d1_count, d2_count, d3_count, img[d1_count][d2_count][d3_count]);

			d2_count += 1;
			if(d2_count > D2-1)
			{
				d1_count += 1;
				d2_count = 0;
				if(d1_count > D1-1)
				{
					d1_count = 0;
					d3_count += 1;
				}
			}
		}
		FILE.close();
		log("Finishing loading!!!");
	}

	template <typename T, int D1, int D2, int D3, int D4>
	void load_txt(std::string txt_locat, T array[D1][D2][D3][D4])
	{
		std::ifstream FILE;
		FILE.open(txt_locat);
		if(FILE.fail())
		{
			log("ERROR cannot open the file: " << txt_locat);
			exit(1);
		}
		T var = 0;
		int d1_count = 0;
		int d2_count = 0;
		int d3_count = 0;
		int d4_count = 0;

		while(FILE >> var)
		{
			array[d1_count][d2_count][d3_count][d4_count] = var;
			//printf("Loading 4D [%i][%d][%i][%i]: %f\n", d1_count, d2_count,  d3_count,  d4_count, array[d1_count][d2_count][d3_count][d4_count]);

			d2_count += 1;
			if(d2_count > D2-1)
			{
				d1_count += 1;
				d2_count = 0;
				if(d1_count > D1-1)
				{
					d1_count = 0;
					d3_count += 1;

					if(d3_count > D3-1)
					{
						d3_count = 0;
						d4_count += 1;
					}
				}
			}
		}
		FILE.close();
		log("Finishing loading!!!");
	}
}

#endif
