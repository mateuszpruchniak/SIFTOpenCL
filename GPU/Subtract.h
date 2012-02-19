
#pragma once

#include "../GPUBase.h"


class Subtract :
	public GPUBase
{
public:

	/*!
	* Destructor.
	*/
	~Subtract(void);

	/*!
	* Constructor.
	*/
	Subtract();
	
	bool Process();
};
