#include "norm.h"
float norm(float *a,float *b,int n){
	float c = 0;
	for(int i = 0; i< n; i++){
		c = c + (a[i] - b[i])*(a[i] - b[i]);
	}
	c = sqrt(c);
	return c;
}
