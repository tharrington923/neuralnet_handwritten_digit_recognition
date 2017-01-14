#ifndef ALIASES_H_
#define ALIASES_H_

#include <vector>

/* Alias for a vector of doubles */
typedef std::vector<double> dVector;

/* Alias for a vector of vectors of doubles */
/* This is used for storing the biases at each layer.  */
/* Alias for a vector of vectors of doubles */
typedef std::vector<std::vector<double> > bVector;

/* Alias for a vector  */
typedef std::vector<std::vector<std::vector<double> > > wVector;

#endif
