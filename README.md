# Rollers
(Work in Progress)
### Efficient stateful moving window functions for timeseries.
This Python/C++ library contains implementations of algorithms for calcuating mean, min, max, var & volatility on sliding time-based windows, using the appropriate algorithms<sup>[1]</sup> and data structures<sup>[2]</sup> optimized for computational complexity. 
Application is intended for finanical times series, hence the set of functions available are open, high low, close, mean, log returns\*, var and volatility. 
These 8 functions are performed for each new datapoint and aggregated into one matrix for any number of timeframes specified. 
The windowed functions may aggregate based on either continuous time, with variable window sizes and no assumption of the data being evenly sampled, or fixed window sizes which correspond to periods defined by number of data points sampled, instead of actual time.
<br>

The main use case of such a library is that the classes provide a ***stateful*** object which can update state for singular data updates as well as batches. As it accumulates and retains the necessary memory to avoid redundant recomputation, this is ideal for processing ***streaming*** data
or for chunking data which is too large for memory into batches.  

\* Calculation of log returns assumes all values are positive (log(-new/old) == undefined) 
<br>
## Usage
```python
import numpy as np
import pandas as pd
from rollers import Roller

# example data
N = 10000
arr = np.abs(np.random.randn(N)) # abs to make all values positive
timestamps = pd.date_range(start = "2002-01-01", periods = N,
                           freq="1min", tz="utc")
series = pd.Series(arr, index=timestamp)

# specify timeframes for moving window functions
timeframes = ["15min", "30min", "60min", "4h", "12h"]

# do the roll
roller = Roller(timeframes)
out = roller.roll(series)

assert out.shape == (N, 8, len(timeframes)) # out is a 3d array

# Re-rolling on the same data will produce different results !
# As the roller object is stateful
assert not np.allclose(out, roller.roll(series))

```

## Installation
#### Requirements
A C++ compiler is needed along with CMake for compiling the c++ source into a shared object library which is used from python. 
* Pybind11 - for binding c++ -> python
* Numpy - as core data structure
* Pandas - for datetime handling
* C++11 Compiler
* CMake 

To run tests: 
* pytest

#### To install:  
Navigate to main directory containing setup.py and:  

    python setup.py install  
or:  

    pip install .  
    

### To Do:
* Write proper discrete version of Roller class. Currently fixed window functions are ad-hoc by providing the continuous time-based functions an array of contrived timestamps. 
* Tests for Roller Wrapper. Currently tests operate on RollerX directly.
* Use either templating or dynamic dispatch / overloading to allow user to select
which time series functions they want, instead of providing all 8 by default.
* Switch to Eigen Tensor instead of boost multi for multi-dimenional arrays
* Reconsider data structures for storing memory - must be efficient to resize/re-allocate 

<br><br><br>

[1]. Mean/Var calculations derive from [Welford's online algorithm](
https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance), using a version
which adjusts for moving windows\*.\
[2]. Min/Max algorithm uses a deque\* \


\* Similar implementations will be found in the source code for pandas, which is used a reference for development and testing.
