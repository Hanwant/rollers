from typing import Union, List

import numpy as np
import pandas as pd
from _rollers import RollerX

TIMESTAMP_DTYPES = (np.datetime64, np.uint64)


class Roller:
    """
    Wraps C++ Rollers for a single price/time series
    Does type checking and preparing inputs for the c++ functions
    """
    def __init__(self, timeframes: list):
        self.register_timeframes(timeframes)
        self._roller = RollerX(self.timeframes_uint64_t, nzones=0)

        # Important as discrete windows are implemented ad-hoc
        # Given that time offset arithmetic is performed inside rollers
        # Ie in tail_update function, subtraction which would lead to
        # negative numbers on uint64 will instead lead to incorrect behaviour
        # and seg fault when eventually a wrong index is accessed inside the
        # inner loop of tail_update - outer safeguards don't get chance to
        # operate
        self.current_discrete_idx = max(self.timeframes_uint64_t)

    def register_timeframes(self, timeframes):
        all_same_type = all([isinstance(tf, type(timeframes[0])) for tf in timeframes])
        if not all_same_type:
            raise TypeError("All elements of timeframes list must be of the same type")
        if not isinstance(timeframes[0], (pd.Timedelta, str, int)):
            raise ValueError("timeframes must be either integers for fixed windows"+
                             " or in (str, pd.Timedelta) for time based windows")
        if isinstance(timeframes[0], int):
            self.window_type = "discrete"
            self.timeframes = timeframes
            self.timeframes_uint64_t = self.timeframes
        elif isinstance(timeframes[0], (pd.Timedelta, str)):
            self.window_type = "continuous"
            self.timeframes = [pd.Timedelta(tf) for tf in timeframes]
            self.timeframes_uint64_t = [tf.value for tf in self.timeframes]

    def check_timestamps(self, timestamps):
        """
        Checks whether timestamps are of valid type
        """
        if not isinstance(timestamps, (pd.DatetimeIndex)):
            if isinstance(timestamps, np.ndarray):
                if not any([timestamps.dtype.type is dtype for \
                             dtype in TIMESTAMP_DTYPES]):
                    raise TypeError("Timestamps passed are not of a" +
                                    "suitable dtype, must be one of" +
                                    f"{TIMESTAMP_DTYPES}")
            else: # not pd.DatetimeIndex or np.ndarray
                raise TypeError(f"timestamps must be either pd.DatatimeIndex or an ndarray" +
                                ", type of variable passed: {type(timestamps)}")

    def prepare_data(self, data, timestamps):
        if self.window_type == "continuous":
            if timestamps is not None:
                self.check_timestamps(timestamps)
                if isinstance(timestamps, (pd.Series, pd.DatetimeIndex)):
                    timestamps = timestamps.values.astype("uint64")
                elif isinstance(timestamps, np.ndarray):
                    timestamps = timestamps.astype("uint64")
                else:
                    raise TypeError("timestamps must be one of types: " +
                                    "pd.Series, pd.Datetimeindex, np.ndarray")
                arr = np.array(data, copy=False)
            else:
                if not isinstance(data, pd.Series):
                    raise TypeError("timestamps have not been passed " +
                                    "yet data is not of type pd.Series" +
                                    " with a datetime index")
                if not isinstance(data.index, pd.DatetimeIndex):
                    raise TypeError("timestamps have not been passed " +
                                    "yet data.index is not of type pd.DatetimeIndex")
                timestamps = data.index.values.astype("uint64")
                arr = data.values
        else:
            arr = np.array(data)
            timearr = np.arange(self.current_discrete_idx+1,
                                self.current_discrete_idx + len(data)+1)
            self.current_discrete_idx += len(data)
            return arr, timearr.astype("uint64")
        return arr, timestamps

    def roll(self, data, timestamps=None):
        arr, timearr = self.prepare_data(data, timestamps)
        return np.array(self._roller.roll(arr, timearr), copy=False)


class RollerMulti(Roller):
    """
    UNTESTED
    Aggregates features for multiple price/time series
    Uses the same timeframes for each.
    """
    def __init__(self, timeframes: list, n_series: int):
        self.register_timeframes(timeframes)
        self.n_series = n_series
        self._rollers = []
        for _ in range(n_series):
            self._rollers.append(RollerX(self.timeframes_uint64_t, nzones=0))
        self.current_discrete_idx = max(self.timeframes_uint64_t)

    def roll(self, data: List[np.ndarray],
             timestamps: Union[np.ndarray, list] = None,
             aggregate_array: bool = False):
        out = []
        for i, dat in enumerate(data):
            if isinstance(timestamps, list):
                ts = timestamps[i]
            else:
                ts = timestamps
            single_out = super().roll(dat, ts)
            out.append(single_out)
        if aggregate_array:
            return np.stack(out, axis=-1)
        return out
