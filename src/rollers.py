import numpy as np
import pandas as pd
from _rollers import RollerX

TIMESTAMP_DTYPES = [pd.DatetimeIndex, pd.datetime64, np.datetime64, np.uint64]


class Roller:
    """
    Wraps C++ Rollers
    Does type checking and preparing inputs for the c++ functions
    """
    def __init__(self, timeframes: list):
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
        self._roller = RollerX(self.timeframes_uint64_t, nzones=0)

        self.current_discrete_idx = 0

    def roll(self, data, timestamps=None):
        arr, timearr = self.prepare_data(data, timestamps)
        return self._roller.roll(arr, timearr)

    def check_timestamps(self, timestamps):
        if not isinstance(timestamps, (pd.DatetimeIndex)):
            if isinstance(timestamps, np.ndarray):
                if isinstance(timestamps.dtype, np.datetime64):
                    timestamps = timestamps.astype("uint64")
                elif not isinstance(timestamps.dtype, np.uint64):
                    raise TypeError("Timestamps passed are not of a" +
                                    "suitable dtype, must be one of" +
                                    f"{TIMESTAMP_DTYPES}")
            raise TypeError("timestamps must be either pd.DatatimeIndex or an ndarray" +
                            ", type of variable passed: {type(timestamps)}")

    def prepare_data(self, data, timestamps):
        if self.window_type == "continuous":
            if timestamps is not None:
                self.check_timestamps(timestamps)
                timestamps = np.array(timestamps).astype("uint64")
                arr = np.array(data)
            else:
                if not isinstance(data, pd.Series):
                    raise TypeError("timestamps have not been passed " +
                                    "yet data is not of type pd.Series" +
                                    " with a datetime index")
                if not isinstance(data.index, pd.DatetimeIndex):
                    raise TypeError("timestamps have not been passed " +
                                    "yet data.index is not of type pd.DatetimeIndex")
                timestamps = np.array(data.index).astype("uint64")
                arr = data.values
        else:
            arr = np.array(data)
            timearr = np.arange(self.current_discrete_idx,
                                self.current_discrete_idx + len(data))
            self.current_discrete_idx += len(data)
            return arr, timearr
        return arr, timestamps
