#!/home/hemu/miniconda3/envs/madigan/bin/python3.7
import sys
import time
from pathlib import Path
import pytest
import timeit
import numpy as np
import pandas as pd
import _rollers as rollers
from preprocessing import rollers as rollers_cy
# from preprocessing.utils import SYDNEY, SYDNEY_HOURS, NY, NY_HOURS, TOKYO, TOKYO_HOURS, LONDON, LONDON_HOURS
from algotrading_utils.utils import time_profile

import pytz

SYDNEY = pytz.timezone('Australia/Sydney')
TOKYO = pytz.timezone('Asia/Tokyo')
LONDON = pytz.timezone('Europe/London')
NY = pytz.timezone('US/Eastern')
############### Trading hours (IN LOCAL TIME WHICH DOESN'T CHANGE WITH DST) for Trading Sessions
SYDNEY_HOURS = (7, 16) # start , close
TOKYO_HOURS = (9, 18)
LONDON_HOURS = (8, 16)
NY_HOURS = (8, 17)

N = 10000
timestamps= pd.date_range(start = "2002-01-01", periods = N, freq="1min", tz="utc")
timearr = timestamps.values.astype("uint64")
base = np.linspace(0, 5, N)
arr = np.abs(base * 2 + np.sin(base) + np.random.normal(0, 1, N))
arr = np.abs(np.random.randn(N))
timedeltas = [pd.Timedelta(t) for t in ["15m", "30m", "1h", "2h", "4h", "12h", "1d"]]
tfs = [tdelta.value for tdelta in timedeltas]


@pytest.mark.skip(reason="Utility Plotting Function")
def plot(out, ref, feats=[0, 1, 2, 3, 4, 5, 6, 7], tfs = "all", timestamps=None, sharex=True):
    """
    out: Ndarray from rollerX/Y
    ref: NDarray from pandas calculation functions
    feats: = list of indices for which features to plot, each feature is a separate plot
    tfs: either "all" or a list of timeframe indices, plotted on same graph for each feature
    """
    assert tfs == "all" or type(tfs) == list
    import matplotlib.pyplot as plt
    cols = min(len(feats), 4)
    rows = (len(feats) // 4) if len(feats) % 4 == 0 else len(feats)//4 + 1
    tfs = list(range(out.shape[2])) if tfs=="all" else tfs
    fig, ax = plt.subplots(rows, cols, squeeze=False, sharex=sharex)
    # r, c = 0
    for i, feat in enumerate(feats):
        r = i // cols
        c = i - (r * cols)
        if (r*cols+c) < len(feats):
            _ax = ax[r, c]
            for tf in tfs:
                _ax.plot(pd.Series(ref[:, feat, tf], index=timestamps), label=f"ref feat={r*cols+c} tf={tf}")
                _ax.plot(pd.Series(out[:, feat, tf], index=timestamps), label=f"out feat={r*cols+c} tf={tf}")
                _ax.legend()
        else:
            ax[r, c].axis("off")
    plt.show()

@pytest.mark.skip("Utility Func")
def shift_forward(x, timestamps, timeframes, trunc=False):
    shifted = np.empty(x.shape)
    for f in range(x.shape[1]):
        for t, tf in enumerate(timeframes):
            _shifted = pd.Series(x[:, f, t], index=timestamps).shift(freq=-tf)
            shifted[:len(_shifted), f, t] = _shifted[timestamps]
    return shifted[~np.isnan(shifted).any(axis=2)].reshape(-1, shifted.shape[1],
                                                           shifted.shape[2])[1:] if trunc else shifted[1:]

# def test_mean():
#     window = 50
#     ref = pd.Series(arr).rolling(window, min_periods=1).mean().values
#     test = rollers.movingWindowMean(timearr, arr, window)
#     np.testing.assert_allclose(ref, test, equal_nan=True)

@pytest.mark.skip(reason="Utility Function")
def make_zones(ts):
    return rollers_cy._chronometerFX_discrete(ts).astype(np.bool)

def test_continuousX(ret=False):
    nzones = 0
    nfeats = 8

    rollerX = rollers.RollerX(tfs, nzones=nzones)
    # USE ROLLER_CY FOR SPEED BENCHMARK AND PANDAS FOR TESTING LOGIC
    # rollerX_cy = rollers_cy.RollerX_NEW(np.array(tfs))
    # out_ref_cy = time_profile(1, 1, roller_cy = lambda: rollerX_cy.roll(arr, timestamps))[1][:, :, :len(tfs)]
    print("Doing Pandas Calculations...takes time")
    def calc_pd_ref():
        ref_pd = np.empty((len(arr), nfeats, len(tfs)))
        series = pd.Series(arr, index=timestamps)
        for i, tf in enumerate(timedeltas):
            ref_pd[:, 0, i] = series.rolling(tf).apply(lambda x: x[0])
            ref_pd[:, 1, i] = series.rolling(tf).max()
            ref_pd[:, 2, i] = series.rolling(tf).min()
            ref_pd[:, 3, i] = series.rolling(tf).mean()
            ref_pd[:, 4, i] = (series/ref_pd[:, 0, i]) -1
            ref_pd[:, 5, i] = np.log((series/ref_pd[:, 0, i]) )
            ref_pd[:, 6, i] = series.rolling(tf, 1).std(ddof=1) # STD of sequential obs - type of vol
            ref_pd[:, 7, i] = np.sqrt(((series - ref_pd[:, 0, i]) **2).rolling(tf, 1).apply(lambda x: x.sum()/max(len(x)-1, 0))) # VOLATILITY
        return ref_pd
    def calc_roller_out():
        idx1, idx2 = len(arr)//3, 2* (len(arr)//3)
        _rollerX = rollers.RollerX(tfs, nzones=nzones)
        out1 = np.array(_rollerX.roll(arr[:idx1], timearr[:idx1]), copy=False)
        out2 = np.array(_rollerX.roll(arr[idx1: idx2], timearr[idx1: idx2]), copy=False)
        out3 = np.array(_rollerX.roll(arr[idx2:], timearr[idx2:]), copy=False)
        out = np.concatenate([out1, out2, out3], axis=0)
        return out
    ref = time_profile(1, 1, pd_rolls = calc_pd_ref)
    # For timing - full roll
    _out = np.array(time_profile(1, 1, rollerX_roll = lambda: rollerX.roll(arr, timearr)), copy=False)
    # For logic - partial rolls
    out = calc_roller_out()
    if ret:
        return out, ref
    np.testing.assert_allclose(out, ref, equal_nan=True)


def test_shift(ret=False):
    nzones = 0
    nfeats = 8

    rollerX = rollers.RollerX(tfs, nzones=nzones)
    rollerY = rollers.RollerY(tfs, nzones=nzones)
    xFeats = np.array(rollerX.roll(arr, timearr), dtype=np.float64, copy=False)
    def calc_roller_out():
        idx1, idx2 = len(xFeats)//3, 2*(len(xFeats)//3)
        _rollerY = rollers.RollerY(tfs, nzones=nzones)
        out1 = np.array(_rollerY.shift(xFeats[:idx1], timearr[:idx1]), copy=False)
        out2 = np.array(_rollerY.shift(xFeats[idx1: idx2], timearr[idx1: idx2]), copy=False)
        out3 = np.array(_rollerY.shift(xFeats[idx2:], timearr[idx2:]), copy=False)
        out = np.concatenate([out1, out2, out3], axis=0)
        return out
    # REF
    ref = time_profile(1, 1, pd_shifter = lambda: shift_forward(xFeats, timestamps, timedeltas, trunc=True))[:-1]
    _out = np.array(time_profile(1, 1, rollerY_shift = lambda: rollerY.shift(xFeats, timearr)), dtype=np.float64, copy=False)
    np.testing.assert_allclose(_out, ref, equal_nan=True)
    # For timing
    # For logic testing
    out = calc_roller_out()
    if ret:
        return _out#, _out
    np.testing.assert_allclose(out, ref, equal_nan=True)


@pytest.mark.skip(reason="Not Yet implemented")
def test_labels(ret=False):
    nzones = 0
    nfeats = 8

    rollerX = rollers.RollerX(tfs, nzones=nzones)
    rollerY = rollers.RollerY(tfs, nzones=nzones)
    xFeats = np.array(rollerX.roll(arr, timearr), dtype=np.float64, copy=False)
    ref_shifted = shift_forward(xFeats, timestamps, timedeltas, trunc=True)[:-1]
    yFeats= np.array(rollerY.shift(xFeats, timearr), dtype=np.float64, copy=False)

    def calc_pd_ref():
        eps = 0.
        range_mul = 2.
        ref = np.zeros((len(yFeats), 12, len(tfs)), dtype=np.bool)
        _arr = arr[:len(yFeats), None]
        _arr_closes = shift_forward((arr[:, None, None] + arr[None, None, :len(tfs)])/2, timestamps, timedeltas, trunc=True)[1:, 0, :]
        maxdiff = yFeats[:, 1, :] - _arr
        mindiff = yFeats[:, 2, :] - _arr
        hl_range = np.abs(xFeats[:len(_arr), 1, :] - xFeats[:len(_arr), 2, :])
        vol_range = xFeats[:len(_arr), 7, :]
        ref[:, 0, :] = (maxdiff+mindiff > eps)
        ref[:, 1, :] = (maxdiff+mindiff < -eps)
        ref[:, 2, :] = ~(ref[:, 0, :] | ref[:, 1, :])
        ref[:, 3, :] = maxdiff > (range_mul * hl_range)
        ref[:, 4, :] = mindiff < (-range_mul * hl_range)
        ref[:, 5, :] = ~(ref[:, 3, :] | ref[:, 4, :])
        ref[:, 6, :] = maxdiff > (range_mul * vol_range)
        ref[:, 7, :] = mindiff < (-range_mul * vol_range)
        ref[:, 8, :] = ~(ref[:, 6, :] | ref[:, 7, :])
        ref[:, 9, :] = _arr_closes > (eps + _arr)
        ref[:, 10, :] = _arr_closes < (-eps + _arr)
        ref[:, 11, :] = ~(ref[:, 9, :] | ref[:, 10, :])
        return ref

    def calc_roller_out():
        idx1, idx2 = len(xFeats)//3, 2*(len(xFeats)//3)
        _rollerY = rollers.RollerY(tfs, nzones=nzones)
        yout1 = np.array(_rollerY.shift(xFeats[:idx1], timearr[:idx1]), copy=False)
        out1 = np.array(_rollerY.roll(arr[:idx1], xFeats[:idx1], yout1, timearr[:idx1]), copy=False)
        yout2 = np.array(_rollerY.shift(xFeats[idx1: idx2], timearr[idx1: idx2]), copy=False)
        out2 = np.array(_rollerY.roll(arr[idx1: idx2], xFeats[idx1: idx2], yout2, timearr[idx1: idx2]), copy=False)
        yout3 = np.array(_rollerY.shift(xFeats[idx2:], timearr[idx2:]), copy=False)
        out3 = np.array(_rollerY.roll(arr[idx2: ], xFeats[idx2:], yout3, timearr[idx2:]), copy=False)
        out = np.concatenate([out1, out2, out3], axis=0)
        return out
    # REF
    ref = time_profile(1, 1, pd_labels = lambda: calc_pd_ref())
    # For timing
    _out = np.array(time_profile(1, 1, roller = lambda: rollerY.roll(arr, xFeats, yFeats, timearr)), dtype=np.bool, copy=False)
    # For logic testing
    if ret:
        return _out, ref
    out = calc_roller_out()
    np.testing.assert_allclose(_out, ref, equal_nan=True)
    np.testing.assert_allclose(out, ref, equal_nan=True)

def test_continuousX_sampling(ret=False):
    nzones = 0
    nfeats = 8
    sampling_tf = 0

    rollerX = rollers.RollerX(tfs, nzones=nzones)

    def calc_pd_ref():
        print("Doing RollerX Full roll with numpy comparison+indexing for (self-contained) sampling")
        rollerXF = rollers.RollerX(tfs, nzones=0)
        fullFeats = np.array(rollerXF.roll(arr, timearr), copy=False)
        idxs = np.where((arr == fullFeats[:, 1, sampling_tf]) | (arr == fullFeats[:, 2, sampling_tf]))
        return fullFeats[idxs]

    def calc_roller_out():
        idx1, idx2 = len(arr)//3, 2* (len(arr)//3)
        _rollerX = rollers.RollerX(tfs, nzones=nzones)
        out1 = np.array(_rollerX.roll(arr[:idx1], timearr[:idx1], True, "highlow", 0), copy=False)
        out2 = np.array(_rollerX.roll(arr[idx1: idx2], timearr[idx1: idx2], True, "highlow", 0), copy=False)
        out3 = np.array(_rollerX.roll(arr[idx2:], timearr[idx2:], True, "highlow", 0), copy=False)
        out = np.concatenate([out1, out2, out3], axis=0)
        return out

    ref = time_profile(1, 1, np_comparison_and_indexing= calc_pd_ref)
    # For timing - full roll
    _out = np.array(time_profile(1, 1, rollerX_roll_with_sampling = lambda: rollerX.roll(arr, timearr, True, "highlow", 0)), copy=False)
    # For logic - partial rolls
    out = calc_roller_out()
    if ret:
        return out, ref
    np.testing.assert_allclose(out, ref, equal_nan=True)



@pytest.mark.skip(reason="Not Yet Fixed/Implemented - put in Backlog for Future")
def test_zones(ret=False):
    nzones = 4
    nfeats = 8

    timedeltas = [pd.Timedelta(t) for t in ["1h"]]
    tfs = [tdelta.value for tdelta in timedeltas]
    rollerX = rollers.RollerX(tfs, nzones=nzones)
    # USE ROLLER_CY FOR SPEED BENCHMARK AND PANDAS FOR TESTING LOGIC
    # rollerX_cy = rollers_cy.RollerX_NEW(np.array(tfs))
    # out_ref_cy = time_profile(1, 1, roller_cy = lambda: rollerX_cy.roll(arr, timestamps))[1][:, :, :len(tfs)]
    print("Doing Pandas Calculations...takes time")

    def clean_zones(series, groups):
        series[groups==0] = np.nan
        return series.ffill()

    def calc_pd_ref_zones(nfeats=8, nzones=4):
        ref_pd = np.empty((len(arr), nfeats, nzones))
        for i, (zone, zone_hours) in enumerate(zip([SYDNEY, TOKYO, LONDON, NY],[SYDNEY_HOURS, TOKYO_HOURS, LONDON_HOURS, NY_HOURS])):
            series = pd.Series(arr, index=timestamps.tz_convert(zone))
            groups = np.zeros(len(series)).astype("bool")
            groups[series.index.indexer_between_time(f"{zone_hours[0]}:00", f"{zone_hours[1]}", include_end=False)] = True
            groups_ref = make_zones(timestamps)
            import ipdb; ipdb.set_trace()
            g =  np.roll(groups, -1)
            g[0] = groups[0]
            grouper = np.cumsum(groups != g)
            grouped = series.groupby(grouper)
            ref_pd[:, 0, i] = clean_zones(grouped.expanding().apply(lambda x: x[0]), groups).values
            ref_pd[:, 1, i] = clean_zones(grouped.expanding().max(), groups).values
            ref_pd[:, 2, i] = clean_zones(grouped.expanding().min(), groups).values
            ref_pd[:, 3, i] = clean_zones(grouped.expanding().mean(), groups).values
            ref_pd[:, 4, i] = clean_zones(((series/ref_pd[:, 0, i]) -1), groups).values
            ref_pd[:, 5, i] = clean_zones(np.log(series/ref_pd[:, 0, i]), groups).values
            ref_pd[:, 6, i] = clean_zones(grouped.expanding().std(ddof=1), groups).values # STD of sequential obs - type of vol
            # vols = np.sqrt(((series - ref_pd[:, 0, i]) **2).apply(lambda x: x.sum()/max(len(x)-1, 0)))
            vols = np.sqrt((series - ref_pd[:, 0, i] ** 2).groupby(grouper).expanding().apply(lambda x: x.sum()/max(len(x)-1, 0)))
            ref_pd[:, 7, i] = clean_zones(pd.Series(vols), groups) # VOLATILITY
        return ref_pd

    print("Doing Pandas zone calcs")
    ref_pd = time_profile(1, 1, pd_rolls = calc_pd_ref_zones)
    out = np.array(time_profile(1, 1, roller = lambda: rollerX.roll(arr, timearr, make_zones(timestamps))), copy=False)[:, :, 1:]

    if ret:
        return out, ref_pd
    np.testing.assert_allclose(out, ref_pd, equal_nan=True)





if __name__ == "__main__":
    # pass
    # Collect Outputs Here for Debugging
    DEBUG = False
    if not DEBUG:
        tests = [test_continuousX, test_shift,
                 test_continuousX_sampling]
        passed = 0
        failed = []
        for i, test in enumerate(tests):
            try:
                test()
                passed += 1
            except AssertionError:
                failed.append(test)
        if passed == len(tests):
            print(f"All {passed}/{len(tests)} tests passed")
        else:
            print(f"{passed}/{len(tests)} tests passed")
            print('failed: ', failed)

    else:
        outx, refx = test_continuousX(ret=True)
        out = test_shift(ret=True)
        out, ref = test_labels(ret=True)
        out, ref = test_continuousX_sampling(ret=True)
