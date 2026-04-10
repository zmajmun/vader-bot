"""
Smart Money Concepts (SMC) Indicator Library
Source: github.com/joshyattridge/smart-money-concepts  v0.0.27
Bundled locally for zero-import-latency in the trading loop.
"""
from functools import wraps
import pandas as pd
import numpy as np
from pandas import DataFrame, Series
from datetime import datetime


def inputvalidator(input_="ohlc"):
    def dfcheck(func):
        @wraps(func)
        def wrap(*args, **kwargs):
            args = list(args)
            i = 0 if isinstance(args[0], pd.DataFrame) else 1
            args[i] = args[i].rename(columns={c: c.lower() for c in args[i].columns})
            inputs = {
                "o": "open",
                "h": "high",
                "l": "low",
                "c": kwargs.get("column", "close").lower(),
                "v": "volume",
            }
            if inputs["c"] != "close":
                kwargs["column"] = inputs["c"]
            for l in input_:
                if inputs[l] not in args[i].columns:
                    raise LookupError(f'Must have a dataframe column named "{inputs[l]}"')
            return func(*args, **kwargs)
        return wrap
    return dfcheck


def apply(decorator):
    def decorate(cls):
        for attr in cls.__dict__:
            if callable(getattr(cls, attr)):
                setattr(cls, attr, decorator(getattr(cls, attr)))
        return cls
    return decorate


@apply(inputvalidator(input_="ohlc"))
class smc:
    __version__ = "0.0.27"

    @classmethod
    def fvg(cls, ohlc: DataFrame, join_consecutive=False) -> Series:
        fvg = np.where(
            (
                (ohlc["high"].shift(1) < ohlc["low"].shift(-1))
                & (ohlc["close"] > ohlc["open"])
            )
            | (
                (ohlc["low"].shift(1) > ohlc["high"].shift(-1))
                & (ohlc["close"] < ohlc["open"])
            ),
            np.where(ohlc["close"] > ohlc["open"], 1, -1),
            np.nan,
        )
        top = np.where(
            ~np.isnan(fvg),
            np.where(ohlc["close"] > ohlc["open"], ohlc["low"].shift(-1), ohlc["low"].shift(1)),
            np.nan,
        )
        bottom = np.where(
            ~np.isnan(fvg),
            np.where(ohlc["close"] > ohlc["open"], ohlc["high"].shift(1), ohlc["high"].shift(-1)),
            np.nan,
        )
        if join_consecutive:
            for i in range(len(fvg) - 1):
                if fvg[i] == fvg[i + 1]:
                    top[i + 1] = max(top[i], top[i + 1])
                    bottom[i + 1] = min(bottom[i], bottom[i + 1])
                    fvg[i] = top[i] = bottom[i] = np.nan

        mitigated_index = np.zeros(len(ohlc), dtype=np.int32)
        for i in np.where(~np.isnan(fvg))[0]:
            mask = np.zeros(len(ohlc), dtype=np.bool_)
            if fvg[i] == 1:
                mask = ohlc["low"][i + 2:] <= top[i]
            elif fvg[i] == -1:
                mask = ohlc["high"][i + 2:] >= bottom[i]
            if np.any(mask):
                j = np.argmax(mask) + i + 2
                mitigated_index[i] = j
        mitigated_index = np.where(np.isnan(fvg), np.nan, mitigated_index)

        return pd.concat([
            pd.Series(fvg, name="FVG"),
            pd.Series(top, name="Top"),
            pd.Series(bottom, name="Bottom"),
            pd.Series(mitigated_index, name="MitigatedIndex"),
        ], axis=1)

    @classmethod
    def swing_highs_lows(cls, ohlc: DataFrame, swing_length: int = 50) -> Series:
        swing_length *= 2
        swing_highs_lows = np.where(
            ohlc["high"] == ohlc["high"].shift(-(swing_length // 2)).rolling(swing_length).max(),
            1,
            np.where(
                ohlc["low"] == ohlc["low"].shift(-(swing_length // 2)).rolling(swing_length).min(),
                -1,
                np.nan,
            ),
        )
        while True:
            positions = np.where(~np.isnan(swing_highs_lows))[0]
            if len(positions) < 2:
                break
            current = swing_highs_lows[positions[:-1]]
            next_ = swing_highs_lows[positions[1:]]
            highs = ohlc["high"].iloc[positions[:-1]].values
            lows = ohlc["low"].iloc[positions[:-1]].values
            next_highs = ohlc["high"].iloc[positions[1:]].values
            next_lows = ohlc["low"].iloc[positions[1:]].values
            index_to_remove = np.zeros(len(positions), dtype=bool)
            consecutive_highs = (current == 1) & (next_ == 1)
            index_to_remove[:-1] |= consecutive_highs & (highs < next_highs)
            index_to_remove[1:] |= consecutive_highs & (highs >= next_highs)
            consecutive_lows = (current == -1) & (next_ == -1)
            index_to_remove[:-1] |= consecutive_lows & (lows > next_lows)
            index_to_remove[1:] |= consecutive_lows & (lows <= next_lows)
            if not index_to_remove.any():
                break
            swing_highs_lows[positions[index_to_remove]] = np.nan

        positions = np.where(~np.isnan(swing_highs_lows))[0]
        if len(positions) > 0:
            if swing_highs_lows[positions[0]] == 1:
                swing_highs_lows[0] = -1
            if swing_highs_lows[positions[0]] == -1:
                swing_highs_lows[0] = 1
            if swing_highs_lows[positions[-1]] == -1:
                swing_highs_lows[-1] = 1
            if swing_highs_lows[positions[-1]] == 1:
                swing_highs_lows[-1] = -1

        level = np.where(
            ~np.isnan(swing_highs_lows),
            np.where(swing_highs_lows == 1, ohlc["high"], ohlc["low"]),
            np.nan,
        )
        return pd.concat([
            pd.Series(swing_highs_lows, name="HighLow"),
            pd.Series(level, name="Level"),
        ], axis=1)

    @classmethod
    def bos_choch(cls, ohlc: DataFrame, swing_highs_lows: DataFrame, close_break: bool = True) -> Series:
        swing_highs_lows = swing_highs_lows.copy()
        level_order = []
        highs_lows_order = []
        bos = np.zeros(len(ohlc), dtype=np.int32)
        choch = np.zeros(len(ohlc), dtype=np.int32)
        level = np.zeros(len(ohlc), dtype=np.float32)
        last_positions = []

        for i in range(len(swing_highs_lows["HighLow"])):
            if not np.isnan(swing_highs_lows["HighLow"][i]):
                level_order.append(swing_highs_lows["Level"][i])
                highs_lows_order.append(swing_highs_lows["HighLow"][i])
                if len(level_order) >= 4:
                    bos[last_positions[-2]] = (
                        1 if (np.all(highs_lows_order[-4:] == [-1, 1, -1, 1])
                              and np.all(level_order[-4] < level_order[-2] < level_order[-3] < level_order[-1]))
                        else 0
                    )
                    level[last_positions[-2]] = level_order[-3] if bos[last_positions[-2]] != 0 else 0

                    bos[last_positions[-2]] = (
                        -1 if (np.all(highs_lows_order[-4:] == [1, -1, 1, -1])
                               and np.all(level_order[-4] > level_order[-2] > level_order[-3] > level_order[-1]))
                        else bos[last_positions[-2]]
                    )
                    level[last_positions[-2]] = level_order[-3] if bos[last_positions[-2]] != 0 else 0

                    choch[last_positions[-2]] = (
                        1 if (np.all(highs_lows_order[-4:] == [-1, 1, -1, 1])
                              and np.all(level_order[-1] > level_order[-3] > level_order[-4] > level_order[-2]))
                        else 0
                    )
                    level[last_positions[-2]] = (
                        level_order[-3] if choch[last_positions[-2]] != 0 else level[last_positions[-2]]
                    )

                    choch[last_positions[-2]] = (
                        -1 if (np.all(highs_lows_order[-4:] == [1, -1, 1, -1])
                               and np.all(level_order[-1] < level_order[-3] < level_order[-4] < level_order[-2]))
                        else choch[last_positions[-2]]
                    )
                    level[last_positions[-2]] = (
                        level_order[-3] if choch[last_positions[-2]] != 0 else level[last_positions[-2]]
                    )

                last_positions.append(i)

        broken = np.zeros(len(ohlc), dtype=np.int32)
        for i in np.where(np.logical_or(bos != 0, choch != 0))[0]:
            mask = np.zeros(len(ohlc), dtype=np.bool_)
            if bos[i] == 1 or choch[i] == 1:
                mask = ohlc["close" if close_break else "high"][i + 2:] > level[i]
            elif bos[i] == -1 or choch[i] == -1:
                mask = ohlc["close" if close_break else "low"][i + 2:] < level[i]
            if np.any(mask):
                j = np.argmax(mask) + i + 2
                broken[i] = j
                for k in np.where(np.logical_or(bos != 0, choch != 0))[0]:
                    if k < i and broken[k] >= j:
                        bos[k] = 0
                        choch[k] = 0
                        level[k] = 0

        for i in np.where(np.logical_and(np.logical_or(bos != 0, choch != 0), broken == 0))[0]:
            bos[i] = 0
            choch[i] = 0
            level[i] = 0

        bos = np.where(bos != 0, bos, np.nan)
        choch = np.where(choch != 0, choch, np.nan)
        level = np.where(level != 0, level, np.nan)
        broken = np.where(broken != 0, broken, np.nan)

        return pd.concat([
            pd.Series(bos, name="BOS"),
            pd.Series(choch, name="CHOCH"),
            pd.Series(level, name="Level"),
            pd.Series(broken, name="BrokenIndex"),
        ], axis=1)

    @classmethod
    def ob(cls, ohlc: DataFrame, swing_highs_lows: DataFrame, close_mitigation: bool = False) -> Series:
        ohlc_len = len(ohlc)
        _open = ohlc["open"].values
        _high = ohlc["high"].values
        _low = ohlc["low"].values
        _close = ohlc["close"].values
        _volume = ohlc["volume"].values
        swing_hl = swing_highs_lows["HighLow"].values

        crossed = np.full(ohlc_len, False, dtype=bool)
        ob = np.zeros(ohlc_len, dtype=np.int32)
        top_arr = np.zeros(ohlc_len, dtype=np.float32)
        bottom_arr = np.zeros(ohlc_len, dtype=np.float32)
        obVolume = np.zeros(ohlc_len, dtype=np.float32)
        lowVolume = np.zeros(ohlc_len, dtype=np.float32)
        highVolume = np.zeros(ohlc_len, dtype=np.float32)
        percentage = np.zeros(ohlc_len, dtype=np.float32)
        mitigated_index = np.zeros(ohlc_len, dtype=np.int32)
        breaker = np.full(ohlc_len, False, dtype=bool)

        swing_high_indices = np.flatnonzero(swing_hl == 1)
        swing_low_indices = np.flatnonzero(swing_hl == -1)

        active_bullish = []
        for i in range(ohlc_len):
            for idx in active_bullish.copy():
                if breaker[idx]:
                    if _high[i] > top_arr[idx]:
                        ob[idx] = top_arr[idx] = bottom_arr[idx] = 0
                        obVolume[idx] = lowVolume[idx] = highVolume[idx] = mitigated_index[idx] = percentage[idx] = 0
                        active_bullish.remove(idx)
                else:
                    if ((not close_mitigation and _low[i] < bottom_arr[idx])
                            or (close_mitigation and min(_open[i], _close[i]) < bottom_arr[idx])):
                        breaker[idx] = True
                        mitigated_index[idx] = i - 1

            pos = np.searchsorted(swing_high_indices, i)
            last_top = swing_high_indices[pos - 1] if pos > 0 else None
            if last_top is not None and _close[i] > _high[last_top] and not crossed[last_top]:
                crossed[last_top] = True
                default_idx = i - 1
                obBtm = _high[default_idx]
                obTop = _low[default_idx]
                obIndex = default_idx
                if i - last_top > 1:
                    seg = _low[last_top + 1:i]
                    if len(seg):
                        min_val = seg.min()
                        cands = np.nonzero(seg == min_val)[0]
                        if cands.size:
                            ci = last_top + 1 + cands[-1]
                            obBtm, obTop, obIndex = _low[ci], _high[ci], ci
                ob[obIndex] = 1
                top_arr[obIndex] = obTop
                bottom_arr[obIndex] = obBtm
                v0 = _volume[i] if i >= 0 else 0
                v1 = _volume[i - 1] if i >= 1 else 0
                v2 = _volume[i - 2] if i >= 2 else 0
                obVolume[obIndex] = v0 + v1 + v2
                lowVolume[obIndex] = v2
                highVolume[obIndex] = v0 + v1
                mx = max(highVolume[obIndex], lowVolume[obIndex])
                percentage[obIndex] = (min(highVolume[obIndex], lowVolume[obIndex]) / mx * 100) if mx else 100
                active_bullish.append(obIndex)

        active_bearish = []
        for i in range(ohlc_len):
            for idx in active_bearish.copy():
                if breaker[idx]:
                    if _low[i] < bottom_arr[idx]:
                        ob[idx] = top_arr[idx] = bottom_arr[idx] = 0
                        obVolume[idx] = lowVolume[idx] = highVolume[idx] = mitigated_index[idx] = percentage[idx] = 0
                        active_bearish.remove(idx)
                else:
                    if ((not close_mitigation and _high[i] > top_arr[idx])
                            or (close_mitigation and max(_open[i], _close[i]) > top_arr[idx])):
                        breaker[idx] = True
                        mitigated_index[idx] = i

            pos = np.searchsorted(swing_low_indices, i)
            last_btm = swing_low_indices[pos - 1] if pos > 0 else None
            if last_btm is not None and _close[i] < _low[last_btm] and not crossed[last_btm]:
                crossed[last_btm] = True
                default_idx = i - 1
                obTop = _high[default_idx]
                obBtm = _low[default_idx]
                obIndex = default_idx
                if i - last_btm > 1:
                    seg = _high[last_btm + 1:i]
                    if len(seg):
                        max_val = seg.max()
                        cands = np.nonzero(seg == max_val)[0]
                        if cands.size:
                            ci = last_btm + 1 + cands[-1]
                            obTop, obBtm, obIndex = _high[ci], _low[ci], ci
                ob[obIndex] = -1
                top_arr[obIndex] = obTop
                bottom_arr[obIndex] = obBtm
                v0 = _volume[i] if i >= 0 else 0
                v1 = _volume[i - 1] if i >= 1 else 0
                v2 = _volume[i - 2] if i >= 2 else 0
                obVolume[obIndex] = v0 + v1 + v2
                lowVolume[obIndex] = v0 + v1
                highVolume[obIndex] = v2
                mx = max(highVolume[obIndex], lowVolume[obIndex])
                percentage[obIndex] = (min(highVolume[obIndex], lowVolume[obIndex]) / mx * 100) if mx else 100
                active_bearish.append(obIndex)

        ob = np.where(ob != 0, ob, np.nan)
        top_arr = np.where(~np.isnan(ob), top_arr, np.nan)
        bottom_arr = np.where(~np.isnan(ob), bottom_arr, np.nan)
        obVolume = np.where(~np.isnan(ob), obVolume, np.nan)
        mitigated_index = np.where(~np.isnan(ob), mitigated_index, np.nan)
        percentage = np.where(~np.isnan(ob), percentage, np.nan)

        return pd.concat([
            pd.Series(ob, name="OB"),
            pd.Series(top_arr, name="Top"),
            pd.Series(bottom_arr, name="Bottom"),
            pd.Series(obVolume, name="OBVolume"),
            pd.Series(mitigated_index, name="MitigatedIndex"),
            pd.Series(percentage, name="Percentage"),
        ], axis=1)

    @classmethod
    def liquidity(cls, ohlc: DataFrame, swing_highs_lows: DataFrame, range_percent: float = 0.01) -> Series:
        shl = swing_highs_lows.copy()
        n = len(ohlc)
        pip_range = (ohlc["high"].max() - ohlc["low"].min()) * range_percent
        ohlc_high = ohlc["high"].values
        ohlc_low = ohlc["low"].values
        shl_HL = shl["HighLow"].values.copy()
        shl_Level = shl["Level"].values.copy()

        liquidity = np.full(n, np.nan, dtype=np.float32)
        liquidity_level = np.full(n, np.nan, dtype=np.float32)
        liquidity_end = np.full(n, np.nan, dtype=np.float32)
        liquidity_swept = np.full(n, np.nan, dtype=np.float32)

        bull_indices = np.nonzero(shl_HL == 1)[0]
        for i in bull_indices:
            if shl_HL[i] != 1:
                continue
            high_level = shl_Level[i]
            range_low = high_level - pip_range
            range_high = high_level + pip_range
            group_levels = [high_level]
            group_end = i
            c_start = i + 1
            swept = 0
            if c_start < n:
                cond = ohlc_high[c_start:] >= range_high
                if np.any(cond):
                    swept = c_start + int(np.argmax(cond))
            for j in bull_indices:
                if j <= i:
                    continue
                if swept and j >= swept:
                    break
                if shl_HL[j] == 1 and (range_low <= shl_Level[j] <= range_high):
                    group_levels.append(shl_Level[j])
                    group_end = j
                    shl_HL[j] = 0
            if len(group_levels) > 1:
                liquidity[i] = 1
                liquidity_level[i] = sum(group_levels) / len(group_levels)
                liquidity_end[i] = group_end
                liquidity_swept[i] = swept

        bear_indices = np.nonzero(shl_HL == -1)[0]
        for i in bear_indices:
            if shl_HL[i] != -1:
                continue
            low_level = shl_Level[i]
            range_low = low_level - pip_range
            range_high = low_level + pip_range
            group_levels = [low_level]
            group_end = i
            c_start = i + 1
            swept = 0
            if c_start < n:
                cond = ohlc_low[c_start:] <= range_low
                if np.any(cond):
                    swept = c_start + int(np.argmax(cond))
            for j in bear_indices:
                if j <= i:
                    continue
                if swept and j >= swept:
                    break
                if shl_HL[j] == -1 and (range_low <= shl_Level[j] <= range_high):
                    group_levels.append(shl_Level[j])
                    group_end = j
                    shl_HL[j] = 0
            if len(group_levels) > 1:
                liquidity[i] = -1
                liquidity_level[i] = sum(group_levels) / len(group_levels)
                liquidity_end[i] = group_end
                liquidity_swept[i] = swept

        return pd.concat([
            pd.Series(liquidity, name="Liquidity"),
            pd.Series(liquidity_level, name="Level"),
            pd.Series(liquidity_end, name="End"),
            pd.Series(liquidity_swept, name="Swept"),
        ], axis=1)

    @classmethod
    def previous_high_low(cls, ohlc: DataFrame, time_frame: str = "1D") -> DataFrame:
        ohlc = ohlc.copy()
        ohlc.index = pd.to_datetime(ohlc.index)
        n = len(ohlc)
        resampled = ohlc.resample(time_frame).agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        ).dropna()
        if len(resampled) < 2:
            return pd.concat([
                pd.Series(np.full(n, np.nan, dtype=np.float32), name="PreviousHigh"),
                pd.Series(np.full(n, np.nan, dtype=np.float32), name="PreviousLow"),
                pd.Series(np.zeros(n, dtype=np.int32), name="BrokenHigh"),
                pd.Series(np.zeros(n, dtype=np.int32), name="BrokenLow"),
            ], axis=1)

        resampled_times = resampled.index.values
        resampled_highs = resampled["high"].values
        resampled_lows = resampled["low"].values
        candle_times = ohlc.index.values

        periods_before = np.searchsorted(resampled_times, candle_times, side="left")
        prev_period_idx = periods_before - 2
        valid_mask = periods_before > 1

        previous_high = np.full(n, np.nan, dtype=np.float32)
        previous_low = np.full(n, np.nan, dtype=np.float32)
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices):
            lookup_indices = prev_period_idx[valid_indices]
            previous_high[valid_indices] = resampled_highs[lookup_indices]
            previous_low[valid_indices] = resampled_lows[lookup_indices]

        group_changes = np.concatenate([[True], prev_period_idx[1:] != prev_period_idx[:-1]])
        group_id = np.cumsum(group_changes)
        df_temp = pd.DataFrame({"group": group_id, "high": ohlc["high"].values, "low": ohlc["low"].values})
        cummax_high = df_temp.groupby("group")["high"].cummax().values
        cummin_low = df_temp.groupby("group")["low"].cummin().values

        broken_high = np.where(valid_mask & (cummax_high > previous_high), 1, 0).astype(np.int32)
        broken_low = np.where(valid_mask & (cummin_low < previous_low), 1, 0).astype(np.int32)

        return pd.concat([
            pd.Series(previous_high, name="PreviousHigh"),
            pd.Series(previous_low, name="PreviousLow"),
            pd.Series(broken_high, name="BrokenHigh"),
            pd.Series(broken_low, name="BrokenLow"),
        ], axis=1)

    @classmethod
    def sessions(cls, ohlc: DataFrame, session: str, start_time: str = "",
                 end_time: str = "", time_zone: str = "UTC") -> Series:
        if session == "Custom" and (not start_time or not end_time):
            raise ValueError("Custom session requires a start and end time")
        default_sessions = {
            "Sydney": {"start": "21:00", "end": "06:00"},
            "Tokyo": {"start": "00:00", "end": "09:00"},
            "London": {"start": "07:00", "end": "16:00"},
            "New York": {"start": "13:00", "end": "22:00"},
            "Asian kill zone": {"start": "00:00", "end": "04:00"},
            "London open kill zone": {"start": "06:00", "end": "09:00"},
            "New York kill zone": {"start": "11:00", "end": "14:00"},
            "london close kill zone": {"start": "14:00", "end": "16:00"},
            "Custom": {"start": start_time, "end": end_time},
        }
        ohlc.index = pd.to_datetime(ohlc.index)
        if time_zone != "UTC":
            tz = time_zone.replace("GMT", "Etc/GMT").replace("UTC", "Etc/GMT")
            ohlc.index = ohlc.index.tz_localize(tz).tz_convert("UTC")
        s = datetime.strptime(default_sessions[session]["start"], "%H:%M")
        e = datetime.strptime(default_sessions[session]["end"], "%H:%M")

        active = np.zeros(len(ohlc), dtype=np.int32)
        high = np.zeros(len(ohlc), dtype=np.float32)
        low = np.zeros(len(ohlc), dtype=np.float32)

        for i in range(len(ohlc)):
            ct = datetime.strptime(ohlc.index[i].strftime("%H:%M"), "%H:%M")
            if (s < e and s <= ct <= e) or (s >= e and (s <= ct or ct <= e)):
                active[i] = 1
                high[i] = max(ohlc["high"].iloc[i], high[i - 1] if i > 0 else 0)
                low[i] = min(ohlc["low"].iloc[i], low[i - 1] if i > 0 and low[i - 1] != 0 else float("inf"))

        return pd.concat([
            pd.Series(active, name="Active"),
            pd.Series(high, name="High"),
            pd.Series(low, name="Low"),
        ], axis=1)

    @classmethod
    def retracements(cls, ohlc: DataFrame, swing_highs_lows: DataFrame) -> Series:
        swing_highs_lows = swing_highs_lows.copy()
        direction = np.zeros(len(ohlc), dtype=np.int32)
        current_retracement = np.zeros(len(ohlc), dtype=np.float64)
        deepest_retracement = np.zeros(len(ohlc), dtype=np.float64)
        top = bottom = 0

        for i in range(len(ohlc)):
            if swing_highs_lows["HighLow"][i] == 1:
                direction[i] = 1
                top = swing_highs_lows["Level"][i]
            elif swing_highs_lows["HighLow"][i] == -1:
                direction[i] = -1
                bottom = swing_highs_lows["Level"][i]
            else:
                direction[i] = direction[i - 1] if i > 0 else 0

            if direction[i - 1] == 1:
                d = top - bottom
                current_retracement[i] = round(100 - (((ohlc["low"].iloc[i] - bottom) / d) * 100) if d else 0, 1)
                deepest_retracement[i] = max(deepest_retracement[i - 1] if i > 0 and direction[i - 1] == 1 else 0, current_retracement[i])
            if direction[i] == -1:
                d = bottom - top
                current_retracement[i] = round(100 - ((ohlc["high"].iloc[i] - top) / d) * 100 if d else 0, 1)
                deepest_retracement[i] = max(deepest_retracement[i - 1] if i > 0 and direction[i - 1] == -1 else 0, current_retracement[i])

        current_retracement = np.roll(current_retracement, 1)
        deepest_retracement = np.roll(deepest_retracement, 1)
        direction = np.roll(direction, 1)

        remove_first_count = 0
        for i in range(len(direction)):
            if i + 1 == len(direction):
                break
            if direction[i] != direction[i + 1]:
                remove_first_count += 1
            direction[i] = current_retracement[i] = deepest_retracement[i] = 0
            if remove_first_count == 3:
                direction[i + 1] = current_retracement[i + 1] = deepest_retracement[i + 1] = 0
                break

        return pd.concat([
            pd.Series(direction, name="Direction"),
            pd.Series(current_retracement, name="CurrentRetracement%"),
            pd.Series(deepest_retracement, name="DeepestRetracement%"),
        ], axis=1)
