"""Iterative clean-up loop requested by the user:

    y0 = mix
    repeat:  y = MARK6.5(y)            -- the whole signal, every pass
             y = dictionary refine(y)  -- NMF finds where noise remains, removes it
    until the dictionary estimates noise < `stop_db` below the voice,
          or the last pass improved that by < `min_gain_db`,
          or `max_iter` passes.
    optional: DSRA voice boost at the end.
"""
import numpy as np
import torch
from .losses import stft, istft
from .dsra_level import level_voice


def enhance_loop(mix, model, cfg, refiner=None, max_iter=3, stop_db=-30.0,
                 min_gain_db=0.5, dsra_boost_db=None):
    y = mix.astype(np.float32)
    n = len(y)
    trace = []
    prev = None
    for k in range(max_iter):
        with torch.no_grad():
            Y = model(stft(torch.from_numpy(y).unsqueeze(0), cfg))["est"][0]
        nvr = None
        if refiner is not None:
            Y, nvr, _ = refiner(Y)
        y = istft(Y.unsqueeze(0), cfg, length=n)[0].numpy()
        trace.append(nvr)
        if nvr is None:
            continue
        if nvr < stop_db:
            break
        if prev is not None and prev - nvr < min_gain_db:
            break
        prev = nvr
    if dsra_boost_db:
        y = level_voice(y, max_boost_db=dsra_boost_db)[0]
    return y.astype(np.float32), trace
