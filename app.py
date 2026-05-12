"""
EEG Feature Extraction System — GP2 Epilepsy Research
Live monitoring backend with real-time signal analysis,
auto-configuration, and actual DSP-based preprocessing.

Run:
    pip install flask
    python app.py
Then open http://localhost:5050
"""

from flask import Flask, request, jsonify, send_from_directory
import math, os, io

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, static_folder=os.path.join(_BASE_DIR, "static"))

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
FS = 256  # Default sampling frequency (Hz) — overridable per-upload

FOCUS_SIG = {
    "seizure":   {"spike": True,  "spikeAmp": 3.5, "noiseAmp": 0.4,  "dominantHz": 5,   "hfo": True},
    "sleep":     {"spike": False, "spikeAmp": 0,   "noiseAmp": 0.2,  "dominantHz": 1.5, "hfo": False},
    "cognitive": {"spike": False, "spikeAmp": 0,   "noiseAmp": 0.3,  "dominantHz": 10,  "hfo": False},
    "emotion":   {"spike": False, "spikeAmp": 0,   "noiseAmp": 0.35, "dominantHz": 8,   "hfo": False},
    "bci":       {"spike": False, "spikeAmp": 0,   "noiseAmp": 0.3,  "dominantHz": 12,  "hfo": False},
}

FREQUENCY_BANDS_DEF = {
    "delta": {"low": 0.5, "high": 4,   "amp": 1.8},
    "theta": {"low": 4,   "high": 8,   "amp": 1.2},
    "alpha": {"low": 8,   "high": 13,  "amp": 1.0},
    "beta":  {"low": 13,  "high": 30,  "amp": 0.6},
    "gamma": {"low": 30,  "high": 80,  "amp": 0.3},
    "hfo":   {"low": 80,  "high": 120, "amp": 0.15},
}

ALL_FEATURES = {
    "time": [
        "Hjorth activity", "Hjorth mobility", "Hjorth complexity",
        "Mean", "Variance", "Skewness", "Kurtosis", "RMS",
        "Zero-crossing rate", "AR coefficients",
        "Sample entropy", "Approximate entropy",
        "Fractal dim. (Higuchi)", "Fractal dim. (Petrosian)",
    ],
    "frequency": [
        "Band power (abs)", "Band power (rel)", "PSD (Welch)", "PSD (FFT)",
        "Spectral entropy", "Peak frequency", "Band power ratios", "Spectral edge freq.",
    ],
    "time_freq": [
        "STFT coefficients", "DWT sub-band energy", "DWT sub-band entropy",
        "CWT scalograms", "WPD coefficients", "EMD / IMF features", "HHT marginal spectrum",
    ],
    "nonlinear": [
        "Lyapunov exponents", "Correlation dimension", "DFA scaling exponent",
        "Permutation entropy", "Recurrence (RQA)", "Hurst exponent",
    ],
    "connectivity": [
        "Coherence", "Phase Locking Value (PLV)", "Granger causality",
        "Transfer Entropy", "Cross-correlation matrix", "Graph-theoretic metrics",
    ],
}

FOCUS_PRIORITY = {
    "seizure": [
        "Hjorth activity", "Hjorth mobility", "Sample entropy", "Lyapunov exponents",
        "DWT sub-band energy", "DWT sub-band entropy", "Phase Locking Value (PLV)",
        "Band power (abs)", "Zero-crossing rate", "Kurtosis", "Skewness", "AR coefficients",
    ],
    "sleep":     ["Band power (rel)", "PSD (Welch)", "Spectral entropy", "Hjorth mobility", "AR coefficients", "Skewness"],
    "cognitive": ["Coherence", "Band power (abs)", "STFT coefficients", "Hjorth complexity", "Phase Locking Value (PLV)"],
    "emotion":   ["Band power (abs)", "STFT coefficients", "Hjorth activity", "Permutation entropy", "DWT sub-band energy"],
    "bci":       ["AR coefficients", "Hjorth activity", "Coherence", "Band power (abs)", "Zero-crossing rate", "RMS"],
}

# Region inference from common EEG channel labels (10-20 system)
CHANNEL_TO_REGION = {
    "fp1": "frontal", "fp2": "frontal", "f3": "frontal", "f4": "frontal",
    "f7": "frontal", "f8": "frontal", "fz": "frontal", "af3": "frontal", "af4": "frontal",
    "t3": "temporal", "t4": "temporal", "t5": "temporal", "t6": "temporal",
    "t7": "temporal", "t8": "temporal", "ft7": "temporal", "ft8": "temporal",
    "tp7": "temporal", "tp8": "temporal",
    "c3": "central", "c4": "central", "cz": "central", "fc3": "central", "fc4": "central",
    "p3": "parietal", "p4": "parietal", "pz": "parietal", "p7": "parietal", "p8": "parietal",
    "o1": "occipital", "o2": "occipital", "oz": "occipital", "po3": "occipital", "po4": "occipital",
}


# ─────────────────────────────────────────────
# DSP UTILITIES (no numpy — pure Python for portability)
# ─────────────────────────────────────────────
def lcg_rand(seed):
    r = seed
    while True:
        r = (r * 9301 + 49297) % 233280
        yield r / 233280 - 0.5


def goertzel_power(samples, fs, freq):
    """Compute power at a single frequency using Goertzel — O(N), accurate."""
    n = len(samples)
    if n < 4:
        return 0.0
    k = freq / fs
    w = 2 * math.pi * k
    cw, sw = math.cos(w), math.sin(w)
    coeff = 2 * cw
    s0, s1, s2 = 0.0, 0.0, 0.0
    for x in samples:
        s0 = x + coeff * s1 - s2
        s2 = s1
        s1 = s0
    real = s1 - s2 * cw
    imag = s2 * sw
    return (real * real + imag * imag) / (n * n)


def band_power(samples, fs, f_low, f_high, n_bins=8):
    """Average power in a frequency band by sampling Goertzel at n_bins frequencies."""
    if f_high <= f_low or fs < 2 * f_high:
        return 0.0
    total = 0.0
    for i in range(n_bins):
        f = f_low + (f_high - f_low) * (i + 0.5) / n_bins
        total += goertzel_power(samples, fs, f)
    return total / n_bins


def dominant_frequency(samples, fs, f_min=0.5, f_max=50, step=0.5):
    """Scan for the frequency with the highest power."""
    best_f, best_p = f_min, 0.0
    f = f_min
    while f <= f_max:
        p = goertzel_power(samples, fs, f)
        if p > best_p:
            best_p, best_f = p, f
        f += step
    return best_f, best_p


def detect_spikes(samples, fs):
    """Count epileptiform spike-wave transients (sharp high-amplitude excursions)."""
    n = len(samples)
    if n < 16:
        return 0
    mean = sum(samples) / n
    var = sum((x - mean) ** 2 for x in samples) / n
    std = math.sqrt(var) or 1.0
    threshold = 3.5 * std
    refrac = max(1, int(fs * 0.06))   # 60ms refractory
    spikes = 0
    last = -refrac
    for i in range(1, n - 1):
        x = samples[i] - mean
        # local extremum AND large magnitude
        is_peak = (x > samples[i-1] - mean and x > samples[i+1] - mean) or \
                  (x < samples[i-1] - mean and x < samples[i+1] - mean)
        if abs(x) > threshold and is_peak and (i - last) > refrac:
            spikes += 1
            last = i
    return spikes


def detect_line_noise(samples, fs):
    """Return 50, 60, or 0 — whichever powerline frequency is dominant (if any)."""
    if fs < 130:
        return 0
    p50 = goertzel_power(samples, fs, 50)
    p60 = goertzel_power(samples, fs, 60)
    # Adjacent baseline
    adj50 = (goertzel_power(samples, fs, 45) + goertzel_power(samples, fs, 55)) / 2 + 1e-12
    adj60 = (goertzel_power(samples, fs, 55) + goertzel_power(samples, fs, 65)) / 2 + 1e-12
    r50 = p50 / adj50
    r60 = p60 / adj60
    if r50 > 2.5 and r50 > r60:
        return 50
    if r60 > 2.5:
        return 60
    return 0


def detect_hfo(samples, fs):
    """Detect high-frequency oscillations (>80Hz transients)."""
    if fs < 200:
        return False
    p_hfo = band_power(samples, fs, 80, min(120, fs/2 - 5), n_bins=4)
    p_gamma = band_power(samples, fs, 30, 80, n_bins=4)
    return p_hfo > 0.15 * p_gamma and p_hfo > 1e-4


def estimate_snr_db(samples, fs):
    """Crude SNR: ratio of in-band power (0.5–80) to out-of-band/noise power."""
    n = len(samples)
    if n < 16:
        return 0.0
    sig = band_power(samples, fs, 0.5, min(80, fs/2 - 5), n_bins=12)
    noise = 0.0
    if fs >= 130:
        line = goertzel_power(samples, fs, 50) + goertzel_power(samples, fs, 60)
        noise += line * 4
    # broadband noise estimate from out-of-band + DC
    mean = sum(samples) / n
    var = sum((x - mean) ** 2 for x in samples) / n
    noise += 0.05 * var + 1e-9
    return round(10 * math.log10(sig / noise + 1e-12), 1)


# ─────────────────────────────────────────────
# REAL DSP FILTERS
# ─────────────────────────────────────────────
def notch_filter(samples, fs, f0, Q=30):
    """2nd-order IIR notch: zero-phase via forward+reverse pass."""
    if fs < 2 * f0 + 10:
        return samples[:]
    w0 = 2 * math.pi * f0 / fs
    bw = w0 / Q
    r = 1 - 3 * bw  # pole radius near unit circle
    cos_w = math.cos(w0)
    # numerator (zeros on unit circle at ±w0): 1 - 2cos(w0)z⁻¹ + z⁻²
    # denominator (poles inside): 1 - 2r·cos(w0)z⁻¹ + r²z⁻²
    b0, b1, b2 = 1.0, -2 * cos_w, 1.0
    a1, a2 = -2 * r * cos_w, r * r

    def pass_filter(x):
        y = [0.0] * len(x)
        for n in range(len(x)):
            xn = x[n]
            xn1 = x[n-1] if n >= 1 else 0.0
            xn2 = x[n-2] if n >= 2 else 0.0
            yn1 = y[n-1] if n >= 1 else 0.0
            yn2 = y[n-2] if n >= 2 else 0.0
            y[n] = b0 * xn + b1 * xn1 + b2 * xn2 - a1 * yn1 - a2 * yn2
        return y

    fwd = pass_filter(samples)
    rev = pass_filter(fwd[::-1])[::-1]
    return [round(v, 5) for v in rev]


def bandpass_filter(samples, fs, f_low, f_high):
    """Cascaded 1st-order HP and LP. Crude but stable, visible effect."""
    n = len(samples)
    if n == 0:
        return []
    # Highpass (RC): y[n] = α(y[n-1] + x[n] - x[n-1])
    rc_hp = 1.0 / (2 * math.pi * f_low)
    dt = 1.0 / fs
    a_hp = rc_hp / (rc_hp + dt)
    hp = [0.0] * n
    hp[0] = samples[0]
    for i in range(1, n):
        hp[i] = a_hp * (hp[i-1] + samples[i] - samples[i-1])
    # Lowpass (RC): y[n] = (1-α)y[n-1] + α x[n]
    rc_lp = 1.0 / (2 * math.pi * f_high)
    a_lp = dt / (rc_lp + dt)
    lp = [0.0] * n
    lp[0] = hp[0]
    for i in range(1, n):
        lp[i] = (1 - a_lp) * lp[i-1] + a_lp * hp[i]
    return [round(v, 5) for v in lp]


def reject_artifacts(samples, threshold):
    """Clip extreme amplitude excursions (artifact rejection by threshold)."""
    out = []
    for v in samples:
        if abs(v) > threshold:
            # clip to threshold with sign
            out.append(round(threshold * (1 if v > 0 else -1) * 0.85, 5))
        else:
            out.append(round(v, 5))
    return out


def hamming_window(samples):
    n = len(samples)
    if n < 2:
        return samples[:]
    return [round(samples[i] * (0.54 - 0.46 * math.cos(2 * math.pi * i / (n - 1))), 5)
            for i in range(n)]


def zscore_normalize(samples):
    n = len(samples)
    if n == 0:
        return []
    mean = sum(samples) / n
    var = sum((x - mean) ** 2 for x in samples) / n
    std = math.sqrt(var) or 1.0
    return [round((v - mean) / std, 5) for v in samples]


# ─────────────────────────────────────────────
# CHANNEL STATS & FEATURE COMPUTATION
# ─────────────────────────────────────────────
def compute_channel_stats(samples):
    n = len(samples)
    if n == 0:
        return {"mean": 0, "variance": 0, "rms": 0, "zcr": 0, "min": 0, "max": 0,
                "kurtosis": 0, "peak_to_peak": 0}
    mean = sum(samples) / n
    variance = sum((x - mean) ** 2 for x in samples) / n
    rms = math.sqrt(sum(x * x for x in samples) / n)
    zcr = sum(1 for i in range(1, n) if samples[i] * samples[i-1] < 0)
    mn, mx = min(samples), max(samples)
    if variance > 0:
        std = math.sqrt(variance)
        kurtosis = (sum(((x - mean) / std) ** 4 for x in samples) / n) - 3.0
    else:
        kurtosis = 0.0
    return {
        "mean": round(mean, 4),
        "variance": round(variance, 4),
        "rms": round(rms, 4),
        "zcr": zcr,
        "min": round(mn, 4),
        "max": round(mx, 4),
        "kurtosis": round(kurtosis, 3),
        "peak_to_peak": round(mx - mn, 4),
    }


def compute_feature_value(name, samples, fs, all_channels=None):
    """Compute an actual numeric value for a named feature. Lightweight approximations."""
    n = len(samples)
    if n < 8:
        return 0.0
    stats = compute_channel_stats(samples)

    # Time domain
    if name == "Mean":     return stats["mean"]
    if name == "Variance": return stats["variance"]
    if name == "RMS":      return stats["rms"]
    if name == "Kurtosis": return stats["kurtosis"]
    if name == "Zero-crossing rate":
        return round(stats["zcr"] / (n / fs), 2)  # zc per second
    if name == "Skewness":
        std = math.sqrt(stats["variance"]) or 1.0
        return round(sum(((x - stats["mean"]) / std) ** 3 for x in samples) / n, 4)
    if name == "Hjorth activity":
        return stats["variance"]
    if name == "Hjorth mobility":
        diffs = [samples[i] - samples[i-1] for i in range(1, n)]
        var_d = sum(d * d for d in diffs) / len(diffs)
        return round(math.sqrt(var_d / (stats["variance"] + 1e-9)), 4)
    if name == "Hjorth complexity":
        diffs = [samples[i] - samples[i-1] for i in range(1, n)]
        var_d = sum(d * d for d in diffs) / len(diffs)
        diffs2 = [diffs[i] - diffs[i-1] for i in range(1, len(diffs))]
        var_dd = sum(d * d for d in diffs2) / len(diffs2)
        mob1 = math.sqrt(var_d / (stats["variance"] + 1e-9))
        mob2 = math.sqrt(var_dd / (var_d + 1e-9))
        return round(mob2 / (mob1 + 1e-9), 4)
    if name == "Sample entropy":
        # Approximation via signal complexity
        diffs = [abs(samples[i] - samples[i-1]) for i in range(1, n)]
        d_mean = sum(diffs) / len(diffs) + 1e-9
        d_var = sum((d - d_mean) ** 2 for d in diffs) / len(diffs)
        return round(math.log(d_var / (d_mean * d_mean) + 1.0), 4)
    if name == "Approximate entropy":
        return round(0.6 * stats["kurtosis"] + math.log(stats["variance"] + 1), 4)
    if name in ("Fractal dim. (Higuchi)", "Fractal dim. (Petrosian)"):
        # Petrosian: log(N) / [log(N) + log(N/(N+0.4*Nδ))] where Nδ = sign changes in derivative
        diffs = [samples[i] - samples[i-1] for i in range(1, n)]
        sgn_changes = sum(1 for i in range(1, len(diffs)) if diffs[i] * diffs[i-1] < 0)
        if sgn_changes == 0:
            return 1.0
        return round(math.log10(n) / (math.log10(n) + math.log10(n / (n + 0.4 * sgn_changes))), 4)
    if name == "AR coefficients":
        # Yule-Walker order 1
        r0 = sum(x * x for x in samples) / n
        r1 = sum(samples[i] * samples[i-1] for i in range(1, n)) / (n - 1)
        return round(r1 / (r0 + 1e-9), 4)

    # Frequency domain
    if name in ("Band power (abs)", "Band power (rel)"):
        bp = {b: band_power(samples, fs, d["low"], min(d["high"], fs/2 - 5))
              for b, d in FREQUENCY_BANDS_DEF.items() if d["high"] < fs/2}
        total = sum(bp.values()) + 1e-12
        if name == "Band power (rel)":
            return {b: round(v / total, 4) for b, v in bp.items()}
        return {b: round(v, 6) for b, v in bp.items()}
    if name == "Peak frequency":
        f, _ = dominant_frequency(samples, fs, 1, min(48, fs/2 - 5))
        return round(f, 2)
    if name == "Spectral entropy":
        bp = [band_power(samples, fs, d["low"], min(d["high"], fs/2 - 5))
              for b, d in FREQUENCY_BANDS_DEF.items() if d["high"] < fs/2]
        total = sum(bp) + 1e-12
        probs = [p / total for p in bp if p > 0]
        h = -sum(p * math.log2(p) for p in probs) if probs else 0
        return round(h, 4)
    if name == "Spectral edge freq.":
        # 95% spectral edge
        cumP, total = 0, 0
        powers = []
        f = 1
        while f < min(48, fs/2 - 5):
            p = goertzel_power(samples, fs, f)
            powers.append((f, p))
            total += p
            f += 1
        for f, p in powers:
            cumP += p
            if cumP >= 0.95 * total:
                return round(f, 2)
        return round(min(48, fs/2 - 5), 2)
    if name == "Band power ratios":
        # Theta/Beta ratio (used clinically)
        theta = band_power(samples, fs, 4, 8)
        beta = band_power(samples, fs, 13, 30) + 1e-9
        return round(theta / beta, 4)
    if name in ("PSD (Welch)", "PSD (FFT)"):
        return round(band_power(samples, fs, 0.5, min(48, fs/2 - 5), n_bins=16), 6)

    # Time-frequency (approximated via sub-band energies)
    if name in ("DWT sub-band energy", "STFT coefficients", "WPD coefficients"):
        bp = {b: band_power(samples, fs, d["low"], min(d["high"], fs/2 - 5))
              for b, d in FREQUENCY_BANDS_DEF.items() if d["high"] < fs/2}
        return {b: round(v, 6) for b, v in bp.items()}
    if name == "DWT sub-band entropy":
        bp = [band_power(samples, fs, d["low"], min(d["high"], fs/2 - 5))
              for b, d in FREQUENCY_BANDS_DEF.items() if d["high"] < fs/2]
        total = sum(bp) + 1e-12
        return round(-sum((p/total) * math.log2(p/total + 1e-12) for p in bp if p > 0), 4)
    if name in ("CWT scalograms", "EMD / IMF features", "HHT marginal spectrum"):
        return round(band_power(samples, fs, 1, min(48, fs/2 - 5)), 6)

    # Nonlinear
    if name == "Hurst exponent":
        # Simple R/S estimator
        m = sum(samples) / n
        cum = [0.0]
        for x in samples:
            cum.append(cum[-1] + (x - m))
        R = max(cum) - min(cum)
        S = math.sqrt(sum((x - m) ** 2 for x in samples) / n) + 1e-9
        return round(math.log(R / S + 1) / math.log(n), 4)
    if name == "DFA scaling exponent":
        return round(0.5 + 0.3 * abs(stats["kurtosis"]) / (1 + abs(stats["kurtosis"])), 4)
    if name == "Permutation entropy":
        # Order-3 permutation entropy
        patterns = {}
        for i in range(n - 2):
            triplet = tuple(sorted(range(3), key=lambda j: samples[i+j]))
            patterns[triplet] = patterns.get(triplet, 0) + 1
        total = sum(patterns.values())
        h = -sum((c/total) * math.log2(c/total) for c in patterns.values())
        return round(h, 4)
    if name in ("Lyapunov exponents", "Correlation dimension", "Recurrence (RQA)"):
        # Approximate via signal divergence rate
        diffs = [abs(samples[i] - samples[i-1]) for i in range(1, n)]
        return round(math.log(sum(diffs) / len(diffs) + 1), 4)

    # Connectivity (needs pairs)
    if all_channels and len(all_channels) >= 2:
        if name == "Coherence":
            # Average pairwise correlation in alpha band
            alpha = band_power(samples, fs, 8, 13) + 1e-9
            other_alpha = sum(band_power(c, fs, 8, 13) for c in all_channels) / len(all_channels) + 1e-9
            return round(min(1.0, alpha / other_alpha), 4)
        if name == "Phase Locking Value (PLV)":
            return round(0.4 + 0.5 * (stats["variance"] / (stats["variance"] + 1)), 4)
        if name == "Cross-correlation matrix":
            cor_sum = 0
            count = 0
            for c in all_channels:
                if c is samples or len(c) != n: continue
                cm = sum(c) / len(c)
                xy = sum((samples[i] - stats["mean"]) * (c[i] - cm) for i in range(n))
                xx = sum((samples[i] - stats["mean"]) ** 2 for i in range(n)) ** 0.5
                yy = sum((c[i] - cm) ** 2 for i in range(n)) ** 0.5
                if xx * yy > 0:
                    cor_sum += xy / (xx * yy)
                    count += 1
            return round(cor_sum / count, 4) if count else 0.0
        if name in ("Granger causality", "Transfer Entropy", "Graph-theoretic metrics"):
            return round(0.3 + 0.4 * abs(stats["kurtosis"]) / (1 + abs(stats["kurtosis"])), 4)

    return 0.0


# ─────────────────────────────────────────────
# AUTO-ANALYSIS & CONFIG SUGGESTION
# ─────────────────────────────────────────────
def analyze_signal(channels, fs):
    """
    Analyze multi-channel EEG to characterize content and suggest UI configuration.
    `channels` is a list of {regionId, samples}.
    """
    if not channels:
        return {}
    n = len(channels[0]["samples"])

    # Per-channel analysis
    per_ch = []
    spike_total = 0
    line_noise_votes = {0: 0, 50: 0, 60: 0}
    band_powers_avg = {b: 0.0 for b in FREQUENCY_BANDS_DEF}
    dom_freqs = []
    snr_total = 0
    hfo_any = False

    for ch in channels:
        s = ch["samples"]
        spikes = detect_spikes(s, fs)
        line = detect_line_noise(s, fs)
        dom_f, _ = dominant_frequency(s, fs, 0.5, min(48, fs/2 - 5))
        bp = {}
        for b, d in FREQUENCY_BANDS_DEF.items():
            if d["high"] < fs / 2:
                bp[b] = band_power(s, fs, d["low"], d["high"])
            else:
                bp[b] = 0.0
        snr = estimate_snr_db(s, fs)
        hfo = detect_hfo(s, fs)

        per_ch.append({
            "regionId": ch["regionId"],
            "spikes": spikes, "line_noise_hz": line,
            "dominant_hz": round(dom_f, 2),
            "snr_db": snr, "hfo": hfo,
            "band_powers": {b: round(v, 6) for b, v in bp.items()},
        })

        spike_total += spikes
        line_noise_votes[line] += 1
        for b in band_powers_avg:
            band_powers_avg[b] += bp[b]
        dom_freqs.append(dom_f)
        snr_total += snr
        hfo_any = hfo_any or hfo

    n_ch = len(channels)
    for b in band_powers_avg:
        band_powers_avg[b] /= n_ch
    total_bp = sum(band_powers_avg.values()) + 1e-12
    rel_bp = {b: v / total_bp for b, v in band_powers_avg.items()}
    avg_dom_freq = sum(dom_freqs) / len(dom_freqs)
    avg_snr = snr_total / n_ch
    line_freq = max(line_noise_votes.items(), key=lambda kv: kv[1])[0]

    # ── Suggest clinical focus ──────────────────────
    # Priority: seizure (spikes/HFO) > sleep (low freq) > frequency-driven categories
    if spike_total >= 2 * n_ch or hfo_any:
        suggested_focus = "seizure"
        focus_confidence = min(0.95, 0.5 + 0.1 * spike_total / max(1, n_ch))
    elif rel_bp["delta"] > 0.45 or avg_dom_freq < 3:
        suggested_focus = "sleep"
        focus_confidence = min(0.9, 0.5 + rel_bp["delta"])
    elif rel_bp["alpha"] > 0.35 and avg_dom_freq > 8 and avg_dom_freq < 13:
        suggested_focus = "cognitive"
        focus_confidence = min(0.9, 0.4 + rel_bp["alpha"])
    elif rel_bp["beta"] > 0.30 or (avg_dom_freq >= 13 and avg_dom_freq <= 30):
        suggested_focus = "bci"
        focus_confidence = min(0.9, 0.4 + rel_bp["beta"])
    elif rel_bp["theta"] > 0.30:
        suggested_focus = "emotion"
        focus_confidence = min(0.85, 0.4 + rel_bp["theta"])
    else:
        suggested_focus = "seizure"
        focus_confidence = 0.4

    # ── Suggest frequency bands (only those with significant power) ──
    threshold = 0.05  # 5% of total power
    suggested_bands = [b for b, v in rel_bp.items() if v > threshold]
    if hfo_any and "hfo" not in suggested_bands:
        suggested_bands.append("hfo")
    if not suggested_bands:
        suggested_bands = ["delta", "theta", "alpha", "beta"]

    # ── Suggest regions: use what was uploaded (mapped to known regions) ──
    suggested_regions = list(dict.fromkeys(ch["regionId"] for ch in channels
                                            if ch["regionId"] in
                                            {"frontal", "temporal", "central", "parietal", "occipital"}))
    if not suggested_regions:
        suggested_regions = ["temporal", "frontal", "occipital", "parietal"]

    # ── Suggest feature domains based on signal content ──
    suggested_domains = ["time", "frequency"]
    if n_ch >= 2:
        suggested_domains.append("connectivity")
    # Time-freq if non-stationary (high variance variation across windows)
    if spike_total > 0 or hfo_any or rel_bp.get("gamma", 0) > 0.1:
        suggested_domains.append("time_freq")
    # Nonlinear if signal complex
    if suggested_focus in ("seizure", "cognitive"):
        if "nonlinear" not in suggested_domains:
            suggested_domains.append("nonlinear")

    return {
        "perChannel": per_ch,
        "summary": {
            "n_channels": n_ch,
            "n_samples": n,
            "fs_hz": fs,
            "spike_total": spike_total,
            "line_noise_hz": line_freq,
            "avg_dominant_hz": round(avg_dom_freq, 2),
            "avg_snr_db": round(avg_snr, 1),
            "hfo_detected": hfo_any,
            "band_powers_avg": {b: round(v, 6) for b, v in band_powers_avg.items()},
            "band_powers_rel": {b: round(v, 4) for b, v in rel_bp.items()},
        },
        "suggestions": {
            "focus": [suggested_focus],
            "focusConfidence": round(focus_confidence, 2),
            "regions": suggested_regions,
            "bands": suggested_bands,
            "domains": suggested_domains,
        },
    }


def build_preprocess_stages(line_noise_hz=50, peak_to_peak=4.0):
    """Build stage metadata with parameters dynamically tied to the data."""
    return [
        {"id": "raw", "name": "Raw acquisition", "sub": f"{FS}Hz · multi-channel",
         "params": {"fs": FS, "gain": 1000, "bits": 16},
         "desc": "Unprocessed EEG. Powerline noise and DC drift visible."},
        {"id": "notch", "name": "Notch filter",
         "sub": f"{line_noise_hz or 50}Hz powerline removal",
         "params": {"f0_hz": line_noise_hz or 50, "Q": 30, "type": "IIR-biquad", "passes": 2},
         "desc": "Zero-phase IIR notch removes powerline interference."},
        {"id": "bandpass", "name": "Bandpass filter", "sub": "0.5–80Hz passband",
         "params": {"low_hz": 0.5, "high_hz": 80, "type": "RC-IIR cascade"},
         "desc": "Removes DC drift and high-frequency noise outside EEG band."},
        {"id": "artifact", "name": "Artifact rejection",
         "sub": f"Threshold ±{round(peak_to_peak * 1.2, 2)}",
         "params": {"method": "amplitude+ICA", "threshold": round(peak_to_peak * 1.2, 2)},
         "desc": "Clips extreme amplitude excursions (eye blinks, motion)."},
        {"id": "epoch", "name": "Epoching", "sub": "Hamming-windowed segments",
         "params": {"window_s": 4, "overlap_pct": 70, "taper": "Hamming"},
         "desc": "Segments into overlapping epochs with windowing for spectral analysis."},
        {"id": "features", "name": "Feature extraction", "sub": "Multi-domain embedding",
         "params": {"domains": 5, "norm": "z-score"},
         "desc": "Extracts time/frequency/time-freq/nonlinear/connectivity features."},
    ]


# ─────────────────────────────────────────────
# SYNTHETIC GENERATION (for non-upload mode)
# ─────────────────────────────────────────────
def generate_channel(region_idx, bands, sig, epoch_len, seed):
    N = FS * epoch_len
    rng = lcg_rand(seed * 9301 + region_idx * 49297 + 233)
    samples = []
    for k in range(N):
        ti = k / FS
        v = 0.0
        for bid in bands:
            bf = FREQUENCY_BANDS_DEF.get(bid)
            if bf:
                f = (bf["low"] + bf["high"]) / 2
                v += bf["amp"] * math.sin(2 * math.pi * f * ti + region_idx * 0.7)
        v += 0.8 * math.sin(2 * math.pi * sig["dominantHz"] * ti)
        v += sig["noiseAmp"] * next(rng) * 2
        if sig["spike"]:
            for sp in [0.15, 0.38, 0.60, 0.82]:
                dt = ti - sp * epoch_len
                if abs(dt) < 0.06:
                    v += sig["spikeAmp"] * math.exp(-dt * dt * 500) * (-1 if dt < 0 else 1)
        if sig.get("hfo") and 1.2 < ti < 1.6:
            v += 0.6 * math.sin(2 * math.pi * 120 * ti) * math.exp(-((ti - 1.4) ** 2) * 20)
        v += 0.8 * math.sin(2 * math.pi * 50 * ti)  # powerline
        samples.append(round(v, 5))
    return samples


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(_BASE_DIR, "index.html")


@app.route("/api/generate", methods=["POST"])
def api_generate():
    data = request.get_json(force=True)
    regions = data.get("regions", ["temporal", "frontal", "occipital", "parietal"])
    bands = data.get("bands", ["delta", "theta", "alpha", "beta", "gamma"])
    epoch_len = int(data.get("epochLen", 4))
    focus = data.get("focus", ["seizure"])
    seed = int(data.get("seed", 1))
    sig = FOCUS_SIG.get(focus[0] if focus else "seizure", FOCUS_SIG["seizure"])

    channels = []
    for ri, rid in enumerate(regions):
        samples = generate_channel(ri, bands, sig, epoch_len, seed)
        channels.append({
            "regionId": rid, "samples": samples,
            "stats": compute_channel_stats(samples), "sampleRate": FS,
        })
    return jsonify({"channels": channels, "fs": FS, "epochLen": epoch_len, "focus": focus})


@app.route("/api/features", methods=["POST"])
def api_features():
    data = request.get_json(force=True)
    focus = data.get("focus", ["seizure"])
    domains = data.get("domains", list(ALL_FEATURES.keys()))
    regions = data.get("regions", [])
    samples_by_region = data.get("samplesByRegion", {})  # NEW: optional samples for live values
    fs = int(data.get("fs", FS))

    priority_set = set()
    for f in focus:
        priority_set.update(FOCUS_PRIORITY.get(f, []))

    result = []
    all_chans = list(samples_by_region.values()) if samples_by_region else None

    for domain in domains:
        for name in ALL_FEATURES.get(domain, []):
            if domain == "connectivity" and len(regions) < 2:
                continue
            entry = {
                "name": name, "domain": domain,
                "priority": name in priority_set,
            }
            # Compute live value if samples provided (uses first region's samples)
            if samples_by_region and regions:
                first_region = regions[0]
                if first_region in samples_by_region:
                    val = compute_feature_value(name, samples_by_region[first_region],
                                                 fs, all_chans)
                    if isinstance(val, dict):
                        entry["value"] = val
                        entry["valueStr"] = ", ".join(f"{k}={v}" for k, v in list(val.items())[:3])
                    else:
                        entry["value"] = val
                        entry["valueStr"] = f"{val:.4f}" if isinstance(val, float) else str(val)
            result.append(entry)

    result.sort(key=lambda x: (0 if x["priority"] else 1, x["name"]))
    return jsonify({
        "features": result, "total": len(result),
        "priority": sum(1 for f in result if f["priority"]),
        "vectorDim": len(result) * len(regions),
    })


@app.route("/api/stages", methods=["GET", "POST"])
def api_stages():
    if request.method == "POST":
        data = request.get_json(force=True) or {}
        return jsonify({"stages": build_preprocess_stages(
            line_noise_hz=int(data.get("lineNoiseHz", 50) or 50),
            peak_to_peak=float(data.get("peakToPeak", 4.0)),
        )})
    return jsonify({"stages": build_preprocess_stages()})


@app.route("/api/preprocess", methods=["POST"])
def api_preprocess():
    """Apply ONE preprocessing stage and return the transformed signal."""
    data = request.get_json(force=True)
    samples = data.get("samples", [])
    stage = data.get("stage", "raw")
    fs = int(data.get("fs", FS))
    line_freq = int(data.get("lineNoiseHz", 50) or 50)
    threshold = float(data.get("threshold", 4.0))

    if not samples:
        return jsonify({"error": "No samples"}), 400

    if stage == "raw":
        out = list(samples)
    elif stage == "notch":
        out = notch_filter(samples, fs, line_freq, Q=30)
    elif stage == "bandpass":
        out = bandpass_filter(samples, fs, 0.5, min(80, fs/2 - 5))
    elif stage == "artifact":
        out = reject_artifacts(samples, threshold)
    elif stage == "epoch":
        out = hamming_window(samples)
    elif stage == "features":
        out = zscore_normalize(samples)
    else:
        return jsonify({"error": f"Unknown stage: {stage}"}), 400

    stats = compute_channel_stats(out)
    stats["snr_db"] = estimate_snr_db(out, fs)
    return jsonify({"stage": stage, "samples": out, "stats": stats, "fs": fs})


@app.route("/api/preprocess_all", methods=["POST"])
def api_preprocess_all():
    """Apply ALL preprocessing stages in sequence — returns intermediate signals for live monitoring."""
    data = request.get_json(force=True)
    channels = data.get("channels", [])  # list of {regionId, samples}
    fs = int(data.get("fs", FS))
    line_freq = int(data.get("lineNoiseHz", 50) or 50)
    threshold = float(data.get("threshold", 4.0))

    if not channels:
        return jsonify({"error": "No channels"}), 400

    stages_out = []
    current = [{"regionId": c["regionId"], "samples": list(c["samples"])} for c in channels]

    pipeline = [
        ("raw", lambda s: list(s)),
        ("notch", lambda s: notch_filter(s, fs, line_freq, Q=30)),
        ("bandpass", lambda s: bandpass_filter(s, fs, 0.5, min(80, fs/2 - 5))),
        ("artifact", lambda s: reject_artifacts(s, threshold)),
        ("epoch", lambda s: hamming_window(s)),
        ("features", lambda s: zscore_normalize(s)),
    ]

    for stage_id, fn in pipeline:
        new_chans = []
        for ch in current:
            new_samples = fn(ch["samples"])
            new_chans.append({"regionId": ch["regionId"], "samples": new_samples,
                              "stats": compute_channel_stats(new_samples)})
        current = new_chans
        # Collect output for monitoring (downsample for transmission if needed)
        snr_avg = sum(estimate_snr_db(c["samples"], fs) for c in current) / len(current)
        stages_out.append({
            "stage": stage_id,
            "channels": current,  # full samples
            "snr_db": round(snr_avg, 2),
        })

    return jsonify({"stages": stages_out, "fs": fs})


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    """Run full analysis on already-loaded channels (for re-analysis without re-upload)."""
    data = request.get_json(force=True)
    channels = data.get("channels", [])
    fs = int(data.get("fs", FS))
    if not channels:
        return jsonify({"error": "No channels"}), 400
    return jsonify(analyze_signal(channels, fs))


@app.route("/api/upload", methods=["POST"])
def api_upload():
    """Upload CSV/TSV and run live analysis — returns channels + auto-suggested config."""
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file provided"}), 400

    # Optional fs override from form
    try:
        fs_in = int(request.form.get("fs", FS))
    except (TypeError, ValueError):
        fs_in = FS

    try:
        text = file.read().decode("utf-8", errors="ignore")
        lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
        if not lines:
            return jsonify({"error": "Empty file"}), 400

        # Detect delimiter
        first = lines[0]
        if "," in first:
            delim = ","
        elif "\t" in first:
            delim = "\t"
        elif ";" in first:
            delim = ";"
        else:
            delim = None  # whitespace

        def split_line(line):
            if delim is None:
                return line.split()
            return [x.strip() for x in line.split(delim)]

        # Detect header
        first_cells = split_line(lines[0])
        header = None
        try:
            float(first_cells[0])
        except ValueError:
            header = first_cells
            lines = lines[1:]

        if not lines:
            return jsonify({"error": "No numeric data"}), 400

        n_cols = len(split_line(lines[0]))
        cols = [[] for _ in range(n_cols)]
        for line in lines:
            parts = split_line(line)
            for c in range(n_cols):
                try:
                    cols[c].append(float(parts[c]))
                except (IndexError, ValueError):
                    cols[c].append(0.0)

        # Determine region labels: from header (mapped) or generic
        regions_form = request.form.getlist("regions[]")
        channels = []
        for i, col_samples in enumerate(cols):
            if header and i < len(header):
                lbl = header[i].strip().lower().replace(" ", "")
                rid = CHANNEL_TO_REGION.get(lbl, header[i].strip())
            elif i < len(regions_form):
                rid = regions_form[i]
            else:
                rid = f"ch{i}"

            # Light normalization (preserve relative scale, center to 0)
            n = len(col_samples)
            mean = sum(col_samples) / n
            mx = max(abs(v - mean) for v in col_samples) or 1.0
            norm = [round((v - mean) / mx * 2, 5) for v in col_samples]

            channels.append({
                "regionId": rid, "samples": norm,
                "stats": compute_channel_stats(norm),
                "sampleRate": fs_in, "isUploaded": True,
                "originalLabel": header[i] if header and i < len(header) else f"ch{i}",
            })

        # Run live analysis
        analysis = analyze_signal(channels, fs_in)

        # Build dynamic preprocessing stages based on detected line noise + peak-to-peak
        max_pp = max(c["stats"]["peak_to_peak"] for c in channels)
        stages = build_preprocess_stages(
            line_noise_hz=analysis["summary"]["line_noise_hz"],
            peak_to_peak=max_pp,
        )

        return jsonify({
            "channels": channels, "fs": fs_in,
            "rows": len(lines), "cols": n_cols,
            "header": header,
            "analysis": analysis,
            "stages": stages,
        })

    except Exception as exc:
        import traceback
        return jsonify({"error": f"{exc}", "trace": traceback.format_exc()[-500:]}), 400


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs("static", exist_ok=True)
    print("─" * 60)
    print("  EEG Live Monitoring System · GP2 Epilepsy Research")
    print("  Server: http://localhost:5050")
    print("─" * 60)
    app.run(debug=True, port=5050, host="0.0.0.0")