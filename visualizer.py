#!/usr/bin/env python3
"""
VISUALIZER — audio file in, music-video out.

Requirements:  pip install -r requirements.txt     +  ffmpeg on your PATH

Renderers (--renderer):
  auto (default)  OpenGL if it works on your machine, otherwise PIL
  gl              Full GPU renderer (moderngl): anti-aliased SDF lines/discs, trail buffer, bloom, tone-mapped
                  compositing and dithering all in shaders. Needs OpenGL 3.3+ (any NVIDIA/AMD/Intel GPU).
  pil             CPU drawing (+ optional PyTorch GPU post-processing via --gpu)

Examples:
  python visualizer.py song.mp3                              # default (radial / neon)
  python visualizer.py song.mp3 -p tunnel --palette fire
  python visualizer.py song.wav -p mix --palette vaporwave --res 1080p --fps 60
  python visualizer.py song.flac -p bars --colors "#ff0055,#ffaa00,#fff200"
  python visualizer.py song.mp3 -p radial --preview 42       # one PNG frame at 42s (fast tweaking)
  python visualizer.py --list

What's under the hood
  * Multi-resolution STFT (8192-pt FFT for bass, 2048-pt for the rest, Hann windows)
  * Log-spaced triangular filterbank (default 64 bands, 30 Hz - 16 kHz)
  * +4.5 dB/octave spectral tilt compensation so highs aren't invisible
  * Global percentile dynamic-range normalization + gamma contrast curve
  * Asymmetric attack/release envelope follower (fast attack, tunable decay)
  * Spectral-flux onset detector with adaptive threshold + refractory period
  * Spectral centroid -> slow palette hue drift
  * Zero-crossing triggered oscilloscope windows, mid/side Lissajous
  * Two-scale bloom, additive compositing, motion trails, chromatic aberration on beats
"""
import argparse
import glob
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import queue
import threading
import time
from collections import deque
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter

# ----------------------------------------------------------------------------
# Palettes
# ----------------------------------------------------------------------------
PALETTES = {
    "neon":      ["#ff00cc", "#7a00ff", "#00d4ff", "#00ffa3"],
    "sunset":    ["#ff512f", "#ff9a1f", "#ff2a68", "#8a1c9e"],
    "ocean":     ["#0a3cff", "#0575e6", "#00f2fe", "#a1ffce"],
    "fire":      ["#b30000", "#ff2a00", "#ff9100", "#ffe066"],
    "matrix":    ["#008f11", "#00ff41", "#b6ffb6"],
    "ice":       ["#4cc9f0", "#a0e7ff", "#ffffff", "#b8c0ff"],
    "vaporwave": ["#ff71ce", "#b967ff", "#01cdfe", "#05ffa1", "#fffb96"],
    "gold":      ["#b37400", "#ffb703", "#fff1b8"],
    "toxic":     ["#39ff14", "#ccff00", "#00ffc8"],
    "blood":     ["#4a0000", "#ff0022", "#ff6a6a", "#ffffff"],
    "mono":      ["#7a7a7a", "#ffffff"],
    "rainbow":   None,  # generated below
    "aurora":       ["#061b33", "#087e8b", "#20bf9f", "#b2f7ef", "#d8f3dc"],
    "synthwave":    ["#12002f", "#5b00a8", "#d500f9", "#ff4fd8", "#ffb86b"],
    "cyberpunk":    ["#080018", "#5c00a4", "#ff2a6d", "#00e5ff", "#faff00"],
    "lavender":     ["#30204f", "#8064a2", "#c8a2c8", "#e6cfff", "#fff0fa"],
    "cherry":       ["#310012", "#8f0037", "#e6005c", "#ff668a", "#ffd1dc"],
    "ember":        ["#180b08", "#762b16", "#d45b24", "#ff9f1c", "#ffe6a7"],
    "arctic":       ["#061826", "#126782", "#4cc9f0", "#a9def9", "#f1faee"],
    "forest":       ["#071b12", "#14532d", "#238636", "#84cc16", "#e0f2a8"],
    "cotton_candy": ["#4b315f", "#f7aef8", "#b8c0ff", "#9bf6ff", "#fdffb6"],
    "ultraviolet":  ["#10002b", "#3c096c", "#7b2cbf", "#c77dff", "#e0aaff"],
    "copper":       ["#24130b", "#8c4a2f", "#c87941", "#e6a15c", "#ffe0b5"],
    "deep_space":   ["#020617", "#111b54", "#312e81", "#7c3aed", "#22d3ee"],
    "sakura":       ["#3b1026", "#a53860", "#e56b8a", "#ffb3c6", "#fff0f3"],
    "limewire":     ["#071a05", "#2b8a0b", "#80d000", "#d6ff00", "#f4ffb8"],
    "inferno":      ["#140000", "#710000", "#e00000", "#ff6b00", "#ffe600"],
    "teal_gold":    ["#062b2b", "#087e8b", "#18a999", "#d4a72c", "#fff0b3"],
    "pastel":       ["#ffadad", "#ffd6a5", "#fdffb6", "#caffbf", "#9bf6ff", "#bdb2ff"],
}


def hex_to_rgb(h):
    """Accept #RGB/#RRGGBB hex or an RGB triplet such as '255, 80, 20'."""
    value = str(h).strip()
    if "," in value:
        parts = [int(x.strip()) for x in value.split(",")]
        if len(parts) != 3 or any(x < 0 or x > 255 for x in parts):
            raise ValueError("RGB must contain three values from 0 to 255")
        return tuple(parts)
    value = value.lstrip("#")
    if len(value) == 3:
        value = "".join(c * 2 for c in value)
    if len(value) != 6:
        raise ValueError("Use #RGB, #RRGGBB, or R,G,B")
    return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))


def _rainbow():
    import colorsys
    return ["#%02x%02x%02x" % tuple(int(v * 255) for v in colorsys.hsv_to_rgb(i / 7, 0.85, 1.0))
            for i in range(8)]


PALETTES["rainbow"] = _rainbow()


class Palette:
    def __init__(self, hex_stops):
        arr = np.array([hex_to_rgb(h) for h in hex_stops], float)
        xs = np.linspace(0, 1, len(arr))
        t = np.linspace(0, 1, 256)
        self.lut = np.stack([np.interp(t, xs, arr[:, c]) for c in range(3)], 1)

    def __call__(self, x, b=1.0):
        i = int(min(max(x, 0.0), 1.0) * 255)
        c = self.lut[i] * min(max(b, 0.0), 1.4)
        return (int(min(c[0], 255)), int(min(c[1], 255)), int(min(c[2], 255)))

    def cyc(self, x, b=1.0):
        """Cyclic (triangle-wave) lookup so any x maps smoothly."""
        return self(1.0 - abs(2.0 * (x % 1.0) - 1.0), b)

    def arr(self, x, b=1.0):
        """Vectorized lookup -> (N,3) floats in 0..1.4 (for the OpenGL renderer)."""
        x = np.clip(np.asarray(x, float), 0, 1)
        b = np.broadcast_to(np.clip(np.asarray(b, float), 0, 1.4), x.shape)
        return self.lut[(x * 255).astype(int)] / 255.0 * b[..., None]

    def cyc_arr(self, x, b=1.0):
        return self.arr(1.0 - np.abs(2.0 * (np.asarray(x, float) % 1.0) - 1.0), b)

    def rgb(self, x, b=1.0):
        return np.array(self(x, b), float) / 255.0


# ----------------------------------------------------------------------------
# Audio loading + analysis
# ----------------------------------------------------------------------------
FFMPEG_BIN = "ffmpeg"


def load_audio(path, sr, start, dur):
    cmd = [FFMPEG_BIN, "-v", "error"]
    if start:
        cmd += ["-ss", str(start)]
    cmd += ["-i", path]
    if dur:
        cmd += ["-t", str(dur)]
    cmd += ["-vn", "-ac", "2", "-ar", str(sr), "-f", "f32le", "-"]
    p = subprocess.run(cmd, capture_output=True)
    if p.returncode != 0 or not p.stdout:
        sys.exit("ffmpeg could not decode that file:\n" + p.stderr.decode(errors="ignore"))
    return np.frombuffer(p.stdout, dtype=np.float32).reshape(-1, 2)


def make_filterbank(freqs, edges):
    nb = len(edges) - 2
    df = freqs[1] - freqs[0]
    fb = np.zeros((len(freqs), nb), np.float32)
    for k in range(nb):
        lo, c, hi = edges[k], edges[k + 1], edges[k + 2]
        lo = min(lo, c - 1.5 * df)   # keep bass filters at least a few bins wide
        hi = max(hi, c + 1.5 * df)
        up = (freqs - lo) / (c - lo)
        dn = (hi - freqs) / (hi - c)
        fb[:, k] = np.clip(np.minimum(up, dn), 0, None)
    fb /= fb.sum(0, keepdims=True) + 1e-12
    return fb


class Analysis:
    PAD = 8192

    def __init__(self, stereo, sr, fps, nb=64, smooth=0.5, fmin=30.0, fmax=16000.0):
        self.sr, self.fps, self.nb = sr, fps, nb
        self.nf = max(1, int(len(stereo) / sr * fps))
        dt = 1.0 / fps
        mono = stereo.mean(1)

        # waveform scaling + padded stereo for scope windows
        self.wscale = float(min(50.0, 1.0 / (np.percentile(np.abs(mono), 99.5) + 1e-6)))
        self.st = np.pad(stereo, ((2048, 2048 + sr), (0, 0)))

        # --- multi-resolution STFT -> log filterbank
        fmax = min(fmax, sr / 2 * 0.95)
        edges = np.geomspace(fmin, fmax, nb + 2)
        centers = edges[1:-1]
        self.centers = centers
        use_long = centers < 300.0
        padded = np.pad(mono, (self.PAD, self.PAD + self.PAD))
        frame_centers = (np.arange(self.nf) * sr / fps).astype(np.int64)
        spec = np.zeros((self.nf, nb), np.float32)
        for N, mask in ((8192, use_long), (2048, ~use_long)):
            win = np.hanning(N).astype(np.float32)
            fb = make_filterbank(np.fft.rfftfreq(N, 1.0 / sr), edges)
            norm = float(win.sum()) ** 2
            B = 96
            for i in range(0, self.nf, B):
                cs = frame_centers[i:i + B]
                idx = (cs[:, None] + self.PAD - N // 2) + np.arange(N)[None, :]
                seg = padded[idx] * win
                p = (np.abs(np.fft.rfft(seg, axis=1)) ** 2) / norm
                spec[i:i + B][:, mask] = (p @ fb)[:, mask]

        # --- dB, tilt compensation, global normalization
        db = 10 * np.log10(spec + 1e-12) + 4.5 * np.log2(centers / 1000.0)[None, :]
        lo, hi = np.percentile(db, [3, 99.5])
        lo = max(lo, hi - 60)
        x = np.clip((db - lo) / (hi - lo + 1e-9), 0, 1) ** 1.4
        p = np.pad(x, ((0, 0), (1, 1)), mode="edge")
        x = 0.25 * p[:, :-2] + 0.5 * p[:, 1:-1] + 0.25 * p[:, 2:]
        raw = x

        # --- asymmetric envelope follower
        a_att = 1 - math.exp(-dt / 0.025)
        a_rel = 1 - math.exp(-dt / (0.05 + 0.3 * smooth))
        sm = np.empty_like(raw)
        y = np.zeros(nb, np.float32)
        for i in range(self.nf):
            k = np.where(raw[i] > y, a_att, a_rel)
            y = y + (raw[i] - y) * k
            sm[i] = y
        self.bands = sm

        def band_stat(mask):
            v = sm[:, mask].mean(1)
            return np.clip(v / (np.percentile(v, 99) + 1e-9), 0, 1)

        self.bass = band_stat(centers < 150)
        self.mid = band_stat((centers >= 150) & (centers < 2000))
        self.treble = band_stat(centers >= 2000)
        self.energy = band_stat(centers > 0)

        # --- spectral-flux onset detection
        d = np.diff(raw, axis=0, prepend=raw[:1])
        flux = np.maximum(d, 0).sum(1)
        flux = flux / (np.percentile(flux, 98) + 1e-9)
        w = max(3, int(0.6 * fps))
        ma = np.convolve(flux, np.ones(w) / w, mode="same")
        thr = ma * 1.35 + 0.12
        self.beat = np.zeros(self.nf, bool)
        self.strength = np.zeros(self.nf, np.float32)
        last = -999
        for i in range(1, self.nf - 1):
            if (flux[i] > thr[i] and flux[i] >= flux[i - 1] and flux[i] >= flux[i + 1]
                    and (i - last) / fps > 0.14):
                self.beat[i] = True
                self.strength[i] = min(1.0, max(0.3, flux[i] / 1.6))
                last = i
        self.pulse = np.zeros(self.nf, np.float32)
        pv, decay = 0.0, math.exp(-dt / 0.16)
        for i in range(self.nf):
            pv = max(pv * decay, self.strength[i] if self.beat[i] else 0.0)
            self.pulse[i] = pv

        # --- spectral centroid -> slow hue drift
        logc = np.log2(centers / centers[0]) / np.log2(centers[-1] / centers[0])
        cent = (sm * logc).sum(1) / (sm.sum(1) + 1e-9)
        ema, a = np.empty_like(cent), 1 - math.exp(-dt / 1.0)
        v = cent[0]
        for i in range(self.nf):
            v += (cent[i] - v) * a
            ema[i] = v
        self.hue = ((ema - ema.mean()) * 1.6).astype(np.float32)

        # --- integrated motion phases
        self.rot = np.cumsum((0.12 + 0.5 * self.mid + 0.4 * self.pulse) * dt)
        self.tphase = np.cumsum((0.3 + 2.0 * self.bass + 1.2 * self.pulse) * dt)

    @classmethod
    def from_cache(cls, stereo, sr, fps, cached):
        """Rebuild an Analysis instance from cached derived arrays and current audio samples."""
        self = cls.__new__(cls)
        self.sr, self.fps = sr, fps
        self.nb = int(cached["nb"])
        self.nf = int(cached["nf"])
        self.wscale = float(cached["wscale"])
        self.st = np.pad(stereo, ((2048, 2048 + sr), (0, 0)))
        for name in ("bands", "bass", "mid", "treble", "energy", "beat", "strength",
                     "pulse", "hue", "rot", "tphase"):
            setattr(self, name, cached[name])
        return self

    def frame(self, i):
        i = min(max(i, 0), self.nf - 1)
        c = int(i * self.sr / self.fps) + 2048
        seg = self.st[c - 1024:c + 1025] * self.wscale
        m = seg.mean(1)
        cond = (m[:-1] < 0) & (m[1:] >= 0)
        cond[1024:] = False
        st = int(np.argmax(cond)) if cond.any() else 512
        return SimpleNamespace(
            i=i, t=i / self.fps, bands=self.bands[i],
            bass=float(self.bass[i]), mid=float(self.mid[i]),
            treble=float(self.treble[i]), energy=float(self.energy[i]),
            pulse=float(self.pulse[i]), beat=bool(self.beat[i]),
            strength=float(self.strength[i]), hue=float(self.hue[i]),
            rot=float(self.rot[i]), tphase=float(self.tphase[i]),
            wl=seg[st:st + 1024, 0], wr=seg[st:st + 1024, 1],
        )


# ----------------------------------------------------------------------------
# Persistent audio-analysis cache
# ----------------------------------------------------------------------------
ANALYSIS_CACHE_VERSION = 1
ANALYSIS_CACHE_ARRAYS = ("bands", "bass", "mid", "treble", "energy", "beat",
                         "strength", "pulse", "hue", "rot", "tphase")


def analysis_cache_path(audio_path, sr, fps, nb, smooth, start, duration):
    """Build a stable cache key from source identity and every analysis-affecting setting."""
    source = os.path.realpath(audio_path)
    stat = os.stat(source)
    digest = hashlib.sha256()
    with open(source, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    identity = {
        "version": ANALYSIS_CACHE_VERSION,
        "path": os.path.normcase(source),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": digest.hexdigest(),
        "sr": int(sr), "fps": int(fps), "bands": int(nb),
        "smooth": float(smooth), "start": float(start), "duration": float(duration),
    }
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()
    appdata = os.environ.get("APPDATA") or os.path.join(os.path.expanduser("~"), "AppData", "Roaming")
    folder = os.path.join(appdata, "NOVVisualizer", "cache", "analysis")
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, key + ".npz")


def load_analysis_cache(path, stereo, sr, fps, nb):
    """Load a validated cache entry; return None for missing, stale, or damaged data."""
    if not os.path.isfile(path):
        return None
    try:
        with np.load(path, allow_pickle=False) as z:
            if int(z["version"]) != ANALYSIS_CACHE_VERSION:
                return None
            nf = int(z["nf"])
            cached_nb = int(z["nb"])
            if cached_nb != nb or nf != max(1, int(len(stereo) / sr * fps)):
                return None
            data = {name: z[name].copy() for name in ANALYSIS_CACHE_ARRAYS}
            if any(len(value) != nf for value in data.values()):
                return None
            data.update(nf=nf, nb=cached_nb, wscale=float(z["wscale"]))
        return Analysis.from_cache(stereo, sr, fps, data)
    except Exception as exc:
        print(f"Analysis cache unreadable ({exc}); rebuilding it.")
        try:
            os.remove(path)
        except OSError:
            pass
        return None


def save_analysis_cache(path, analysis):
    """Write compressed analysis arrays atomically so interrupted renders cannot corrupt the cache."""
    temp = path + ".tmp"
    try:
        with open(temp, "wb") as f:
            np.savez_compressed(
                f, version=np.int32(ANALYSIS_CACHE_VERSION), nf=np.int64(analysis.nf),
                nb=np.int32(analysis.nb), wscale=np.float64(analysis.wscale),
                **{name: getattr(analysis, name) for name in ANALYSIS_CACHE_ARRAYS}
            )
        os.replace(temp, path)
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"Saved audio analysis cache ({size_mb:.1f} MB): {path}")
    except Exception as exc:
        print(f"Could not save analysis cache: {exc}")
        try:
            if os.path.exists(temp):
                os.remove(temp)
        except OSError:
            pass


# ----------------------------------------------------------------------------
# Presets
# ----------------------------------------------------------------------------
def resample(a, n):
    return np.interp(np.linspace(0, len(a) - 1, n), np.arange(len(a)), a)


def spec_mirror(bands, p):
    """Sample the spectrum at normalized position p in [0,1] (0 = bass)."""
    return np.interp(np.asarray(p) * (len(bands) - 1), np.arange(len(bands)), bands)


class Preset:
    name, trail, desc = "", 0.0, ""

    def __init__(self, c):
        self.c = c


class Bars(Preset):
    name, trail, desc = "bars", 0.0, "Classic spectrum bars with falling peak caps and reflection"

    def __init__(self, c):
        super().__init__(c)
        self.n = 50
        self.peak = np.zeros(self.n)

    def draw(self, d, f):
        c = self.c
        W, H = c.W, c.H
        h = resample(f.bands, self.n)
        self.peak = np.maximum(h, self.peak - 0.9 / c.fps)
        m = W * 0.08
        bw = (W - 2 * m) / self.n
        gap = bw * 0.28
        base = H * 0.66
        maxh = H * 0.52 * (1 + 0.06 * f.pulse)
        for i in range(self.n):
            x0, x1 = m + i * bw + gap / 2, m + (i + 1) * bw - gap / 2
            u = i / (self.n - 1)
            hh = max(h[i] * maxh, 2 * c.s)
            d.rectangle([x0, base - hh, x1, base], fill=c.pal(u, 0.55 + 0.45 * h[i]))
            d.rectangle([x0, base + gap, x1, base + gap + hh * 0.35], fill=c.pal(u, 0.18))
            py = base - self.peak[i] * maxh - 6 * c.s
            d.rectangle([x0, py - 3 * c.s, x1, py], fill=c.pal(u, 1.0))


class Mirror(Preset):
    name, trail, desc = "mirror", 0.0, "Symmetric mirrored spectrum, bass in the center"

    def draw(self, d, f):
        c = self.c
        W, H = c.W, c.H
        step = max(2, int(3 * c.s))
        xs = np.arange(0, W + 1, step)
        p = np.abs(2 * xs / W - 1)
        v = spec_mirror(f.bands, p)
        hh = v * H * 0.40 * (1 + 0.1 * f.pulse) + 2 * c.s
        for x, pp, h, vv in zip(xs, p, hh, v):
            d.line([(x, c.cy - h), (x, c.cy + h)], fill=c.pal.cyc(pp * 0.9 + f.hue * 0.5, 0.45 + 0.55 * vv), width=step)
            d.line([(x, c.cy - h * 0.3), (x, c.cy + h * 0.3)], fill=(255, 255, 255), width=step)
        d.line([(0, c.cy), (W, c.cy)], fill=c.pal(0.5, 0.5), width=max(1, int(c.s)))


class Radial(Preset):
    name, trail, desc = "radial", 0.82, "Circular spectrum with waveform ring, beat-pulsing core and trails"
    n = 140

    def draw(self, d, f):
        c = self.c
        base = min(c.W, c.H)
        R = base * 0.20
        r0 = R * (1 + 0.10 * f.pulse + 0.12 * f.bass)
        j = np.arange(self.n)
        p = np.abs(2 * j / self.n - 1)
        v = spec_mirror(f.bands, p)
        ang = j / self.n * 2 * np.pi + f.rot
        ln = v * base * 0.30 + 3 * c.s
        cs, sn = np.cos(ang), np.sin(ang)
        wd = max(2, int(2 * np.pi * r0 / self.n * 0.55))
        for k in range(self.n):
            col = c.pal.cyc(p[k] * 0.9 + f.hue * 0.6, 0.45 + 0.55 * v[k])
            d.line([(c.cx + cs[k] * r0, c.cy + sn[k] * r0),
                    (c.cx + cs[k] * (r0 + ln[k]), c.cy + sn[k] * (r0 + ln[k]))], fill=col, width=wd)
        w = resample(f.wl, 360)
        w = np.convolve(np.pad(w, 4, mode="wrap"), np.ones(9) / 9, "valid")
        a = np.linspace(0, 2 * np.pi, 361)
        rr = r0 * 0.80 * (1 + 0.35 * np.append(w, w[0]))
        pts = [(c.cx + math.cos(a[k] + f.rot) * rr[k], c.cy + math.sin(a[k] + f.rot) * rr[k]) for k in range(361)]
        d.line(pts, fill=c.pal.cyc(f.hue + 0.5), width=max(2, int(2.5 * c.s)), joint="curve")
        d.ellipse([c.cx - r0, c.cy - r0, c.cx + r0, c.cy + r0],
                  outline=c.pal(0.2, 0.6 + 0.4 * f.pulse), width=max(2, int(3 * c.s)))


class Terrain(Preset):
    name, trail, desc = "terrain", 0.0, "Joy-Division-style scrolling spectrum mountains in perspective"
    K, M = 30, 120

    def __init__(self, c):
        super().__init__(c)
        self.rows = deque([np.zeros(self.M) for _ in range(self.K)], maxlen=self.K)
        self.tick = 0

    def draw(self, d, f):
        c = self.c
        W, H = c.W, c.H
        every = max(1, round(c.fps / 10))
        if self.tick % every == 0:
            u = np.linspace(0, 1, self.M)
            self.rows.appendleft(spec_mirror(f.bands, np.abs(2 * u - 1)))
        self.tick += 1
        u = np.linspace(0, 1, self.M)
        win = np.sin(np.pi * u) ** 0.9
        for k in range(self.K - 1, -1, -1):
            z = k / (self.K - 1)
            y0 = H * (0.88 - 0.50 * z)
            span = W * (0.94 - 0.44 * z)
            amp = H * 0.30 * (1 - 0.55 * z)
            xs = c.cx + (u - 0.5) * span
            ys = y0 - self.rows[k] * amp * win
            pts = list(zip(xs, ys))
            d.polygon(pts + [(xs[-1], H), (xs[0], H)], fill=(0, 0, 0))
            d.line(pts, fill=c.pal(1 - z, 1.0 - 0.65 * z), width=max(1, int((2.4 - 1.5 * z) * c.s)))


class Particles(Preset):
    name, trail, desc = "particles", 0.86, "Beat-triggered particle bursts with swirl, around a pulsing core"
    MAX = 900

    def __init__(self, c):
        super().__init__(c)
        self.pos = np.zeros((self.MAX, 2))
        self.vel = np.zeros((self.MAX, 2))
        self.life = np.zeros(self.MAX)
        self.mlife = np.ones(self.MAX)
        self.hue = np.zeros(self.MAX)
        self.head = 0

    def spawn(self, n, lo, hi, r0):
        if n <= 0:
            return
        c = self.c
        idx = (self.head + np.arange(n)) % self.MAX
        self.head = (self.head + n) % self.MAX
        a = c.rng.uniform(0, 2 * np.pi, n)
        sp = c.rng.uniform(lo, hi, n)
        self.pos[idx, 0] = c.cx + np.cos(a) * r0
        self.pos[idx, 1] = c.cy + np.sin(a) * r0
        self.vel[idx, 0] = np.cos(a) * sp
        self.vel[idx, 1] = np.sin(a) * sp
        self.life[idx] = self.mlife[idx] = c.rng.uniform(1.0, 2.4, n)
        self.hue[idx] = c.rng.uniform(0, 1, n)

    def step(self, f):
        c = self.c
        dt = 1.0 / c.fps
        r0 = min(c.W, c.H) * 0.07 * (1 + 0.35 * f.pulse + 0.3 * f.bass)
        if f.beat:
            self.spawn(int(25 + 110 * f.strength), 120 * c.s, 520 * c.s * (0.5 + f.strength), r0)
        self.spawn(int(f.treble * 5), 20 * c.s, 120 * c.s, r0)
        alive = (self.life > 0)[:, None]
        rel = self.pos - np.array([c.cx, c.cy])
        perp = np.stack([-rel[:, 1], rel[:, 0]], 1)
        self.vel += perp * (0.2 + 0.5 * f.mid) * dt * alive
        self.vel *= 0.985 ** (30 * dt)
        self.pos += self.vel * dt * alive
        self.life -= dt
        return r0

    def draw(self, d, f):
        c = self.c
        r0 = self.step(f)
        for i in np.nonzero(self.life > 0)[0]:
            lf = self.life[i] / self.mlife[i]
            r = (1.2 + 3.2 * lf) * c.s * (1 + 0.6 * f.energy)
            x, y = self.pos[i]
            d.ellipse([x - r, y - r, x + r, y + r], fill=c.pal.cyc(self.hue[i] * 0.7 + f.hue, 0.25 + 0.75 * lf))
        d.ellipse([c.cx - r0, c.cy - r0, c.cx + r0, c.cy + r0], fill=c.pal(0.3, 0.25 + 0.4 * f.pulse),
                  outline=c.pal.cyc(f.hue + 0.5), width=max(2, int(3 * c.s)))


class Tunnel(Preset):
    name, trail, desc = "tunnel", 0.5, "Infinite polygon tunnel; speed follows bass, edges wobble with the spectrum"
    K = 20

    def draw(self, d, f):
        c = self.c
        diag = math.hypot(c.W, c.H) / 2
        sides = c.sides
        npts = sides * 12
        th = np.linspace(0, 2 * np.pi, npts, endpoint=False)
        poly = np.cos(np.pi / sides) / np.cos((th % (2 * np.pi / sides)) - np.pi / sides)
        v = spec_mirror(f.bands, np.abs(2 * th / (2 * np.pi) - 1))
        for k in range(self.K):
            dp = ((k + f.tphase) % self.K) / self.K
            r = diag * 0.04 + diag * 1.1 * dp ** 2.4
            ang = th + f.rot * 0.5 + dp * 0.9 * math.sin(f.t * 0.2)
            rr = r * poly * (1 + 0.5 * v * dp ** 0.6)
            pts = [(c.cx + math.cos(a) * q, c.cy + math.sin(a) * q) for a, q in zip(ang, rr)]
            pts.append(pts[0])
            fade = min(1.0, dp * 6) * (0.35 + 0.65 * dp ** 0.5)
            col = c.pal.cyc(k / self.K + f.hue * 0.5 + f.t * 0.03, fade)
            d.line(pts, fill=col, width=max(1, int((1 + 5 * dp) * c.s * (1 + 0.8 * f.pulse))))


class Lissajous(Preset):
    name, trail, desc = "lissajous", 0.90, "Stereo phase scope (mid/side XY plot) with long phosphor trails"

    def draw(self, d, f):
        c = self.c
        x = np.clip((f.wl - f.wr) * 0.7071, -1.3, 1.3)
        y = np.clip((f.wl + f.wr) * 0.7071, -1.3, 1.3)
        sc = min(c.W, c.H) * 0.36
        px, py = c.cx + x * sc, c.cy - y * sc
        d.ellipse([c.cx - sc, c.cy - sc, c.cx + sc, c.cy + sc], outline=c.pal(0.5, 0.15), width=max(1, int(c.s)))
        n = len(px)
        for i in range(0, n - 1, 8):
            pts = list(zip(px[i:i + 9], py[i:i + 9]))
            d.line(pts, fill=c.pal.cyc(i / n * 0.6 + f.hue + f.t * 0.04, 0.55 + 0.45 * f.energy), width=max(1, int(2 * c.s)))


class Scope(Preset):
    name, trail, desc = "scope", 0.60, "Zero-crossing-triggered stereo oscilloscope"

    def draw(self, d, f):
        c = self.c
        n = len(f.wl)
        xs = np.linspace(c.W * 0.04, c.W * 0.96, n)
        taper = np.sin(np.pi * np.linspace(0, 1, n)) ** 0.6
        amp = c.H * 0.28
        for w, off in ((f.wl, 0.0), (f.wr, 0.5)):
            ys = c.cy - w * amp * taper
            for i in range(0, n - 1, 16):
                pts = list(zip(xs[i:i + 17], ys[i:i + 17]))
                d.line(pts, fill=c.pal.cyc(i / n * 0.8 + f.hue + off, 0.6 + 0.4 * f.energy),
                       width=max(2, int(3 * c.s * (1 + 0.5 * f.pulse))))


class LightShow(Preset):
    name, trail, desc = "lightshow", 0.12, "Beat-triggered stage lights: each detected beat spawns a fresh burst of moving beams and spotlights"
    LIFE = 0.62

    def __init__(self, c):
        super().__init__(c)
        self.events = []

    def _update(self, f):
        c = self.c
        if f.beat:
            # Each beat launches a new lighting cue; strong beats spawn more fixtures.
            count = int(3 + 5 * f.strength)
            for _ in range(count):
                side = int(c.rng.integers(0, 4))
                if side == 0:
                    sx, sy = c.rng.uniform(0, c.W), 0.0
                elif side == 1:
                    sx, sy = c.W, c.rng.uniform(0, c.H * 0.35)
                elif side == 2:
                    sx, sy = c.rng.uniform(0, c.W), c.H * 0.12
                else:
                    sx, sy = 0.0, c.rng.uniform(0, c.H * 0.35)
                tx = c.rng.uniform(c.W * 0.08, c.W * 0.92)
                ty = c.rng.uniform(c.H * 0.28, c.H * 0.88)
                hue = c.rng.random()
                self.events.append((f.t, sx, sy, tx, ty, hue, c.rng.uniform(0.75, 1.25)))
        self.events = [e for e in self.events if f.t - e[0] < self.LIFE]
        return self.events

    def draw(self, d, f):
        c = self.c
        W, H = c.W, c.H
        for born, sx, sy, tx, ty, hue, scale in self._update(f):
            age = max(0.0, f.t - born)
            fade = max(0.0, 1.0 - age / self.LIFE) ** 1.7
            col = c.pal.cyc(hue + f.hue * 0.12, 0.95 * fade)
            tip_r = min(W, H) * 0.045 * scale * (0.65 + 0.35 * fade)
            # Broad translucent-looking beam cone, with a sharp bright core.
            dx, dy = tx - sx, ty - sy
            length = max(1.0, math.hypot(dx, dy))
            nx, ny = -dy / length, dx / length
            spread = min(W, H) * 0.055 * scale
            d.polygon([(sx - 2*c.s, sy - 2*c.s), (sx + 2*c.s, sy + 2*c.s),
                       (tx + nx*spread, ty + ny*spread), (tx - nx*spread, ty - ny*spread)], fill=col)
            d.line([(sx, sy), (tx, ty)], fill=c.pal.cyc(hue, fade), width=max(2, int(4*c.s*scale)))
            d.ellipse([tx-tip_r, ty-tip_r, tx+tip_r, ty+tip_r],
                      fill=c.pal.cyc(hue, fade), outline=c.pal.cyc(hue + 0.5, fade),
                      width=max(1, int(2*c.s)))
        # No permanent center light: the stage goes dark between beat-triggered cues.


class Orbit(Radial):
    name, trail, desc = "orbit", 0.90, "Layered orbital rings, spectral satellites and a pulsing energy core"


class Kaleido(Radial):
    name, trail, desc = "kaleido", 0.82, "Eight-way mirrored spectral geometry with rotating crystalline spokes"


class Grid(Terrain):
    name, trail, desc = "grid", 0.35, "Retrofuturist perspective grid with an audio-reactive horizon"


class Comets(Particles):
    name, trail, desc = "comets", 0.94, "Long-tailed audio-reactive comets orbiting and bursting from the core"


class Glitch(Bars):
    name, trail, desc = "glitch", 0.22, "Digital scanline interference, sliced spectrum blocks and signal tearing"


class Prism(Tunnel):
    name, trail, desc = "prism", 0.70, "Rotating glass-like polygon prisms with spectral refraction"


class MV(LightShow):
    name, trail, desc = "MV", 0.76, "Cinematic music-video scene: moving stage beams, horizon, skyline, starfield and waveform"


def emit_orbit(self, f):
    c = self.c
    base = min(c.W, c.H)
    out = []
    for k, mul in enumerate((0.19, 0.29, 0.40, 0.53)):
        a = np.linspace(0, 2*np.pi, 241)
        wob = 1 + 0.035*np.sin(a*5 + f.tphase + k) * (0.3 + f.mid)
        r = base * mul * (1 + 0.05*f.pulse) * wob
        out.append(("line", _poly(c.cx + np.cos(a + f.rot*(0.35+k*0.12))*r,
                                  c.cy + np.sin(a + f.rot*(0.35+k*0.12))*r,
                                  max(1.5, (2.0-k*0.25)*c.s),
                                  c.pal.cyc_arr(np.full(len(a)-1, k/4+f.hue), 0.48-k*0.06))))
    n = 96
    a = np.arange(n)*2*np.pi/n + f.rot
    v = resample(f.bands, n)
    r0 = base*0.22
    p0 = np.stack([c.cx+np.cos(a)*r0, c.cy+np.sin(a)*r0], 1)
    p1 = np.stack([c.cx+np.cos(a)*(r0+v*base*0.19), c.cy+np.sin(a)*(r0+v*base*0.19)], 1)
    out.append(("line", _segs(p0, p1, max(2, 2.4*c.s), c.pal.cyc_arr(np.arange(n)/n+f.hue, 0.55+0.45*v))))
    out.append(("disc", _discs([c.cx], c.cy, base*0.055*(1+f.pulse*0.5), 0, c.pal.cyc_arr([f.hue+0.5], 0.8))))
    return out


def emit_kaleido(self, f):
    c = self.c
    n = 72
    a = np.arange(n+1)*2*np.pi/n
    v = resample(f.bands, n)
    out = []
    for arm in range(8):
        ang = a + arm*np.pi/4 + f.rot*0.25
        r = min(c.W,c.H)*(0.10 + 0.31*np.r_[v,v[0]]*(1+0.1*f.pulse))
        x, y = c.cx+np.cos(ang)*r, c.cy+np.sin(ang)*r
        col = c.pal.cyc_arr(np.linspace(0,1,n)+arm/8+f.hue, 0.55+0.45*v)
        out.append(("line", _poly(x,y,max(1.5,2*c.s),col)))
    rings = []
    for k in range(1,5):
        a2 = np.linspace(0,2*np.pi,193)
        rr = min(c.W,c.H)*(0.08+k*0.075)*(1+0.025*np.sin(a2*8+f.tphase))
        rings.append(_poly(c.cx+np.cos(a2+f.rot*0.2)*rr,c.cy+np.sin(a2+f.rot*0.2)*rr,
                           max(1,c.s),c.pal.cyc_arr(np.full(len(a2)-1,k/5+f.hue),0.45)))
    out += [("line",np.concatenate(rings)),("disc",_discs([c.cx],c.cy,min(c.W,c.H)*0.035*(1+f.pulse),0,c.pal.cyc_arr([f.hue],0.9)))]
    return out


def emit_grid(self, f):
    c = self.c
    W,H=c.W,c.H
    horizon=H*(0.48-0.035*f.bass)
    out=[]
    # Perspective rails converge toward the audio-reactive horizon.
    for k in range(-12,13):
        x=W/2+k*W*0.075
        out.append(("line",_segs(np.array([[W/2+(x-W/2)*0.035,horizon]]),np.array([[x,H*1.08]]),
                                  max(1,c.s),c.pal.rgb((k+12)/24,0.55))))
    phase=(f.tphase*0.55)%1
    for j in range(18):
        z=((j/18+phase)%1)**2
        y=horizon+(H-horizon)*z
        out.append(("line",_segs(np.array([[W*0.02,y]]),np.array([[W*0.98,y]]),
                                  max(1,c.s*(0.8+z)),c.pal.rgb(0.35+z*0.55,0.25+0.55*z))))
    v=resample(f.bands,96)
    x=np.linspace(0,W,96)
    y=horizon-H*0.18*v*(0.5+0.5*f.pulse)
    out.append(("line",_poly(x,y,max(2,2.5*c.s),c.pal.cyc_arr(np.linspace(0,1,95)+f.hue,0.8))))
    return out


def emit_comets(self, f):
    c=self.c
    r0=self.step(f)
    idx=np.nonzero(self.life>0)[0]
    lf=self.life[idx]/self.mlife[idx]
    pos=self.pos[idx]
    prev=pos-self.vel[idx]*(1/c.fps)*5
    trails=_segs(prev,pos,np.maximum(1,c.s*(1+2*lf)),c.pal.cyc_arr(self.hue[idx]*0.7+f.hue,0.35+0.65*lf))
    heads=_discs(pos[:,0],pos[:,1],(1.5+4*lf)*c.s,0,c.pal.cyc_arr(self.hue[idx]*0.7+f.hue,0.4+0.6*lf))
    core=_discs([c.cx],c.cy,r0,0,c.pal.cyc_arr([f.hue+0.5],0.35+0.4*f.pulse))
    return [("line",trails),("disc",heads),("disc",core)]


def emit_glitch(self, f):
    c=self.c
    W,H=c.W,c.H
    n=50
    v=resample(f.bands,n)
    out=[]
    rows=[]
    for j in range(n):
        y=(j+0.5)*H/n
        width=W*(0.08+0.84*v[j])
        shift=math.sin(f.t*18+j*2.17)*W*0.012*(0.2+f.pulse)
        x0=max(0,W/2-width/2+shift)
        x1=min(W,W/2+width/2+shift)
        if x1>x0:
            rows.append(_rects([x0],[y-H/n*0.25],[x1],[y+H/n*0.25],c.pal.cyc_arr([j/n+f.hue],0.35+0.65*v[j])))
    if rows: out.append(("quad",np.concatenate(rows)))
    # Thin displaced scanlines and beat-triggered signal tears.
    ys=np.arange(0,H,max(5,int(8*c.s)))
    x0=np.zeros(len(ys)); x1=np.full(len(ys),W)
    out.append(("line",_segs(np.stack([x0,ys],1),np.stack([x1,ys],1),max(1,c.s),c.pal.rgb(0.6,0.13))))
    if f.beat:
        y=f.tphase%1*H
        out.append(("line",_segs(np.array([[0,y],[W,y+H*0.025]]),np.array([[W*0.35,y],[W,y+H*0.025]]),
                                  max(2,5*c.s),c.pal.cyc_arr([f.hue],0.9))))
    return out


def emit_prism(self, f):
    c=self.c
    W,H=c.W,c.H
    out=[]
    for k in range(12):
        z=((k/12+f.tphase*0.12)%1)
        r=min(W,H)*(0.06+0.68*z)
        sides=3+(k%5)
        a=np.linspace(0,2*np.pi,sides+1)+f.rot*(0.25+z*0.8)+k*0.13
        wob=1+0.12*f.bass*np.sin(a*3+f.tphase)
        x=c.cx+np.cos(a)*r*wob
        y=c.cy+np.sin(a)*r*wob
        col=c.pal.cyc_arr(np.full(sides,k/12+f.hue),0.28+0.62*(1-z))
        out.append(("line",_poly(x,y,max(1.2,(1+3*z)*c.s),col)))
        if k%2==0:
            out.append(("line",_segs(np.stack([np.full(sides,c.cx),np.full(sides,c.cy)],1),
                                      np.stack([x[:-1],y[:-1]],1),max(1,c.s),col,0.45)))
    return out


def emit_mv(self, f):
    c=self.c
    W,H=c.W,c.H
    out=[]
    # Scene chapters evolve over time; each chapter changes the camera composition.
    chapter=int(f.t/11.0)%4
    drift=f.t*0.18
    horizon=H*(0.58+0.035*math.sin(f.t*0.22))
    # Starfield / distant city lights.
    rng=np.random.default_rng(90210)
    stars=rng.uniform([0,0],[W,horizon*0.78],(110,2))
    twinkle=0.25+0.55*(0.5+0.5*math.sin(f.t*2.4))
    out.append(("disc",_discs(stars[:,0],stars[:,1],np.full(110,max(0.8,c.s)),0,
                              c.pal.cyc_arr(np.linspace(0,1,110),twinkle))))
    # Wide cinematic spotlights from alternating overhead trusses.
    beams=[]
    for k in range(6):
        sx=W*(0.08+k*0.168)
        sweep=math.sin(drift*(1+(k%3)*0.22)+k*1.7)*W*0.24
        tx=np.clip(sx+sweep,W*0.04,W*0.96)
        ty=horizon*(0.68+0.18*math.sin(f.t*0.35+k))
        beams.append(_segs(np.array([[sx,H*0.015]]),np.array([[tx,ty]]),max(2,c.s*(5+3*f.pulse)),
                           c.pal.cyc_arr([k/6+f.hue],0.30+0.35*f.energy)))
    out.append(("line",np.concatenate(beams)))
    # Mountain / skyline silhouette made from layered horizon strokes.
    for layer in range(3):
        xs=np.linspace(0,W,96)
        noise=np.sin(xs/W*math.pi*(3+layer)+f.t*(0.12+layer*0.04))*H*(0.035+layer*0.018)
        ys=horizon+noise+layer*H*0.045
        out.append(("line",_poly(xs,ys,max(1.5,c.s*(2-layer*0.35)),
                                  c.pal.cyc_arr(np.full(95,0.58+layer*0.1+f.hue),0.24+layer*0.08))))
    # Cinematic anamorphic waveform ribbon crossing the frame.
    x=np.linspace(-W*0.04,W*1.04,180)
    bands=resample(f.bands,180)
    y=H*(0.43+0.09*np.sin(x/W*math.pi*2+f.t*0.32)) - bands*H*(0.20+0.08*f.bass)
    ribbon=_poly(x,y,max(2.5,4*c.s),c.pal.cyc_arr(np.linspace(0,1,179)+f.hue,0.5+0.5*bands[:-1]))
    out.append(("line",ribbon))
    # Foreground skyline blocks, with windows pulsing on bass.
    buildings=[]
    for k in range(24):
        pitch = W / 24
        bw = pitch * 0.88
        bx = k * pitch + (pitch - bw) * 0.5
        bh=H*(0.08+0.13*(0.5+0.5*math.sin(k*8.31+chapter)))
        buildings.append(_rects([bx],[H-bh],[bx+bw],[H],
                                c.pal.cyc_arr([0.68+f.hue],0.12+0.12*f.bass)))
    out.append(("quad",np.concatenate(buildings)))
    # Chapter-specific geometric motif and a strong central lens flare on major beats.
    if chapter==0:
        a=np.linspace(0,2*np.pi,160)
        rr=min(W,H)*0.16*(1+0.12*f.pulse)
        out.append(("line",_poly(c.cx+np.cos(a+f.rot*0.25)*rr,c.cy+np.sin(a+f.rot*0.25)*rr,
                                  max(2,c.s*2),c.pal.cyc_arr(np.full(159,f.hue+0.5),0.55))))
    elif chapter==1:
        out.append(("line",_segs(np.array([[W*0.08,H*0.2],[W*0.92,H*0.2]]),
                                  np.array([[W*0.5,H*0.78],[W*0.5,H*0.78]]),
                                  max(2,c.s*2),c.pal.cyc_arr([f.hue,f.hue+0.5],0.7))))
    elif chapter==2:
        for k in range(5):
            yy=H*(0.18+k*0.12)
            out.append(("line",_segs(np.array([[W*0.12,yy],[W*0.88,yy]]),
                                      np.array([[W*0.88,yy],[W*0.12,yy]]),
                                      max(1,c.s),c.pal.cyc_arr([k/5+f.hue],0.45))))
    else:
        a=np.linspace(0,2*np.pi,12)
        rr=min(W,H)*(0.12+0.05*f.pulse)
        out.append(("line",_poly(c.cx+np.cos(a+f.rot)*rr,c.cy+np.sin(a+f.rot)*rr,
                                  max(2,c.s*3),c.pal.cyc_arr(np.full(11,f.hue),0.75))))
    if f.beat:
        out.append(("disc",_discs([c.cx],[c.cy],min(W,H)*(0.035+0.035*f.strength),0,
                                  c.pal.cyc_arr([f.hue+0.5],0.95))))
        # Anamorphic lens flare streak on the downbeat.
        flare_y = c.cy + math.sin(f.t*0.7)*H*0.08
        out.append(("line",_segs(np.array([[W*0.12,flare_y],[W*0.88,flare_y]]),
                                  np.array([[W*0.40,flare_y],[W*0.60,flare_y]]),
                                  max(2,c.s*(2+8*f.strength)),c.pal.cyc_arr([f.hue,f.hue+0.5],0.75))))
    # 2.39:1 letterbox framing gives the scene a filmic composition.
    out.append(("quad",_rects([0,0],[0,H*0.925],[W,W],[H*0.075,H],np.array([0.0,0.0,0.0]),1.0)))
    return out


class Wave(Lissajous):
    name, trail, desc = "wave", 0.88, "Layered flowing waveform ribbons with audio-reactive amplitude"


class Aurora(Terrain):
    name, trail, desc = "aurora", 0.92, "Flowing aurora curtains with layered spectral light bands"


class Rings(Radial):
    name, trail, desc = "rings", 0.86, "Concentric frequency rings that ripple and expand with bass"


class Spectrum(Bars):
    name, trail, desc = "spectrum", 0.18, "Wide cinematic spectrum columns with mirrored reflections"


class Launch(Preset):
    name, trail, desc = "launch", 0.10, "Fifty evenly spaced vertical launch lanes; beat strength selects short or long rising blocks"

    LANES = 50

    def __init__(self, c):
        super().__init__(c)
        self.events = []

    def emit(self, f):
        c = self.c
        W, H = c.W, c.H
        n = self.LANES
        pitch = W / n
        lane_x = (np.arange(n, dtype=np.float32) + 0.5) * pitch
        lane_energy = np.clip(resample(f.bands, n), 0.0, 1.0)

        # Each beat uses the same fixed 50-lane grid. No random X targets:
        # every block travels vertically, so columns stay perfectly aligned.
        if f.beat:
            active_floor = max(0.10, float(np.percentile(lane_energy, 38)))
            for lane, energy in enumerate(lane_energy):
                if energy < active_floor and c.rng.random() > 0.22:
                    continue
                # Two intentional size classes instead of noisy arbitrary lengths.
                is_long = (f.strength >= 0.72) or (energy >= np.percentile(lane_energy, 68))
                height = (0.235 if is_long else 0.105) * H
                life = 0.72 if is_long else 0.58
                hue = lane / n + f.hue * 0.10
                self.events.append((f.t, lane, height, hue, life, float(energy)))

        self.events = [e for e in self.events if f.t - e[0] < e[4]]
        if len(self.events) > 500:
            self.events = self.events[-500:]

        # The 50 launch pads share an exact pitch and common baseline.
        pad_w = pitch * 0.58
        pad_y0, pad_y1 = H * 0.955, H * 0.978
        pads = _rects(lane_x - pad_w / 2, pad_y0, lane_x + pad_w / 2, pad_y1,
                      c.pal.cyc_arr(np.arange(n) / n + f.hue * 0.10,
                                    0.22 + 0.42 * lane_energy))

        quads = []
        for born, lane, block_h, hue, life, energy in self.events:
            age = max(0.0, f.t - born)
            progress = min(1.0, age / life)
            # Ease-out motion: starts decisively, then settles toward a shared upper zone.
            travel = 1.0 - (1.0 - progress) ** 1.7
            x = lane_x[lane]
            bottom = H * 0.955 - travel * H * 0.79
            top = bottom - block_h
            width = pitch * (0.48 + 0.10 * energy)
            fade = max(0.0, 1.0 - progress) ** 0.72
            q = np.empty((1, 12), np.float32)
            q[0, 0:2] = (x - width / 2, top)
            q[0, 2:4] = (x + width / 2, top)
            q[0, 4:6] = (x + width / 2, bottom)
            q[0, 6:8] = (x - width / 2, bottom)
            q[0, 8:11] = c.pal.cyc_arr(np.array([hue + f.hue * 0.04]),
                                       0.48 + 0.52 * fade)[0]
            q[0, 11] = fade
            quads.append(q)

        if quads:
            return [("quad", pads), ("quad", np.concatenate(quads))]
        return [("quad", pads)]


class Morph(Preset):
    name, trail, desc = "morph", 0.08, "Continuously evolving liquid-marble fields with warped organic color regions"

    def emit(self, f):
        c = self.c
        W, H = c.W, c.H
        nx, ny = 96, 54
        x = (np.arange(nx) + 0.5) / nx
        y = (np.arange(ny) + 0.5) / ny
        xx, yy = np.meshgrid(x, y)
        # Smooth domain warping makes the color boundaries curl like flowing paint.
        t = f.t
        wx = xx + 0.105 * np.sin(yy * 8.2 + t * 0.19) + 0.045 * np.sin(xx * 13.0 - yy * 5.0 - t * 0.11)
        wy = yy + 0.105 * np.sin(xx * 7.0 - t * 0.16) + 0.040 * np.sin(yy * 12.0 + xx * 4.0 + t * 0.13)
        field = (np.sin(wx * 8.0 + np.sin(wy * 5.1 + t * 0.17) * 2.25)
                 + 0.58 * np.sin(wy * 10.5 - wx * 3.8 + t * 0.14)
                 + 0.28 * np.sin((wx + wy) * 17.0 - t * 0.10))
        field += 0.18 * f.bass * np.sin(wx * 5.0 + wy * 6.0 + t * 0.5)
        tone = np.clip(0.5 + field / 2.15 + f.hue * 0.035, 0.0, 1.0)
        # Stepped pigment regions, separated by fine dark contours like painted marbling.
        levels = np.minimum(15, np.floor(tone * 16.0).astype(np.int32))
        tone = levels / 15.0
        brightness = 0.68 + 0.24 * (0.5 + 0.5 * np.sin(field * 1.35 + t * 0.12)) + 0.08 * f.pulse
        colors = c.pal.arr(tone.ravel(), brightness.ravel())
        x0 = np.tile(np.arange(nx) * (W / nx), ny)
        y0 = np.repeat(np.arange(ny) * (H / ny), nx)
        # Build one colored quad per cell; the fine grid keeps boundaries visually smooth.
        quads = np.empty((nx * ny, 12), np.float32)
        quads[:, 0], quads[:, 1] = x0, y0
        quads[:, 2], quads[:, 3] = x0 + W / nx + 1.25 * c.s, y0
        quads[:, 4], quads[:, 5] = x0 + W / nx + 1.25 * c.s, y0 + H / ny + 1.25 * c.s
        quads[:, 6], quads[:, 7] = x0, y0 + H / ny + 1.25 * c.s
        quads[:, 8:11], quads[:, 11] = colors, 1.0

        # Trace only where neighboring pigment regions differ; these are thin, not block-sized, outlines.
        edge_v = levels[:, 1:] != levels[:, :-1]
        edge_h = levels[1:, :] != levels[:-1, :]
        vy, vx = np.nonzero(edge_v)
        hy, hx = np.nonzero(edge_h)
        p0 = np.concatenate([
            np.stack([(vx + 1) * (W / nx), vy * (H / ny)], axis=1),
            np.stack([hx * (W / nx), (hy + 1) * (H / ny)], axis=1)
        ]).astype(np.float32)
        p1 = np.concatenate([
            np.stack([(vx + 1) * (W / nx), (vy + 1) * (H / ny)], axis=1),
            np.stack([(hx + 1) * (W / nx), (hy + 1) * (H / ny)], axis=1)
        ]).astype(np.float32)
        edge_colors = np.tile(np.array([[0.025, 0.025, 0.035]], np.float32), (len(p0), 1))
        edges = _segs(p0, p1, max(1.2, 1.8 * c.s), edge_colors, 0.88)
        return [("quad", quads), ("line", edges)]


def emit_wave(self, f):
    c = self.c
    n = 240
    x = np.linspace(0, c.W, n)
    out = []
    for layer in range(5):
        u = np.linspace(0, 1, n)
        band = resample(f.bands, n)
        phase = f.tphase * (0.35 + layer * 0.08) + layer * 1.3
        amp = c.H * (0.055 + layer * 0.018)
        y = c.cy + np.sin(u * math.pi * (2.0 + layer * 0.32) + phase) * amp
        y += (band - 0.35) * c.H * (0.12 + layer * 0.018)
        out.append(("line", _poly(x, y, max(2, (3.5-layer*0.35)*c.s),
                                  c.pal.cyc_arr(u + layer*0.17 + f.hue, 0.38+0.5*band))))
    return out


def emit_aurora(self, f):
    c = self.c
    x = np.linspace(-0.05*c.W, 1.05*c.W, 260)
    u = x / c.W
    out = []
    for layer in range(7):
        phase = f.tphase * (0.22 + layer*0.025) + layer*0.9
        envelope = np.sin(np.pi*np.clip(u, 0, 1)) ** 0.45
        wave = np.sin(u*(5.2+layer*0.24)+phase) * (0.055+0.012*layer)
        wave += np.sin(u*11.0-phase*0.7) * 0.018
        y = c.H*(0.24+layer*0.075) + wave*c.H*envelope - f.bass*c.H*0.025
        col = c.pal.cyc_arr(np.full(len(x)-1, layer/7+f.hue), 0.25+0.12*layer+0.18*f.pulse)
        out.append(("line", _poly(x, y, max(2, (7-layer*0.55)*c.s), col, 0.8)))
    return out


def emit_rings(self, f):
    c = self.c
    base = min(c.W, c.H)
    out = []
    for k in range(12):
        n = 192
        a = np.linspace(0, 2*np.pi, n)
        v = resample(f.bands, n)
        idx = (k*7) % len(v)
        energy = float(v[idx])
        r = base*(0.055+k*0.038) * (1+0.10*f.pulse+0.08*energy)
        rr = r*(1+0.035*np.sin(a*(3+k%5)+f.tphase+k))
        x, y = c.cx+np.cos(a+f.rot*(0.15+k*0.025))*rr, c.cy+np.sin(a+f.rot*(0.15+k*0.025))*rr
        col = c.pal.cyc_arr(np.full(n-1, k/12+f.hue), 0.25+0.65*energy)
        out.append(("line", _poly(x,y,max(1.5,(1.4+energy*2.2)*c.s),col)))
    return out


def emit_spectrum(self, f):
    c = self.c
    n = 50
    v = resample(f.bands, n)
    x = np.linspace(c.W*0.06,c.W*0.94,n)
    bw = c.W*0.88/n*0.72
    base = c.H*0.52
    top = v*c.H*0.38*(1+0.12*f.pulse)
    x0,x1=x-bw/2,x+bw/2
    out = [("quad",_rects(x0,base-top,x1,base,c.pal.cyc_arr(np.linspace(0,1,n)+f.hue,0.4+0.6*v)))]
    out.append(("quad",_rects(x0,base+2*c.s,x1,base+top*0.28+2*c.s,
                               c.pal.cyc_arr(np.linspace(0,1,n)+f.hue,0.16))))
    return out


PRESETS = {cls.name: cls for cls in (Bars, Mirror, Radial, Terrain, Particles, Tunnel, Lissajous, Scope, LightShow,
                                      Orbit, Kaleido, Grid, Comets, Glitch, Prism, MV, Wave, Aurora, Rings, Spectrum,
                                      Launch, Morph)}


# ----------------------------------------------------------------------------
# Renderer
# ----------------------------------------------------------------------------
def make_bg(W, H, pal, solid=None):
    yy = np.linspace(0, 1, H)[:, None]
    xx = np.linspace(-1, 1, W)[None, :]
    if solid is not None:
        top = bot = np.array(solid, float)
    else:
        top, bot = pal.lut[0] * 0.10, pal.lut[-1] * 0.05
    grad = top[None, None, :] * (1 - yy)[..., None] + bot[None, None, :] * yy[..., None]
    vig = 1 - 0.55 * (xx ** 2 + (yy * 2 - 1) ** 2) / 2
    return Image.fromarray(np.clip(grad * vig[..., None], 0, 255).astype(np.uint8))


class Renderer:
    def __init__(self, args, an, pal, W, H):
        self.args, self.an = args, an
        self.W, self.H, self.aa = W, H, args.aa
        RW, RH = W * self.aa, H * self.aa
        self.RW, self.RH = RW, RH
        ctx = SimpleNamespace(W=RW, H=RH, s=RH / 720.0, cx=RW / 2, cy=RH / 2, pal=pal, fps=an.fps,
                              rng=np.random.default_rng(args.seed), nb=an.nb, sides=args.sides)
        names = list(PRESETS) if args.preset == "mix" else [args.preset]
        self.mix = args.preset == "mix"
        self.presets = [PRESETS[n](ctx) for n in names]
        self.cur = 0
        self.fx = Image.new("RGB", (RW, RH))
        self.bg = make_bg(RW, RH, pal, hex_to_rgb(args.bg) if args.bg else None)
        self.glow_lut = [min(255, int(i * args.glow * 1.3)) for i in range(256)] * 3

    def bloom(self, fx):
        W, H = fx.size
        a = fx.resize((W // 4, H // 4), Image.BILINEAR).filter(ImageFilter.GaussianBlur(2.5))
        b = fx.resize((W // 8, H // 8), Image.BILINEAR).filter(ImageFilter.GaussianBlur(4))
        a = a.resize((W, H), Image.BILINEAR)
        b = b.resize((W, H), Image.BILINEAR)
        return ImageChops.add(a, b).point(self.glow_lut)

    def layer(self, i):
        """GPU path: draw ONLY this frame's geometry on black (CPU).
        Trails, bloom, compositing, AA and aberration then happen on the GPU."""
        f = self.an.frame(i)
        reset = False
        if self.mix:
            k = int(f.t // self.args.switch) % len(self.presets)
            if k != self.cur:
                self.cur, reset = k, True
        p = self.presets[self.cur]
        img = Image.new("RGB", (self.RW, self.RH))
        p.draw(ImageDraw.Draw(img), f)
        return np.array(img), (p.trail, reset, f.pulse, f.bass)

    def render(self, i):
        f = self.an.frame(i)
        if self.mix:
            k = int(f.t // self.args.switch) % len(self.presets)
            if k != self.cur:
                self.cur = k
                self.fx = Image.new("RGB", (self.RW, self.RH))
        p = self.presets[self.cur]
        if p.trail > 0:
            self.fx = self.fx.point([int(v * p.trail) for v in range(256)] * 3)
        else:
            self.fx = Image.new("RGB", (self.RW, self.RH))
        p.draw(ImageDraw.Draw(self.fx), f)

        bg = ImageEnhance.Brightness(self.bg).enhance(1 + 0.8 * f.pulse + 0.3 * f.bass)
        out = ImageChops.add(bg, ImageChops.add(self.fx, self.bloom(self.fx)))
        if self.aa > 1:
            out = out.resize((self.W, self.H), Image.LANCZOS)
        ab = int(self.args.aberration * f.pulse * 4 * self.W / 1280)
        if ab > 0:
            r, g, b = out.split()
            out = Image.merge("RGB", (ImageChops.offset(r, ab, 0), g, ImageChops.offset(b, -ab, 0)))
        return out


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
RES = {"480p": (854, 480), "720p": (1280, 720), "1080p": (1920, 1080), "4k": (3840, 2160),
       "square": (1080, 1080), "vertical": (1080, 1920)}


def parse_res(s):
    if s in RES:
        return RES[s]
    try:
        w, h = s.lower().split("x")
        return int(w), int(h)
    except Exception:
        sys.exit("Bad --res. Use WxH (e.g. 1280x720) or one of: " + ", ".join(RES))


class GPUPost:
    """Trails + two-scale bloom + compositing + AA downsample + chromatic aberration on the GPU (PyTorch)."""

    def __init__(self, torch, dev, bg_img, W, H, aa, glow, aberration):
        self.t, self.F, self.dev = torch, torch.nn.functional, dev
        self.W, self.H, self.aa = W, H, aa
        self.glow, self.ab = glow * 1.3, aberration
        self.bg = self._up(np.array(bg_img))
        self.fx = torch.zeros_like(self.bg)
        self.kcache = {}

    def _up(self, arr):
        t = self.t.from_numpy(np.ascontiguousarray(arr)).to(self.dev)
        return t.permute(2, 0, 1).unsqueeze(0).float().div_(255.0)

    def _blur(self, x, sigma):
        k = self.kcache.get(sigma)
        if k is None:
            r = int(math.ceil(3 * sigma))
            ax = self.t.arange(-r, r + 1, device=self.dev, dtype=self.t.float32)
            g = self.t.exp(-(ax ** 2) / (2 * sigma * sigma))
            g = g / g.sum()
            k = (g.view(1, 1, 1, -1).repeat(3, 1, 1, 1), g.view(1, 1, -1, 1).repeat(3, 1, 1, 1), r)
            self.kcache[sigma] = k
        kh, kv, r = k
        x = self.F.conv2d(x, kh, padding=(0, r), groups=3)
        return self.F.conv2d(x, kv, padding=(r, 0), groups=3)

    def _bloom(self, fx):
        F = self.F
        a = self._blur(F.avg_pool2d(fx, 4), 2.5)
        b = self._blur(F.avg_pool2d(fx, 8), 4.0)
        size = fx.shape[-2:]
        a = F.interpolate(a, size=size, mode="bilinear", align_corners=False)
        b = F.interpolate(b, size=size, mode="bilinear", align_corners=False)
        return (a + b) * self.glow

    def process(self, arr, meta):
        trail, reset, pulse, bass = meta
        torch = self.t
        with torch.no_grad():
            new = self._up(arr)
            if trail > 0 and not reset:
                self.fx = torch.maximum(self.fx * trail, new)
            else:
                self.fx = new
            glow = self._bloom(self.fx) if self.glow > 0 else 0
            out = (self.bg * (1 + 0.8 * pulse + 0.3 * bass) + self.fx + glow).clamp_(0, 1)
            if self.aa > 1:
                out = self.F.avg_pool2d(out, self.aa)
            ab = int(self.ab * pulse * 4 * self.W / 1280)
            if ab > 0:
                out = torch.cat([torch.roll(out[:, 0:1], ab, 3), out[:, 1:2], torch.roll(out[:, 2:3], -ab, 3)], 1)
            out = (out * 255 + 0.5).clamp_(0, 255).byte().permute(0, 2, 3, 1)[0].contiguous().cpu().numpy()
        return out.tobytes()


def init_gpu(mode):
    """Returns (torch, device, name) or None."""
    if mode == "off":
        return None
    try:
        import torch
    except ImportError:
        if mode == "on":
            sys.exit("--gpu on needs PyTorch: pip install torch  (NVIDIA: see pytorch.org for the CUDA build)")
        print("Tip: `pip install torch` enables GPU post-processing (bloom/trails/compositing).")
        return None
    if torch.cuda.is_available():
        return torch, torch.device("cuda"), torch.cuda.get_device_name(0)
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch, torch.device("mps"), "Apple GPU (MPS)"
    msg = "PyTorch is installed but has no GPU support (CPU-only build?)."
    if mode == "on":
        sys.exit(msg + " For NVIDIA: pip install torch --index-url https://download.pytorch.org/whl/cu124")
    print(msg + " Falling back to CPU post-processing.")
    return None


def resolve_ffmpeg(prefer_hardware=True):
    """Prefer the bundled FFmpeg, then a modern system build with QSV support."""
    candidates = []
    # PyInstaller --onefile extracts --add-binary files into _MEIPASS.
    # Check this first so the portable build uses its own FFmpeg copy.
    if getattr(sys, "frozen", False):
        bundled = os.path.join(getattr(sys, "_MEIPASS", os.path.dirname(sys.executable)),
                               "ffmpeg.exe" if os.name == "nt" else "ffmpeg")
        if os.path.isfile(bundled):
            candidates.append(bundled)
    # Winget packages are not always added to PATH in the current terminal session.
    local = os.environ.get("LOCALAPPDATA", "")
    if local:
        candidates.extend(glob.glob(os.path.join(local, "Microsoft", "WinGet", "Packages",
                                                  "Gyan.FFmpeg*", "*", "bin", "ffmpeg.exe")))
    found = shutil.which("ffmpeg")
    if found:
        candidates.append(found)
    if os.name == "nt":
        try:
            r = subprocess.run(["where.exe", "ffmpeg"], capture_output=True, text=True, timeout=5)
            candidates.extend(x.strip() for x in r.stdout.splitlines() if x.strip())
        except Exception:
            pass
    # Keep order while removing duplicates.
    candidates = list(dict.fromkeys(candidates))
    valid = []
    for path in candidates:
        try:
            ver = subprocess.run([path, "-version"], capture_output=True, text=True, timeout=5)
            if ver.returncode != 0 or "ffmpeg version" not in (ver.stdout + ver.stderr).lower():
                continue
            enc = subprocess.run([path, "-hide_banner", "-encoders"], capture_output=True, text=True, timeout=8)
            text_enc = enc.stdout + enc.stderr
            has_qsv = "h264_qsv" in text_enc
            if has_qsv:
                # Encoder presence alone is not enough: verify that the Intel device initializes.
                test = subprocess.run([path, "-v", "error", "-f", "lavfi", "-i",
                                       "color=black:s=640x360:d=0.3:r=30", "-pix_fmt", "yuv420p",
                                       "-c:v", "h264_qsv", "-global_quality", "18", "-f", "null", "-"],
                                      capture_output=True, timeout=12)
                has_qsv = test.returncode == 0
            valid.append((path, has_qsv))
        except Exception:
            continue
    if not valid:
        return found or "ffmpeg"
    if prefer_hardware:
        for path, has_qsv in valid:
            if has_qsv:
                return path
    return valid[0][0]


HW_ENCODERS = {
    "nvenc": ("h264_nvenc", lambda crf: ["-rc", "vbr", "-cq", str(crf), "-b:v", "0"]),
    "qsv": ("h264_qsv", lambda crf: ["-global_quality", str(crf)]),
    "amf": ("h264_amf", lambda crf: ["-rc", "cqp", "-qp_i", str(crf), "-qp_p", str(crf)]),
    "videotoolbox": ("h264_videotoolbox", lambda crf: ["-b:v", "25M"]),
}


def video_codec_args(choice, crf, x264_preset, fps=30):
    """Pick an encoder and test it at the exact requested frame rate before rendering."""
    cpu = (["-c:v", "libx264", "-preset", x264_preset, "-crf", str(crf)], "libx264 (CPU)")
    if choice == "cpu":
        return cpu
    for key in (list(HW_ENCODERS) if choice == "auto" else [choice]):
        name, fn = HW_ENCODERS[key]
        a = ["-c:v", name] + fn(crf)
        source = f"color=black:s=640x360:d=0.5:r={fps}"
        try:
            t = subprocess.run([FFMPEG_BIN, "-v", "error", "-f", "lavfi", "-i", source,
                                "-pix_fmt", "yuv420p", "-r", str(fps)] + a + ["-f", "null", "-"],
                               capture_output=True, text=True, timeout=20)
        except Exception as exc:
            t = None
            detail = str(exc)
        else:
            detail = (t.stderr or "").strip()
        if t is not None and t.returncode == 0:
            return a, name + " (hardware)"
        print(f"{name} cannot encode at {fps} FPS with the current settings; "
              f"falling back to the next available encoder.")
        if detail:
            print("Encoder check:", detail.splitlines()[-1][:240])
    print("Using libx264 CPU encoding.")
    return cpu



# ----------------------------------------------------------------------------
# OpenGL renderer (moderngl): everything drawn by the GPU
# ----------------------------------------------------------------------------
def _segs(p0, p1, w, col, a=1.0):
    n = len(p0)
    out = np.empty((n, 9), np.float32)
    out[:, 0:2], out[:, 2:4], out[:, 4], out[:, 5:8], out[:, 8] = p0, p1, w, col, a
    return out


def _poly(xs, ys, w, col, a=1.0):
    p = np.stack([xs, ys], 1)
    # Per-vertex colors have one more sample than the line segments between them.
    if isinstance(col, np.ndarray) and col.ndim == 2 and len(col) == len(p):
        col = col[:-1]
    return _segs(p[:-1], p[1:], w, col, a)


def _rects(x0, y0, x1, y1, col, a=1.0):
    x0, y0, x1, y1 = [np.asarray(v, np.float32) for v in (x0, y0, x1, y1)]
    n = max(x0.size, y0.size, x1.size, y1.size)
    out = np.empty((n, 12), np.float32)
    out[:, 0], out[:, 1], out[:, 2], out[:, 3] = x0, y0, x1, y0
    out[:, 4], out[:, 5], out[:, 6], out[:, 7] = x1, y1, x0, y1
    out[:, 8:11], out[:, 11] = col, a
    return out


def _discs(x, y, r, th, col, a=1.0):
    x = np.atleast_1d(np.asarray(x, np.float32))
    out = np.empty((len(x), 8), np.float32)
    out[:, 0], out[:, 1], out[:, 2], out[:, 3], out[:, 4:7], out[:, 7] = x, y, r, th, col, a
    return out


def emit_bars(self, f):
    c = self.c
    W, H, n = c.W, c.H, self.n
    h = resample(f.bands, n)
    self.peak = np.maximum(h, self.peak - 0.9 / c.fps)
    m = W * 0.08
    bw = (W - 2 * m) / n
    gap = bw * 0.28
    base, maxh = H * 0.66, H * 0.52 * (1 + 0.06 * f.pulse)
    i = np.arange(n)
    x0, x1, u = m + i * bw + gap / 2, m + (i + 1) * bw - gap / 2, i / (n - 1)
    hh = np.maximum(h * maxh, 2 * c.s)
    py = base - self.peak * maxh - 6 * c.s
    return [("quad", np.concatenate([
        _rects(x0, base - hh, x1, base, c.pal.arr(u, 0.55 + 0.45 * h)),
        _rects(x0, base + gap, x1, base + gap + hh * 0.35, c.pal.arr(u, 0.18)),
        _rects(x0, py - 3 * c.s, x1, py, c.pal.arr(u, 1.0))]))]


def emit_mirror(self, f):
    c = self.c
    W, H = c.W, c.H
    step = max(2, int(3 * c.s))
    xs = np.arange(0, W + 1, step, dtype=float)
    p = np.abs(2 * xs / W - 1)
    v = spec_mirror(f.bands, p)
    hh = v * H * 0.40 * (1 + 0.1 * f.pulse) + 2 * c.s
    col = c.pal.cyc_arr(p * 0.9 + f.hue * 0.5, 0.45 + 0.55 * v)
    main = _segs(np.stack([xs, c.cy - hh], 1), np.stack([xs, c.cy + hh], 1), step, col)
    core = _segs(np.stack([xs, c.cy - hh * 0.22], 1), np.stack([xs, c.cy + hh * 0.22], 1), step,
                 np.clip(col * 0.35 + 0.65, 0, 1.2), 0.85)
    mid = _segs(np.array([[0, c.cy]]), np.array([[W, c.cy]]), max(1, c.s), c.pal.rgb(0.5, 0.5))
    return [("line", main), ("line", core), ("line", mid)]


def emit_radial(self, f):
    c = self.c
    base = min(c.W, c.H)
    r0 = base * 0.20 * (1 + 0.10 * f.pulse + 0.12 * f.bass)
    j = np.arange(self.n)
    p = np.abs(2 * j / self.n - 1)
    v = spec_mirror(f.bands, p)
    ang = j / self.n * 2 * np.pi + f.rot
    ln = v * base * 0.30 + 3 * c.s
    cs, sn = np.cos(ang), np.sin(ang)
    wd = max(2, int(2 * np.pi * r0 / self.n * 0.55))
    bars = _segs(np.stack([c.cx + cs * r0, c.cy + sn * r0], 1),
                 np.stack([c.cx + cs * (r0 + ln), c.cy + sn * (r0 + ln)], 1), wd,
                 c.pal.cyc_arr(p * 0.9 + f.hue * 0.6, 0.45 + 0.55 * v))
    w = resample(f.wl, 360)
    w = np.convolve(np.pad(w, 4, mode="wrap"), np.ones(9) / 9, "valid")
    a = np.linspace(0, 2 * np.pi, 361)
    rr = r0 * 0.80 * (1 + 0.35 * np.append(w, w[0]))
    ring = _poly(c.cx + np.cos(a + f.rot) * rr, c.cy + np.sin(a + f.rot) * rr,
                 max(2, 2.5 * c.s), c.pal.cyc_arr(f.hue + 0.5))
    edge = _discs([c.cx], c.cy, r0, max(2, 3 * c.s), c.pal.rgb(0.2, 0.6 + 0.4 * f.pulse))
    return [("line", bars), ("line", ring), ("disc", edge)]


def emit_terrain(self, f):
    c = self.c
    W, H, K, M = c.W, c.H, self.K, self.M
    every = max(1, round(c.fps / 10))
    u = np.linspace(0, 1, M)
    if self.tick % every == 0:
        self.rows.appendleft(spec_mirror(f.bands, np.abs(2 * u - 1)))
    self.tick += 1
    win = np.sin(np.pi * u) ** 0.9
    out = []
    for k in range(K - 1, -1, -1):
        z = k / (K - 1)
        y0 = H * (0.88 - 0.50 * z)
        xs = c.cx + (u - 0.5) * W * (0.94 - 0.44 * z)
        ys = y0 - self.rows[k] * H * 0.30 * (1 - 0.55 * z) * win
        occ = np.zeros((M - 1, 12), np.float32)
        occ[:, 0], occ[:, 1], occ[:, 2], occ[:, 3] = xs[:-1], ys[:-1], xs[1:], ys[1:]
        occ[:, 4], occ[:, 5], occ[:, 6], occ[:, 7], occ[:, 11] = xs[1:], H, xs[:-1], H, 1.0
        out.append(("quad", occ))
        out.append(("line", _poly(xs, ys, max(1, (2.4 - 1.5 * z) * c.s), c.pal.rgb(1 - z, 1.0 - 0.65 * z))))
    return out


def emit_particles(self, f):
    c = self.c
    r0 = self.step(f)
    idx = np.nonzero(self.life > 0)[0]
    lf = self.life[idx] / self.mlife[idx]
    r = (1.2 + 3.2 * lf) * c.s * (1 + 0.6 * f.energy)
    parts = _discs(self.pos[idx, 0], self.pos[idx, 1], r, 0.0, c.pal.cyc_arr(self.hue[idx] * 0.7 + f.hue, 0.25 + 0.75 * lf))
    core = _discs([c.cx], c.cy, r0, 0.0, c.pal.rgb(0.3, 0.25 + 0.4 * f.pulse))
    edge = _discs([c.cx], c.cy, r0, max(2, 3 * c.s), c.pal.cyc_arr(f.hue + 0.5))
    return [("disc", parts), ("disc", core), ("disc", edge)]


def emit_tunnel(self, f):
    c = self.c
    diag = math.hypot(c.W, c.H) / 2
    sides = c.sides
    th = np.linspace(0, 2 * np.pi, sides * 12, endpoint=False)
    poly = np.cos(np.pi / sides) / np.cos((th % (2 * np.pi / sides)) - np.pi / sides)
    v = spec_mirror(f.bands, np.abs(2 * th / (2 * np.pi) - 1))
    segs = []
    for k in range(self.K):
        dp = ((k + f.tphase) % self.K) / self.K
        r = diag * 0.04 + diag * 1.1 * dp ** 2.4
        ang = th + f.rot * 0.5 + dp * 0.9 * math.sin(f.t * 0.2)
        rr = r * poly * (1 + 0.5 * v * dp ** 0.6)
        xs, ys = c.cx + np.cos(ang) * rr, c.cy + np.sin(ang) * rr
        fade = min(1.0, dp * 6) * (0.35 + 0.65 * dp ** 0.5)
        segs.append(_poly(np.append(xs, xs[0]), np.append(ys, ys[0]),
                          max(1.0, (1 + 5 * dp) * c.s * (1 + 0.8 * f.pulse)),
                          c.pal.cyc_arr(k / self.K + f.hue * 0.5 + f.t * 0.03, fade)))
    return [("line", np.concatenate(segs))]


def emit_lissajous(self, f):
    c = self.c
    x = np.clip((f.wl - f.wr) * 0.7071, -1.3, 1.3)
    y = np.clip((f.wl + f.wr) * 0.7071, -1.3, 1.3)
    sc = min(c.W, c.H) * 0.36
    n = len(x)
    col = c.pal.cyc_arr(np.arange(n - 1) / n * 0.6 + f.hue + f.t * 0.04, 0.55 + 0.45 * f.energy)
    return [("disc", _discs([c.cx], c.cy, sc, max(1, c.s), c.pal.rgb(0.5, 0.15))),
            ("line", _poly(c.cx + x * sc, c.cy - y * sc, max(1, 2 * c.s), col))]


def emit_lightshow(self, f):
    c = self.c
    segs, discs = [], []
    for born, sx, sy, tx, ty, hue, scale in self._update(f):
        age = max(0.0, f.t - born)
        fade = max(0.0, 1.0 - age / self.LIFE) ** 1.7
        col = c.pal.cyc_arr(np.array([hue]), fade)[0]
        tip_r = min(c.W, c.H) * 0.045 * scale * (0.65 + 0.35 * fade)
        segs.append(_segs(np.array([[sx, sy]]), np.array([[tx, ty]]), max(2, 4*c.s*scale), col))
        discs.append(_discs([tx], ty, tip_r, 0.0, col))
    if not segs:
        return [("line", np.empty((0, 9), np.float32)), ("disc", np.empty((0, 8), np.float32))]
    return [("line", np.concatenate(segs)), ("disc", np.concatenate(discs))]


def emit_scope(self, f):
    c = self.c
    n = len(f.wl)
    xs = np.linspace(c.W * 0.04, c.W * 0.96, n)
    taper = np.sin(np.pi * np.linspace(0, 1, n)) ** 0.6
    out = []
    for w, off in ((f.wl, 0.0), (f.wr, 0.5)):
        col = c.pal.cyc_arr(np.arange(n - 1) / n * 0.8 + f.hue + off, 0.6 + 0.4 * f.energy)
        out.append(("line", _poly(xs, c.cy - w * c.H * 0.28 * taper, max(2, 3 * c.s * (1 + 0.5 * f.pulse)), col)))
    return out


def draw_emitted(self, d, f):
    """PIL fallback for the same geometry batches used by the OpenGL renderer."""
    for kind, arr in self.emit(f):
        for row in arr:
            if kind == "line":
                rgb = tuple(int(np.clip(v * 255, 0, 255)) for v in row[5:8])
                alpha = float(np.clip(row[8], 0, 1))
                col = tuple(int(v * alpha) for v in rgb)
                d.line((tuple(row[0:2]), tuple(row[2:4])), fill=col, width=max(1, int(row[4])))
            elif kind == "disc":
                x, y, r, th = row[:4]
                rgb = tuple(int(np.clip(v * 255, 0, 255)) for v in row[4:7])
                alpha = float(np.clip(row[7], 0, 1))
                col = tuple(int(v * alpha) for v in rgb)
                bb = [x-r, y-r, x+r, y+r]
                if th <= 0:
                    d.ellipse(bb, fill=col)
                else:
                    d.ellipse(bb, outline=col, width=max(1, int(th)))
            elif kind == "quad":
                pts = [tuple(row[j:j+2]) for j in (0,2,4,6)]
                rgb = tuple(int(np.clip(v * 255, 0, 255)) for v in row[8:11])
                alpha = float(np.clip(row[11], 0, 1))
                d.polygon(pts, fill=tuple(int(v * alpha) for v in rgb))


for _cls, _fn in ((Bars, emit_bars), (Mirror, emit_mirror), (Radial, emit_radial), (Terrain, emit_terrain),
                  (Particles, emit_particles), (Tunnel, emit_tunnel), (Lissajous, emit_lissajous), (Scope, emit_scope),
                  (LightShow, emit_lightshow), (Orbit, emit_orbit), (Kaleido, emit_kaleido), (Grid, emit_grid),
                  (Comets, emit_comets), (Glitch, emit_glitch), (Prism, emit_prism), (MV, emit_mv),
                  (Wave, emit_wave), (Aurora, emit_aurora), (Rings, emit_rings), (Spectrum, emit_spectrum)):
    _cls.emit = _fn
for _cls in (Orbit, Kaleido, Grid, Comets, Glitch, Prism, MV, Wave, Aurora, Rings, Spectrum, Launch, Morph):
    _cls.draw = draw_emitted

_FS_VERT = """#version 330
out vec2 uv;
void main(){ vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2); uv = p; gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0); }"""

_LINE_VS = """#version 330
uniform vec2 uRes;
in vec4 seg; in float width; in vec4 color;
out vec2 vPos; flat out vec4 vSeg; flat out float vW; flat out vec4 vCol;
void main(){
  vec2 p0 = seg.xy, p1 = seg.zw, d = p1 - p0;
  float len = length(d);
  vec2 t = len > 1e-4 ? d / len : vec2(1.0, 0.0);
  vec2 n = vec2(-t.y, t.x);
  float hw = width * 0.5 + 1.5;
  float cx = float(gl_VertexID / 2), cy = float(gl_VertexID % 2) * 2.0 - 1.0;
  vec2 pos = mix(p0, p1, cx) + t * (cx * 2.0 - 1.0) * hw + n * cy * hw;
  vPos = pos; vSeg = seg; vW = width; vCol = color;
  gl_Position = vec4(pos.x / uRes.x * 2.0 - 1.0, 1.0 - pos.y / uRes.y * 2.0, 0.0, 1.0);
}"""
_LINE_FS = """#version 330
in vec2 vPos; flat in vec4 vSeg; flat in float vW; flat in vec4 vCol; out vec4 o;
void main(){
  vec2 a = vSeg.xy, ba = vSeg.zw - a, pa = vPos - a;
  float h = clamp(dot(pa, ba) / max(dot(ba, ba), 1e-6), 0.0, 1.0);
  float d = length(pa - ba * h);
  o = vec4(vCol.rgb, clamp(vW * 0.5 + 0.5 - d, 0.0, 1.0) * vCol.a);
}"""

_DISC_VS = """#version 330
uniform vec2 uRes;
in vec4 dp; in vec4 color;
out vec2 vPos; flat out vec4 vDp; flat out vec4 vCol;
void main(){
  float half_ = dp.z + dp.w * 0.5 + 1.5;
  vec2 q = vec2(float(gl_VertexID / 2) * 2.0 - 1.0, float(gl_VertexID % 2) * 2.0 - 1.0);
  vec2 pos = dp.xy + q * half_;
  vPos = pos; vDp = dp; vCol = color;
  gl_Position = vec4(pos.x / uRes.x * 2.0 - 1.0, 1.0 - pos.y / uRes.y * 2.0, 0.0, 1.0);
}"""
_DISC_FS = """#version 330
in vec2 vPos; flat in vec4 vDp; flat in vec4 vCol; out vec4 o;
void main(){
  float d = length(vPos - vDp.xy);
  float a = vDp.w <= 0.0 ? clamp(vDp.z + 0.5 - d, 0.0, 1.0) : clamp(vDp.w * 0.5 + 0.5 - abs(d - vDp.z), 0.0, 1.0);
  o = vec4(vCol.rgb, a * vCol.a);
}"""

_QUAD_VS = """#version 330
uniform vec2 uRes;
in vec4 c01; in vec4 c23; in vec4 color;
flat out vec4 vCol;
void main(){
  vec2 p = gl_VertexID == 0 ? c01.xy : (gl_VertexID == 1 ? c01.zw : (gl_VertexID == 2 ? c23.zw : c23.xy));
  vCol = color;
  gl_Position = vec4(p.x / uRes.x * 2.0 - 1.0, 1.0 - p.y / uRes.y * 2.0, 0.0, 1.0);
}"""
_QUAD_FS = """#version 330
flat in vec4 vCol; out vec4 o;
void main(){ o = vCol; }"""

_FADE_FS = """#version 330
uniform float uA; out vec4 o;
void main(){ o = vec4(0.0, 0.0, 0.0, uA); }"""

_DOWN_FS = """#version 330
uniform sampler2D uTex; uniform vec2 uStep; in vec2 uv; out vec4 o;
void main(){
  o = 0.25 * (texture(uTex, uv + uStep * vec2(-1, -1)) + texture(uTex, uv + uStep * vec2(1, -1)) +
              texture(uTex, uv + uStep * vec2(-1, 1)) + texture(uTex, uv + uStep * vec2(1, 1)));
}"""

_BLUR_FS = """#version 330
uniform sampler2D uTex; uniform vec2 uDir; uniform int uR; uniform float uW[33]; in vec2 uv; out vec4 o;
void main(){
  vec4 s = texture(uTex, uv) * uW[0];
  for (int i = 1; i <= uR; i++) { s += (texture(uTex, uv + uDir * float(i)) + texture(uTex, uv - uDir * float(i))) * uW[i]; }
  o = s;
}"""

_COMP_FS = """#version 330
uniform sampler2D uFx; uniform sampler2D uA; uniform sampler2D uB;
uniform vec3 uTop; uniform vec3 uBot; uniform float uBright, uGlow, uAb, uSeed;
in vec2 uv; out vec4 o;
vec3 scene(vec2 p){ return texture(uFx, p).rgb + (texture(uA, p).rgb + texture(uB, p).rgb) * uGlow; }
float hash(vec2 p){ return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453); }
void main(){
  vec2 s = vec2(uv.x, 1.0 - uv.y);
  float xx = uv.x * 2.0 - 1.0, yy = uv.y, yc = yy * 2.0 - 1.0;
  vec3 bg = mix(uTop, uBot, yy) * (1.0 - 0.275 * (xx * xx + yc * yc)) * uBright;
  vec3 c = vec3(scene(s - vec2(uAb, 0.0)).r, scene(s).g, scene(s + vec2(uAb, 0.0)).b);
  c += bg;
  float m = max(c.r, max(c.g, c.b));
  if (m > 0.8) c *= (0.8 + 0.2 * (1.0 - exp(-(m - 0.8) / 0.2))) / m;   // soft knee, keeps hue instead of clipping to white
  o = vec4(clamp(c, 0.0, 1.0) + (hash(gl_FragCoord.xy + uSeed) - 0.5) / 255.0, 1.0);
}"""


def make_gl_context():
    try:
        import moderngl
    except Exception as e:  # ImportError, missing DLLs, ABI mismatch...
        return None, (f"could not import moderngl: {type(e).__name__}: {e} "
                      f"[running Python {sys.version.split()[0]} at {sys.executable}; "
                      f"install into THIS interpreter with: \"{sys.executable}\" -m pip install moderngl]")
    last = None
    for kw in ({}, {"backend": "egl"}):
        try:
            return moderngl.create_standalone_context(require=330, **kw), None
        except Exception as e:  # noqa
            last = e
    return None, f"no OpenGL 3.3 context available ({str(last).splitlines()[0] if str(last) else last})"


class GLRenderer:
    """Full-GPU renderer: instanced SDF lines/discs/quads -> trail buffer -> bloom -> composite -> RGB bytes."""

    def __init__(self, ctx, args, an, pal, W, H):
        import moderngl
        self.mgl, self.ctx, self.args, self.an, self.W, self.H = moderngl, ctx, args, an, W, H
        self.gpu_name = ctx.info.get("GL_RENDERER", "?")
        c = SimpleNamespace(W=W, H=H, s=H / 720.0, cx=W / 2, cy=H / 2, pal=pal, fps=an.fps,
                            rng=np.random.default_rng(args.seed), nb=an.nb, sides=args.sides)
        names = list(PRESETS) if args.preset == "mix" else [args.preset]
        self.mix, self.presets, self.cur = args.preset == "mix", [PRESETS[n](c) for n in names], 0
        P = ctx.program
        self.prog = {
            "line": P(vertex_shader=_LINE_VS, fragment_shader=_LINE_FS),
            "disc": P(vertex_shader=_DISC_VS, fragment_shader=_DISC_FS),
            "quad": P(vertex_shader=_QUAD_VS, fragment_shader=_QUAD_FS)}
        self.fmt = {"line": ("4f 1f 4f/i", ("seg", "width", "color")), "disc": ("4f 4f/i", ("dp", "color")),
                    "quad": ("4f 4f 4f/i", ("c01", "c23", "color"))}
        self.slots = {"line": [], "disc": [], "quad": []}
        self.fade = P(vertex_shader=_FS_VERT, fragment_shader=_FADE_FS)
        self.down = P(vertex_shader=_FS_VERT, fragment_shader=_DOWN_FS)
        self.blur = P(vertex_shader=_FS_VERT, fragment_shader=_BLUR_FS)
        self.comp = P(vertex_shader=_FS_VERT, fragment_shader=_COMP_FS)
        self.tri = {k: ctx.vertex_array(p, []) for k, p in (("fade", self.fade), ("down", self.down),
                                                             ("blur", self.blur), ("comp", self.comp))}

        def target(w, h, dtype):
            t = ctx.texture((max(1, w), max(1, h)), 4, dtype=dtype)
            t.filter, t.repeat_x, t.repeat_y = (moderngl.LINEAR, moderngl.LINEAR), False, False
            return t, ctx.framebuffer(color_attachments=[t])

        self.fx, self.fx_fbo = target(W, H, "f2")
        self.A, self.A_fbo = target(W // 4, H // 4, "f2")
        self.A2, self.A2_fbo = target(W // 4, H // 4, "f2")
        self.B, self.B_fbo = target(W // 8, H // 8, "f2")
        self.B2, self.B2_fbo = target(W // 8, H // 8, "f2")
        self.out, self.out_fbo = target(W, H, "f1")
        for pr in self.prog.values():
            pr["uRes"].value = (float(W), float(H))
        solid = hex_to_rgb(args.bg) if args.bg else None
        top = np.array(solid if solid else pal.lut[0] * 0.10, float) / 255.0
        bot = np.array(solid if solid else pal.lut[-1] * 0.05, float) / 255.0
        self.comp["uTop"].value, self.comp["uBot"].value = tuple(top), tuple(bot)
        self.glow = args.glow * 1.3
        self.ab = args.aberration
        self.kern = {s: self._kernel(s) for s in (2.5, 4.0)}

    @staticmethod
    def _kernel(sigma):
        r = min(32, int(math.ceil(3 * sigma)))
        w = np.exp(-(np.arange(r + 1) ** 2) / (2 * sigma * sigma))
        w = w / (w[0] + 2 * w[1:].sum())
        return r, np.pad(w, (0, 33 - len(w))).astype("f4")

    def _slot(self, kind, k, nbytes):
        lst = self.slots[kind]
        while len(lst) <= k:
            lst.append(None)
        s = lst[k]
        if s is None or s[2] < nbytes:
            cap = max(4096, 1 << (int(nbytes) - 1).bit_length())
            buf = self.ctx.buffer(reserve=cap, dynamic=True)
            fmt, names = self.fmt[kind]
            s = lst[k] = (buf, self.ctx.vertex_array(self.prog[kind], [(buf, fmt, *names)]), cap)
        return s

    def _blur_pass(self, src, dst_fbo, size, direction, kern):
        r, w = kern
        dst_fbo.use()
        src.use(0)
        self.blur["uTex"].value = 0
        self.blur["uDir"].value = (direction[0] / size[0], direction[1] / size[1])
        self.blur["uR"].value = r
        self.blur["uW"].write(w.tobytes())
        self.tri["blur"].render(self.mgl.TRIANGLES, vertices=3)

    def _down_pass(self, src, src_size, dst_fbo, factor):
        dst_fbo.use()
        src.use(0)
        self.down["uTex"].value = 0
        self.down["uStep"].value = (factor / src_size[0], factor / src_size[1])
        self.tri["down"].render(self.mgl.TRIANGLES, vertices=3)

    def render_bytes(self, i):
        mgl, ctx, W, H = self.mgl, self.ctx, self.W, self.H
        f = self.an.frame(i)
        reset = False
        if self.mix:
            k = int(f.t // self.args.switch) % len(self.presets)
            if k != self.cur:
                self.cur, reset = k, True
        p = self.presets[self.cur]
        batches = p.emit(f)

        self.fx_fbo.use()
        ctx.enable(mgl.BLEND)
        ctx.blend_func = mgl.SRC_ALPHA, mgl.ONE_MINUS_SRC_ALPHA
        if p.trail > 0 and not reset:
            self.fade["uA"].value = 1.0 - p.trail
            self.tri["fade"].render(mgl.TRIANGLES, vertices=3)
        else:
            self.fx_fbo.clear(0.0, 0.0, 0.0, 0.0)
        used = {"line": 0, "disc": 0, "quad": 0}
        for kind, arr in batches:
            if len(arr) == 0:
                continue
            buf, vao, _ = self._slot(kind, used[kind], arr.nbytes)
            used[kind] += 1
            buf.write(np.ascontiguousarray(arr, np.float32).tobytes())
            vao.render(mgl.TRIANGLE_STRIP, vertices=4, instances=len(arr))
        ctx.disable(mgl.BLEND)

        if self.glow > 0:
            self._down_pass(self.fx, (W, H), self.A_fbo, 1.0)
            self._blur_pass(self.A, self.A2_fbo, (W // 4, H // 4), (1, 0), self.kern[2.5])
            self._blur_pass(self.A2, self.A_fbo, (W // 4, H // 4), (0, 1), self.kern[2.5])
            self._down_pass(self.A, (W // 4, H // 4), self.B_fbo, 0.5)
            self._blur_pass(self.B, self.B2_fbo, (W // 8, H // 8), (1, 0), self.kern[4.0])
            self._blur_pass(self.B2, self.B_fbo, (W // 8, H // 8), (0, 1), self.kern[4.0])

        self.out_fbo.use()
        for n, t in enumerate((self.fx, self.A, self.B)):
            t.use(n)
        cp = self.comp
        cp["uFx"].value, cp["uA"].value, cp["uB"].value = 0, 1, 2
        cp["uBright"].value = 1.0 + 0.8 * f.pulse + 0.3 * f.bass
        cp["uGlow"].value = self.glow if self.glow > 0 else 0.0
        cp["uAb"].value = self.ab * f.pulse * 4 / 1280.0
        cp["uSeed"].value = (i * 0.37) % 100.0
        self.tri["comp"].render(mgl.TRIANGLES, vertices=3)
        return self.out_fbo.read(components=3, alignment=1)


def try_create_gl(args, an, pal, W, H, strict):
    ctx, err = make_gl_context()
    if ctx is None:
        if strict:
            sys.exit("OpenGL renderer unavailable: " + err)
        print("OpenGL renderer unavailable (" + err + ") - using the PIL renderer.")
        return None
    name = ctx.info.get("GL_RENDERER", "")
    if any(s in name.lower() for s in ("llvmpipe", "software", "basic render", "gdi generic")):
        print(f"Warning: OpenGL is running on a software renderer ({name}); update your GPU drivers.")
    try:
        return GLRenderer(ctx, args, an, pal, W, H)
    except Exception as e:  # noqa
        if strict:
            sys.exit(f"OpenGL renderer failed to initialize: {e}")
        print(f"OpenGL renderer failed to initialize ({e}) - using the PIL renderer.")
        return None


def pick_audio_codec():
    """Find an AAC-ish encoder this ffmpeg build actually supports."""
    try:
        enc = subprocess.run([FFMPEG_BIN, "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
    except Exception:
        enc = ""
    if " aac " in enc:
        return ["-c:a", "aac", "-strict", "-2", "-b:a", "256k"]  # -strict -2 needed on old builds
    for name in ("libfdk_aac", "libvo_aacenc"):
        if name in enc:
            return ["-c:a", name, "-b:a", "256k"]
    if "libmp3lame" in enc:
        return ["-c:a", "libmp3lame", "-b:a", "256k"]
    return ["-c:a", "aac", "-strict", "-2", "-b:a", "256k"]


def apply_transition(frame, W, H, i, total, fps, args):
    """Apply optional intro/outro transitions after rendering; zero duration is free."""
    intro = max(0.0, args.transition_in)
    outro = max(0.0, args.transition_out)
    t = i / max(1, fps)
    remaining = (total - 1 - i) / max(1, fps)
    amount = None
    if intro > 0 and t < intro:
        amount = min(amount if amount is not None else 1.0, t / intro)
    if outro > 0 and remaining < outro:
        amount = min(amount if amount is not None else 1.0, remaining / outro)
    if amount is None:
        return frame
    x = max(0.0, min(1.0, amount))
    eased = x*x*(3.0-2.0*x)
    style = args.transition_style
    image = Image.frombytes("RGB", (W, H), frame)
    if style in ("blur", "fade_blur"):
        radius = max(0.0, args.transition_blur) * (1.0-eased)
        if radius > 0.05:
            image = image.filter(ImageFilter.GaussianBlur(radius))
    if style == "wipe":
        mask = Image.new("L", (W, H), 0)
        md = ImageDraw.Draw(mask)
        md.rectangle((0, 0, int(W*eased), H), fill=255)
        black = Image.new("RGB", (W, H), (0, 0, 0))
        image = Image.composite(image, black, mask)
    elif style == "zoom":
        scale = 1.0 + (1.0-eased)*0.16
        nw, nh = max(W, int(W*scale)), max(H, int(H*scale))
        image = image.resize((nw, nh), Image.Resampling.BICUBIC)
        left, top = (nw-W)//2, (nh-H)//2
        image = image.crop((left, top, left+W, top+H))
        image = ImageEnhance.Brightness(image).enhance(eased)
    elif style in ("fade", "fade_blur"):
        image = ImageEnhance.Brightness(image).enhance(eased)
    elif style == "blur":
        pass
    return image.tobytes()


def main():
    ap = argparse.ArgumentParser(description="Audio visualizer: audio file -> mp4", formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", nargs="?")
    ap.add_argument("-o", "--output")
    ap.add_argument("-p", "--preset", default="radial", choices=list(PRESETS) + ["mix"])
    ap.add_argument("--palette", default="neon", choices=list(PALETTES))
    ap.add_argument("--colors", help='Custom palette, comma-separated hex: "#ff0055,#ffaa00,#fff200"')
    ap.add_argument("--bg", help="Solid background hex color (default: tinted gradient from palette)")
    ap.add_argument("--res", default="1280x720", help="WxH or 480p/720p/1080p/4k/square/vertical")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--glow", type=float, default=1.0, help="Bloom intensity (0 = off)")
    ap.add_argument("--smooth", type=float, default=0.5, help="0 = twitchy, 1 = silky")
    ap.add_argument("--bands", type=int, default=64)
    ap.add_argument("--aa", type=int, default=1, choices=[1, 2], help="2 = supersampled antialiasing (4x slower)")
    ap.add_argument("--sides", type=int, default=6, help="Polygon sides for tunnel preset")
    ap.add_argument("--aberration", type=float, default=1.0, help="Chromatic aberration on beats (0 = off)")
    ap.add_argument("--transition-in", type=float, default=0.0, help="Intro transition duration in seconds (0 = off)")
    ap.add_argument("--transition-out", type=float, default=0.0, help="Outro transition duration in seconds (0 = off)")
    ap.add_argument("--transition-style", choices=["fade", "blur", "fade_blur", "wipe", "zoom"], default="fade_blur")
    ap.add_argument("--transition-blur", type=float, default=18.0, help="Maximum intro/outro Gaussian blur radius")
    ap.add_argument("--switch", type=float, default=20.0, help="Seconds between presets in 'mix' mode")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--duration", type=float, default=0.0)
    ap.add_argument("--preview", type=float, metavar="SEC", help="Render a single PNG at this time and exit")
    ap.add_argument("--crf", type=int, default=18)
    ap.add_argument("--x264-preset", default="medium")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--no-analysis-cache", action="store_true", help="Disable the persistent audio-analysis cache")
    ap.add_argument("--renderer", default="auto", choices=["auto", "gl", "pil"],
                    help="gl = full OpenGL GPU renderer (fastest), pil = CPU drawing. auto tries gl first")
    ap.add_argument("--gpu", default="auto", choices=["auto", "on", "off"], help="GPU post-processing via PyTorch")
    ap.add_argument("--encoder", default="auto", choices=["auto", "cpu"] + list(HW_ENCODERS),
                    help="Video encoder: auto tries NVENC/QSV/AMF/VideoToolbox, else x264")
    ap.add_argument("--list", action="store_true", help="List presets and palettes")
    args = ap.parse_args()

    if args.list:
        print("PRESETS")
        for n, cls in PRESETS.items():
            print(f"  {n:<10} {cls.desc}")
        print(f"  {'mix':<10} Cycles through every preset every --switch seconds")
        print("\nPALETTES:", ", ".join(PALETTES))
        return
    if not args.input:
        ap.error("input audio file required")
    if not os.path.exists(args.input):
        sys.exit("File not found: " + args.input)
    global FFMPEG_BIN
    FFMPEG_BIN = resolve_ffmpeg()
    if not FFMPEG_BIN or not os.path.exists(FFMPEG_BIN):
        sys.exit("ffmpeg not found. Install it (brew install ffmpeg / apt install ffmpeg / winget install ffmpeg).")
    print("FFmpeg:", FFMPEG_BIN)

    pal = Palette([c.strip() for c in args.colors.split(",")] if args.colors else PALETTES[args.palette])
    W, H = parse_res(args.res)
    W, H = W - W % 2, H - H % 2
    base = os.path.splitext(args.input)[0]

    cache_file = None
    an = None
    if not args.no_analysis_cache:
        print("Checking audio-analysis cache (content fingerprint)...")
        cache_file = analysis_cache_path(args.input, 44100, args.fps, args.bands,
                                         args.smooth, args.start, args.duration)

    print("Decoding audio...")
    stereo = load_audio(args.input, 44100, args.start, args.duration)
    if cache_file:
        an = load_analysis_cache(cache_file, stereo, 44100, args.fps, args.bands)
    if an is not None:
        print(f"Analysis cache hit: loaded {an.nf:,} analyzed frames.")
    else:
        print(f"Analyzing {len(stereo) / 44100:.1f}s of audio (multi-resolution STFT, onset detection)...")
        an = Analysis(stereo, 44100, args.fps, nb=args.bands, smooth=args.smooth)
        if cache_file:
            save_analysis_cache(cache_file, an)
    gl = None
    if args.renderer != "pil":
        gl = try_create_gl(args, an, pal, W, H, strict=(args.renderer == "gl"))
    post = r = None
    if gl:
        print("Renderer: OpenGL on", gl.gpu_name)
        frame_bytes = gl.render_bytes
    else:
        gpu = init_gpu(args.gpu)
        r = Renderer(args, an, pal, W, H)
        if gpu:
            torch, dev, gname = gpu
            post = GPUPost(torch, dev, r.bg, W, H, args.aa, args.glow, args.aberration)
            print("Renderer: PIL drawing + PyTorch post-processing on", gname)
            frame_bytes = lambda j: post.process(*r.layer(j))
        else:
            print("Renderer: PIL (CPU)")
            frame_bytes = lambda j: r.render(j).tobytes()

    if args.transition_in > 0 or args.transition_out > 0:
        raw_frame_bytes = frame_bytes
        frame_bytes = lambda j: apply_transition(raw_frame_bytes(j), W, H, j, an.nf, args.fps, args)

    if args.preview is not None:
        target = min(int(args.preview * args.fps), an.nf - 1)
        data = None
        for j in range(max(0, target - args.fps), target + 1):  # warm up trails
            data = frame_bytes(j)
        out = args.output or base + "_preview.png"
        Image.frombytes("RGB", (W, H), data).save(out)
        print("Saved", out)
        return

    out = args.output or f"{base}_viz.mp4"
    vargs, vname = video_codec_args(args.encoder, args.crf, args.x264_preset, args.fps)
    print("Video encoder:", vname)
    cmd = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(args.fps), "-i", "-"]
    if args.start:
        cmd += ["-ss", str(args.start)]
    if args.duration:
        cmd += ["-t", str(args.duration)]
    cmd += ["-i", args.input, "-map", "0:v", "-map", "1:a:0"] + vargs + ["-pix_fmt", "yuv420p"] \
        + pick_audio_codec() + ["-shortest", out]
    cmd[0] = FFMPEG_BIN
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    t0 = time.time()

    def progress(i):
        if i % 15 == 0 or i == an.nf - 1:
            el = time.time() - t0
            eta = el / (i + 1) * (an.nf - i - 1)
            print(f"\rRendering {i + 1}/{an.nf} ({(i + 1) / an.nf * 100:5.1f}%)  "
                  f"{(i + 1) / max(el, 1e-6):.1f} fps  ETA {eta:5.0f}s ", end="", flush=True)

    try:
        if gl or post:
            # Producer renders frame N+1 while a writer thread pipes frame N into ffmpeg.
            # For the PIL+torch path the producer only draws and the thread does GPU post.
            q, errs = queue.Queue(maxsize=12), []

            def writer():
                dead = False
                while True:
                    item = q.get()
                    if item is None:
                        return
                    if dead:
                        continue
                    try:
                        proc.stdin.write(item if isinstance(item, (bytes, bytearray)) else post.process(*item))
                    except Exception as e:  # keep draining so the producer never blocks
                        errs.append(e)
                        dead = True

            th = threading.Thread(target=writer, daemon=True)
            th.start()
            for i in range(an.nf):
                if errs:
                    break
                q.put(gl.render_bytes(i) if gl else r.layer(i))
                progress(i)
            q.put(None)
            th.join()
            if errs:
                raise errs[0]
        else:
            for i in range(an.nf):
                proc.stdin.write(frame_bytes(i))
                progress(i)
        proc.stdin.close()
        proc.wait()
    except KeyboardInterrupt:
        proc.kill()
        sys.exit("\nStopped.")
    except (BrokenPipeError, OSError):
        proc.kill()
        sys.exit("\nffmpeg stopped unexpectedly (see its error message above). "
                 "Updating ffmpeg usually fixes this; or try --encoder cpu.")
    except RuntimeError as e:
        proc.kill()
        sys.exit(f"\nGPU error: {e}\nTry --renderer pil, or a lower --res.")
    print(f"\nDone -> {out}")


def launch_gui():
    """Small Tk GUI; the renderer itself remains the same tested CLI pipeline."""
    import tkinter as tk
    from tkinter import ttk, filedialog, colorchooser, messagebox

    root = tk.Tk()
    root.title("Audio Visualizer")
    root.geometry("900x820")
    root.minsize(620, 480)
    style = ttk.Style(root)
    try:
        style.theme_use("vista")
    except tk.TclError:
        pass

    audio_var, output_var = tk.StringVar(), tk.StringVar()
    preset_var, palette_var = tk.StringVar(value="radial"), tk.StringVar(value="neon")
    res_var, fps_var = tk.StringVar(value="720p"), tk.StringVar(value="30")
    renderer_var, gpu_var = tk.StringVar(value="auto"), tk.StringVar(value="auto")
    encoder_var = tk.StringVar(value="auto")
    bg_var = tk.StringVar(value="#080014")
    color_vars = [tk.StringVar(value=PALETTES["neon"][min(i, len(PALETTES["neon"])-1)]) for i in range(12)]
    transition_var = tk.BooleanVar(value=False)
    transition_in_var, transition_out_var = tk.StringVar(value="1.5"), tk.StringVar(value="1.5")
    transition_style_var = tk.StringVar(value="fade_blur")
    transition_blur_var = tk.StringVar(value="18")
    status_var = tk.StringVar(value="Choose an audio file, select one fixed pattern, then render.")
    process = {"p": None}

    shell = ttk.Frame(root)
    shell.pack(fill="both", expand=True)
    canvas = tk.Canvas(shell, highlightthickness=0)
    scrollbar = ttk.Scrollbar(shell, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=scrollbar.set)
    scrollbar.pack(side="right", fill="y")
    canvas.pack(side="left", fill="both", expand=True)
    outer = ttk.Frame(canvas, padding=16)
    canvas_window = canvas.create_window((0, 0), window=outer, anchor="nw")
    outer.columnconfigure(1, weight=1)

    def update_scroll_region(_event=None):
        canvas.configure(scrollregion=canvas.bbox("all"))
    def fit_canvas_width(event):
        canvas.itemconfigure(canvas_window, width=event.width)
    outer.bind("<Configure>", update_scroll_region)
    canvas.bind("<Configure>", fit_canvas_width)
    def mousewheel(event):
        canvas.yview_scroll(int(-event.delta / 120), "units")
    canvas.bind_all("<MouseWheel>", mousewheel)

    def add_row(label, widget, row):
        ttk.Label(outer, text=label).grid(row=row, column=0, sticky="w", padx=(0, 12), pady=6)
        widget.grid(row=row, column=1, sticky="ew", pady=6)

    file_frame = ttk.Frame(outer)
    file_frame.columnconfigure(0, weight=1)
    ttk.Entry(file_frame, textvariable=audio_var).grid(row=0, column=0, sticky="ew")
    def choose_audio():
        p = filedialog.askopenfilename(filetypes=[("Audio", "*.mp3 *.wav *.flac *.m4a *.aac *.ogg"), ("All files", "*.*")])
        if p:
            audio_var.set(p)
            if not output_var.get():
                output_var.set(os.path.splitext(p)[0] + "_viz.mp4")
    ttk.Button(file_frame, text="Browse…", command=choose_audio).grid(row=0, column=1, padx=(8, 0))
    add_row("Audio file", file_frame, 0)

    preset_box = ttk.Combobox(outer, textvariable=preset_var, values=list(PRESETS), state="readonly")
    add_row("Pattern (stays fixed)", preset_box, 1)
    palette_box = ttk.Combobox(outer, textvariable=palette_var, values=list(PALETTES), state="readonly")
    add_row("Palette preset", palette_box, 2)

    colors_frame = ttk.LabelFrame(outer, text="Custom visualizer colors — each stop: HEX or R,G,B")
    colors_frame.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(10, 6))
    for i in range(12):
        group = i % 3
        col = group * 3
        row = i // 3
        colors_frame.columnconfigure(col + 1, weight=1)
        ttk.Label(colors_frame, text=f"Color {i + 1}").grid(row=row, column=col, padx=(8, 4), pady=5, sticky="w")
        ttk.Entry(colors_frame, textvariable=color_vars[i], width=12).grid(row=row, column=col + 1, padx=(0, 5), pady=5, sticky="ew")
        def pick_color(var=color_vars[i]):
            try:
                current = "#%02x%02x%02x" % hex_to_rgb(var.get())
            except Exception:
                current = "#ffffff"
            chosen = colorchooser.askcolor(color=current, parent=root)[1]
            if chosen:
                var.set(chosen)
        ttk.Button(colors_frame, text="Pick", command=pick_color, width=5).grid(row=row, column=col + 2, padx=(0, 7), pady=5)

    def load_palette(_event=None):
        stops = PALETTES[palette_var.get()]
        rgb = np.asarray([hex_to_rgb(c) for c in stops], dtype=float)
        for i, var in enumerate(color_vars):
            pos = i / max(1, len(color_vars) - 1) * (len(rgb) - 1)
            lo = int(pos)
            hi = min(lo + 1, len(rgb) - 1)
            mix = pos - lo
            color = np.rint(rgb[lo] * (1.0 - mix) + rgb[hi] * mix).astype(int)
            var.set("#%02x%02x%02x" % tuple(color))
    palette_box.bind("<<ComboboxSelected>>", load_palette)
    load_palette()

    bg_frame = ttk.Frame(outer)
    bg_frame.columnconfigure(0, weight=1)
    ttk.Entry(bg_frame, textvariable=bg_var).grid(row=0, column=0, sticky="ew")
    def pick_bg():
        try:
            current = "#%02x%02x%02x" % hex_to_rgb(bg_var.get())
        except Exception:
            current = "#080014"
        chosen = colorchooser.askcolor(color=current, parent=root)[1]
        if chosen:
            bg_var.set(chosen)
    ttk.Button(bg_frame, text="Pick…", command=pick_bg).grid(row=0, column=1, padx=(8, 0))
    add_row("Background (HEX or R,G,B)", bg_frame, 4)

    settings = ttk.Frame(outer)
    for i in range(5):
        settings.columnconfigure(i, weight=1)
    for col, label in enumerate(("Resolution", "FPS", "Renderer", "GPU post", "Encoder")):
        ttk.Label(settings, text=label).grid(row=0, column=col, sticky="w")
    ttk.Combobox(settings, textvariable=res_var, values=list(RES), state="readonly").grid(row=1, column=0, sticky="ew", padx=(0, 8))
    ttk.Combobox(settings, textvariable=fps_var, values=["24", "30", "60", "120", "240"], state="readonly").grid(row=1, column=1, sticky="ew", padx=(0, 8))
    ttk.Combobox(settings, textvariable=renderer_var, values=["auto", "gl", "pil"], state="readonly").grid(row=1, column=2, sticky="ew", padx=(0, 8))
    ttk.Combobox(settings, textvariable=gpu_var, values=["auto", "on", "off"], state="readonly").grid(row=1, column=3, sticky="ew", padx=(0, 8))
    ttk.Combobox(settings, textvariable=encoder_var, values=["auto", "qsv", "cpu", "nvenc", "amf"], state="readonly").grid(row=1, column=4, sticky="ew")
    settings.grid(row=5, column=0, columnspan=2, sticky="ew", pady=10)

    transition_frame = ttk.LabelFrame(outer, text="Intro / outro transitions")
    transition_frame.grid(row=6, column=0, columnspan=2, sticky="ew", pady=(4, 8))
    ttk.Checkbutton(transition_frame, text="Enable transitions", variable=transition_var).grid(row=0, column=0, padx=8, pady=7, sticky="w")
    ttk.Label(transition_frame, text="Intro (s)").grid(row=0, column=1, padx=(8, 3))
    ttk.Entry(transition_frame, textvariable=transition_in_var, width=6).grid(row=0, column=2)
    ttk.Label(transition_frame, text="Outro (s)").grid(row=0, column=3, padx=(8, 3))
    ttk.Entry(transition_frame, textvariable=transition_out_var, width=6).grid(row=0, column=4)
    ttk.Label(transition_frame, text="Effect").grid(row=1, column=0, padx=8, pady=5, sticky="w")
    ttk.Combobox(transition_frame, textvariable=transition_style_var,
                 values=["fade_blur", "fade", "blur", "wipe", "zoom"], state="readonly", width=13).grid(row=1, column=1, columnspan=2, sticky="ew")
    ttk.Label(transition_frame, text="Blur radius").grid(row=1, column=3, padx=(8, 3))
    ttk.Entry(transition_frame, textvariable=transition_blur_var, width=6).grid(row=1, column=4)

    out_frame = ttk.Frame(outer)
    out_frame.columnconfigure(0, weight=1)
    ttk.Entry(out_frame, textvariable=output_var).grid(row=0, column=0, sticky="ew")
    def choose_output():
        p = filedialog.asksaveasfilename(defaultextension=".mp4", filetypes=[("MP4 video", "*.mp4")])
        if p:
            output_var.set(p)
    ttk.Button(out_frame, text="Save as…", command=choose_output).grid(row=0, column=1, padx=(8, 0))
    add_row("Output video", out_frame, 7)

    log = tk.Text(outer, height=8, wrap="word", state="disabled")
    log.grid(row=8, column=0, columnspan=2, sticky="nsew", pady=(8, 4))
    outer.rowconfigure(8, weight=1)
    ttk.Label(outer, textvariable=status_var).grid(row=9, column=0, columnspan=2, sticky="w", pady=4)

    def append_log(line):
        log.configure(state="normal")
        log.insert("end", line)
        log.see("end")
        log.configure(state="disabled")

    def render():
        if process["p"] and process["p"].poll() is None:
            return
        audio = audio_var.get().strip()
        if not audio or not os.path.isfile(audio):
            messagebox.showerror("Missing audio", "Choose a valid audio file first.", parent=root)
            return
        try:
            colors = ["#%02x%02x%02x" % hex_to_rgb(v.get()) for v in color_vars]
            bg = "#%02x%02x%02x" % hex_to_rgb(bg_var.get())
            fps = int(fps_var.get())
            transition_in = float(transition_in_var.get()) if transition_var.get() else 0.0
            transition_out = float(transition_out_var.get()) if transition_var.get() else 0.0
            transition_blur = float(transition_blur_var.get())
        except Exception as exc:
            messagebox.showerror("Invalid color or setting", str(exc), parent=root)
            return
        output = output_var.get().strip() or os.path.splitext(audio)[0] + "_viz.mp4"
        output_var.set(output)
        # In a PyInstaller build, sys.executable is the app itself; do not pass the
        # extracted .py path as an extra CLI argument. In source mode, invoke the script.
        cmd = [sys.executable] if getattr(sys, "frozen", False) else [sys.executable, os.path.abspath(__file__)]
        cmd += [audio, "--preset", preset_var.get(),
               "--colors", ",".join(colors), "--bg", bg, "--res", res_var.get(), "--fps", str(fps),
               "--renderer", renderer_var.get(), "--gpu", gpu_var.get(), "--encoder", encoder_var.get(),
               "--transition-in", str(transition_in), "--transition-out", str(transition_out),
               "--transition-style", transition_style_var.get(), "--transition-blur", str(transition_blur),
               "--output", output]
        append_log("\n> " + " ".join(cmd) + "\n")
        status_var.set("Rendering… keep this window open.")
        render_button.configure(state="disabled")
        try:
            p = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            process["p"] = p
        except Exception as exc:
            render_button.configure(state="normal")
            messagebox.showerror("Could not start renderer", str(exc), parent=root)
            return
        def read_output():
            for line in p.stdout:
                root.after(0, append_log, line)
            code = p.wait()
            def finished():
                render_button.configure(state="normal")
                status_var.set("Finished successfully: " + output if code == 0 else f"Render failed (exit code {code}). See log above.")
            root.after(0, finished)
        threading.Thread(target=read_output, daemon=True).start()

    render_button = ttk.Button(outer, text="Render video", command=render)
    render_button.grid(row=10, column=0, columnspan=2, sticky="ew", pady=(8, 0), ipady=5)
    root.mainloop()


if __name__ == "__main__":
    if len(sys.argv) == 1:
        launch_gui()
    else:
        main()
