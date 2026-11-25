#!/usr/bin/env python3
"""Erklärvideo "Das Atomkraftwerk" (Druckwasserreaktor) – 1080p, 30 fps, MP4 (H.264).

Aufruf:
  python3 make_video.py                 -> atomkraftwerk.mp4 (neben dem Skript)
  python3 make_video.py --stills 2 12   -> PNG-Einzelbilder zum Testen (Sekunden)
"""
import math
import os
import subprocess
import sys
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H, S, FPS = 1920, 1080, 2, 30      # S = Supersampling-Faktor (Kantenglättung)
TOTAL = 56.0
NF = int(TOTAL * FPS)
HERE = os.path.dirname(os.path.abspath(__file__))

FB = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FR = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

WHITE = (255, 255, 255)
ACCENT = (255, 196, 48)
BGDIM = (14, 34, 58)
HOT = (255, 90, 60)
COLD_W = (95, 170, 245)
STEAM = (228, 240, 252)


# ----------------------------------------------------------------- helpers
def lerp(a, b, t):
    return a + (b - a) * t


def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def smooth(x):
    x = clamp(x)
    return x * x * (3 - 2 * x)


def ramp(t, t0, t1):
    return smooth((t - t0) / (t1 - t0))


def mix(c1, c2, t):
    t = clamp(t)
    return tuple(int(lerp(a, b, t)) for a, b in zip(c1, c2))


def dm(col, a):
    """Farbe abdunkeln, wenn ein Bauteil gerade nicht im Fokus ist."""
    return mix(col, BGDIM, (1 - a) * 0.7)


_fonts = {}


def font(size, bold=False):
    k = (size, bold)
    if k not in _fonts:
        _fonts[k] = ImageFont.truetype(FB if bold else FR, int(size * S))
    return _fonts[k]


def tw(text, size, bold=False):
    return font(size, bold).getlength(text) / S


def wrap(text, size, maxw, bold=False):
    lines, cur = [], ""
    for w in text.split():
        trial = (cur + " " + w).strip()
        if tw(trial, size, bold) <= maxw:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


_bg = None


def bg():
    global _bg
    if _bg is None:
        h, w = H * S, W * S
        top = np.array([8, 20, 38], dtype=np.float32)
        bot = np.array([18, 48, 80], dtype=np.float32)
        g = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
        arr = (top + (bot - top) * g) * np.ones((1, w, 1), dtype=np.float32)
        _bg = Image.fromarray(arr.astype("uint8"), "RGB")
    return _bg


class Canvas:
    def __init__(self):
        self.im = bg().copy()
        self.d = ImageDraw.Draw(self.im, "RGBA")

    def circle(self, x, y, r, fill=None, outline=None, w=0):
        self.d.ellipse([(x - r) * S, (y - r) * S, (x + r) * S, (y + r) * S],
                       fill=fill, outline=outline, width=int(w * S))

    def ellipse(self, cx, cy, rx, ry, fill=None, outline=None, w=0):
        self.d.ellipse([(cx - rx) * S, (cy - ry) * S, (cx + rx) * S, (cy + ry) * S],
                       fill=fill, outline=outline, width=int(w * S))

    def rect(self, x0, y0, x1, y1, fill=None, outline=None, w=0, r=0, corners=None):
        box = [x0 * S, y0 * S, x1 * S, y1 * S]
        if r:
            kw = {}
            if corners is not None:
                kw["corners"] = corners
            self.d.rounded_rectangle(box, radius=r * S, fill=fill, outline=outline,
                                     width=int(w * S), **kw)
        else:
            self.d.rectangle(box, fill=fill, outline=outline, width=int(w * S))

    def poly(self, pts, fill=None, outline=None):
        self.d.polygon([(x * S, y * S) for x, y in pts], fill=fill, outline=outline)

    def line(self, pts, color, w):
        P = [(x * S, y * S) for x, y in pts]
        self.d.line(P, fill=color, width=max(1, int(w * S)), joint="curve")
        r = w * S / 2
        for (x, y) in (P[0], P[-1]):
            self.d.ellipse([x - r, y - r, x + r, y + r], fill=color)

    def text(self, x, y, t, size, color=WHITE, bold=False, anchor="la"):
        self.d.text((x * S, y * S), t, font=font(size, bold), fill=color, anchor=anchor)

    def finish(self):
        return self.im.reduce(S).tobytes()


class Path:
    def __init__(self, pts):
        self.pts = pts
        self.cum = [0.0]
        for a, b in zip(pts, pts[1:]):
            self.cum.append(self.cum[-1] + math.dist(a, b))
        self.L = self.cum[-1]

    def at(self, d):
        d %= self.L
        for i in range(len(self.pts) - 1):
            if d <= self.cum[i + 1]:
                f = (d - self.cum[i]) / (self.cum[i + 1] - self.cum[i])
                a, b = self.pts[i], self.pts[i + 1]
                return (a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f)
        return self.pts[-1]


def flow(c, path, a, colfn, t, speed=100, spacing=48, wall=26, pipe=True):
    if pipe:
        c.line(path.pts, mix((40, 52, 70), (125, 145, 175), a), wall)
        c.line(path.pts, mix((12, 24, 40), (18, 34, 56), a), wall - 8)
    n = int(path.L / spacing)
    for i in range(n):
        d = (i * spacing + speed * t) % path.L
        x, y = path.at(d)
        col, r = colfn(d)
        c.circle(x, y, r, fill=dm(col, a))


# ------------------------------------------------------------ Bausteine
def header(c, step=None, title="So funktioniert ein Atomkraftwerk"):
    c.text(60, 34, title, 40, WHITE, True)
    if step is not None:
        labels = ["Spaltung", "Wärme", "Dampf", "Strom", "Kühlung", "Kreislauf"]
        xs = [1030 + 148 * i for i in range(6)]
        c.line([(xs[0], 54), (xs[-1], 54)], (60, 80, 110), 4)
        c.line([(xs[0], 54), (xs[step - 1], 54)], ACCENT, 4)
        for i, x in enumerate(xs):
            on, cur = i < step, i == step - 1
            c.circle(x, 54, 13 if cur else 9, fill=ACCENT if on else (60, 80, 110))
            c.text(x, 76, labels[i], 20, WHITE if cur else (140, 160, 185), cur, "ma")


def caption(c, n, text, alpha=1.0):
    al = int(255 * alpha)
    c.rect(50, 915, 1870, 1050, fill=(5, 12, 24, int(220 * alpha)),
           outline=(70, 100, 140, al), w=2, r=22)
    c.circle(118, 982, 40, fill=ACCENT + (al,))
    c.text(118, 982, str(n), 44, (20, 30, 45, al), True, "mm")
    lines = wrap(text, 32, 1560)
    lh = 42
    y0 = 982 - lh * len(lines) / 2 + lh / 2
    for k, l in enumerate(lines):
        c.text(190, y0 + k * lh, l, 32, (240, 246, 255, al), False, "lm")


def nucleus(c, cx, cy, R, n, rn, glow=0.0, spin=0.0, alpha=255):
    if glow > 0:
        for k in range(5):
            c.circle(cx, cy, R * (1.9 - k * 0.18), fill=(255, 170, 60, int(glow * 18)))
    for k in range(n):
        rr = (R - rn) * math.sqrt((k + 0.5) / n)
        a = k * 2.399963 + spin
        x, y = cx + rr * math.cos(a), cy + rr * math.sin(a)
        col = (225, 70, 70) if k % 2 == 0 else (90, 150, 235)
        c.circle(x, y, rn, fill=col + (alpha,), outline=(255, 255, 255, 70), w=1)


def neutron(c, x, y, r=9, trail=None):
    if trail:
        c.line([trail, (x, y)], (230, 235, 245, 70), r * 0.9)
    c.circle(x, y, r + 5, fill=(255, 255, 255, 45))
    c.circle(x, y, r, fill=(236, 240, 246), outline=(120, 130, 150), w=1.5)


def atom_icon(c, cx, cy, R, t):
    nucleus(c, cx, cy, R * 0.22, 7, R * 0.07)
    for k in range(3):
        ang = k * math.pi / 3
        pts = []
        for j in range(61):
            th = j / 60 * 2 * math.pi
            x, y = R * math.cos(th), R * 0.38 * math.sin(th)
            pts.append((cx + x * math.cos(ang) - y * math.sin(ang),
                        cy + x * math.sin(ang) + y * math.cos(ang)))
        c.line(pts, (90, 160, 230), 4)
        th = t * (1.6 + 0.4 * k) + k * 2
        x, y = R * math.cos(th), R * 0.38 * math.sin(th)
        c.circle(cx + x * math.cos(ang) - y * math.sin(ang),
                 cy + x * math.sin(ang) + y * math.cos(ang), 10, fill=ACCENT)


def tower(c, cx, ytop, ybot, hww, hwb, col, shade, yw_frac=0.3):
    yw = ytop + yw_frac * (ybot - ytop)
    k = (hwb - hww) / ((ybot - yw) ** 2)
    ys = [ytop + (ybot - ytop) * j / 40 for j in range(41)]
    hw = lambda y: hww + k * (y - yw) ** 2
    left = [(cx - hw(y), y) for y in ys]
    right = [(cx + hw(y), y) for y in ys]
    c.poly(left + right[::-1], fill=col)
    c.poly([(cx + hw(y) * 0.3, y) for y in ys] + right[::-1], fill=shade)
    for j in range(1, 9):
        y = ytop + (ybot - ytop) * j / 9
        c.line([(cx - hw(y), y), (cx + hw(y), y)], (0, 0, 0, 28), 1.5)
    c.ellipse(cx, ytop, hw(ytop), 9, fill=mix(col, (255, 255, 255), 0.12))


def plume(c, cx, ytop, size, amt, t, drift=60):
    N = 16
    for i in range(N):
        age = (t * 0.16 + i / N) % 1.0
        ph = i * 2.399
        x = cx + drift * age + 14 * math.sin(ph + t * 0.9) * age + 10 * math.sin(ph)
        y = ytop - age * size * 2.4
        r = size * (0.28 + 0.7 * age)
        al = int(150 * amt * (1 - age) * min(1, age * 7))
        c.circle(x, y, r, fill=(232, 240, 250, al))


def pump(c, x, y, a):
    c.circle(x, y, 20, fill=dm((90, 105, 125), a), outline=dm((190, 205, 225), a), w=3)
    c.poly([(x - 8, y - 9), (x - 8, y + 9), (x + 10, y)], fill=dm((230, 235, 245), a))


# ------------------------------------------------------------- Szene 1: Titel
def scene_title(c, t):
    for k in range(8):
        c.circle(1450, 700, 700 - k * 70, fill=(40, 90, 150, 10))
    tower(c, 1330, 520, 1100, 130, 230, (34, 52, 78), (26, 40, 62))
    tower(c, 1680, 430, 1100, 170, 290, (46, 66, 96), (34, 50, 76))
    plume(c, 1680, 430, 90, 1.0, t, 90)
    plume(c, 1330, 520, 70, 0.8, t + 3, 70)
    atom_icon(c, 300, 250, 120, t)
    c.text(120, 420, "Das Atomkraftwerk", 92, WHITE + (int(255 * ramp(t, .3, 1.3)),), True)
    c.text(124, 550, "Vom Urankern zum Strom", 52, ACCENT + (int(255 * ramp(t, 1.0, 2.0)),))
    c.text(124, 640, "Ein Erklärvideo", 34, (170, 195, 225, int(255 * ramp(t, 1.8, 2.8))))


# -------------------------------------------------- Szene 2a: Kernspaltung
def scene_fission(c, u):
    header(c, 1)
    cx, cy = 1000, 540
    for k in range(7):
        c.circle(cx, cy, 480 - k * 60, fill=(50, 100, 160, 9))
    split = 3.2
    hit = 2.4
    if u < split:
        amp = 0 if u < hit else 2 + 10 * (u - hit) / (split - hit)
        ox = amp * math.sin(u * 60)
        oy = amp * math.cos(u * 53)
        nucleus(c, cx + ox, cy + oy, 90, 26, 15, glow=0 if u < hit else (u - hit) / (split - hit) * 1.5)
        c.text(cx, cy + 150, "Uran-235-Kern", 28, (200, 220, 245), True, "ma")
        if u < hit:
            nx = 250 + (cx - 90 - 12 - 250) * (u / hit)
            neutron(c, nx, cy, 12, trail=(max(150, nx - 90), cy))
            c.text(nx, cy + 36, "Neutron", 24, (220, 230, 245), False, "ma")
    else:
        v = u - split
        # Blitz und Wärmewellen
        c.circle(cx, cy, 40 + 700 * v, fill=(255, 235, 170, int(160 * (1 - ramp(v, 0, .55)))))
        for k in range(3):
            vv = v - k * 0.28
            if vv > 0:
                r = 420 * vv
                al = int(200 * (1 - clamp(vv / 1.8)))
                if al > 0:
                    c.circle(cx, cy, r, outline=(255, 140, 50, al), w=6)
        dist = v * 120
        a1, a2 = math.radians(20), math.radians(200)
        bx, by = cx + dist * math.cos(a1) + 20, cy + dist * math.sin(a1)
        kx, ky = cx + dist * math.cos(a2) - 20, cy + dist * math.sin(a2)
        nucleus(c, bx, by, 68, 16, 14.7, glow=max(0, 1.2 - v * .5))
        nucleus(c, kx, ky, 52, 10, 14.2, glow=max(0, 1.2 - v * .5))
        c.text(bx, by + 88, "Barium-Kern", 24, (200, 220, 245), True, "ma")
        c.text(kx, ky + 72, "Krypton-Kern", 24, (200, 220, 245), True, "ma")
        for ang in (-80, 100, -10):
            d = 330 * v
            nx = cx + d * math.cos(math.radians(ang))
            ny = cy + d * math.sin(math.radians(ang))
            neutron(c, nx, ny, 11, trail=(cx + max(0, d - 70) * math.cos(math.radians(ang)),
                                          cy + max(0, d - 70) * math.sin(math.radians(ang))))
        al = int(255 * ramp(v, 0.3, 0.9))
        c.text(cx, 800, "+ Wärme (Energie)", 36, (255, 160, 70, al), True, "mm")
        c.text(cx, 850, "+ 2–3 neue Neutronen", 30, (235, 240, 250, al), True, "mm")
    if u < hit:
        txt = "Ein Neutron fliegt auf einen Urankern (Uran-235) zu."
    elif u < split:
        txt = "Der Urankern nimmt das Neutron auf und wird instabil."
    else:
        txt = "Er spaltet sich in zwei kleinere Kerne. Dabei werden viel Wärme und neue Neutronen frei."
    start = 0 if u < hit else (hit if u < split else split)
    caption(c, 1, txt, ramp(u, start, start + 0.4) if u > 0.5 else 1.0)


# ------------------------------------------------ Szene 2b: Kettenreaktion
COLX = [330 + 250 * i for i in range(6)]
ROWY = [380, 570, 760]
GEN = 0.8
T0 = 1.0
ROD_X = COLX[2] + 125


def fired_nodes():
    fired = {0: {1}}
    for col in range(0, 2):
        nxt = set()
        for r in fired[col]:
            for r2 in (r - 1, r, r + 1):
                if 0 <= r2 < 3:
                    nxt.add(r2)
        fired[col + 1] = nxt
    return fired


FIRED = fired_nodes()


def scene_chain(c, u):
    header(c, 1)
    for k in range(7):
        c.circle(1000, 570, 620 - k * 80, fill=(50, 100, 160, 8))
    # Steuerstab
    rod_b = 150 + 750 * ramp(u, 2.1, 2.9)
    # Knoten
    for col in range(6):
        for r in range(3):
            x, y = COLX[col], ROWY[r]
            fired = r in FIRED.get(col, ())
            tf = T0 + GEN * col
            if fired and u >= tf:
                v = u - tf
                c.circle(x, y, 30 + 140 * v, fill=(255, 235, 170, int(170 * (1 - ramp(v, 0, .5)))))
                c.circle(x, y, 78, fill=(255, 150, 50, int(55 * max(.25, 1 - v * .4))))
                sep = min(30, 8 + 40 * v)
                nucleus(c, x - sep, y - 6, 22, 5, 8.5)
                nucleus(c, x + sep, y + 6, 22, 5, 8.5)
            else:
                pulse = 1 + 0.03 * math.sin(u * 6 + col + r)
                nucleus(c, x, y, 38 * pulse, 12, 9.5)
    # Neutronen
    n0 = ramp(u, 0.1, T0)
    if u < T0:
        sx, sy = 80, ROWY[1]
        nx = lerp(sx, COLX[0], (u - 0.1) / (T0 - 0.1))
        neutron(c, nx, sy, 9, trail=(max(40, nx - 70), sy))
        c.text(120, ROWY[1] + 36, "Neutron", 22, (220, 230, 245), False, "mm")
    for col in range(0, 3):
        for r in FIRED.get(col, ()):
            for r2 in (r - 1, r, r + 1):
                if not 0 <= r2 < 3:
                    continue
                tl = T0 + GEN * col
                p = (u - tl) / GEN
                if p < 0:
                    continue
                x0, y0, x1, y1 = COLX[col], ROWY[r], COLX[col + 1], ROWY[r2]
                pmax = 1.0
                if col == 2:
                    pmax = (ROD_X - 13 - x0) / (x1 - x0)
                if p <= pmax:
                    pp = clamp(p)
                    x, y = lerp(x0, x1, pp), lerp(y0, y1, pp)
                    q = max(0, pp - 0.12)
                    neutron(c, x, y, 8, trail=(lerp(x0, x1, q), lerp(y0, y1, q)))
                elif col == 2 and p - pmax < 0.45:
                    xa = ROD_X - 13
                    ya = lerp(y0, y1, pmax)
                    f = (p - pmax) / 0.45
                    c.circle(xa, ya, 8 + 30 * f, outline=(180, 190, 205, int(220 * (1 - f))), w=3)
    # Steuerstab zeichnen (über den Neutronen)
    if u > 2.0:
        c.rect(ROD_X - 13, 150, ROD_X + 13, rod_b, fill=(120, 130, 146), outline=(200, 210, 225), w=2, r=6)
        for yy in range(170, int(rod_b), 36):
            c.line([(ROD_X - 13, yy), (ROD_X + 13, yy + 12)], (60, 68, 82), 3)
        c.text(ROD_X, 128, "Steuerstab", 26, (255, 255, 255, int(255 * ramp(u, 2.1, 2.6))), True, "mm")
    c.text(COLX[0], 850, "Urankerne (Brennstoff)", 22, (190, 210, 235), False, "mm")
    if u < 2.4:
        caption(c, 1, "Die neuen Neutronen treffen weitere Urankerne – es entsteht eine Kettenreaktion.",
                1.0 if u < 0.5 else 1.0)
    else:
        caption(c, 1, "Steuerstäbe nehmen Neutronen auf. So wird die Reaktion geregelt oder gestoppt.",
                ramp(u, 2.4, 2.9))


# ---------------------------------------------- Szene 3: Gesamtschema
PH = [17, 23, 29, 35, 41]
GROUPS = ["containment", "reactor", "primary", "steamgen", "secondary", "turbine",
          "generator", "line", "condenser", "cooling", "tower"]


def _tgt(**kw):
    d = {g: 0.3 for g in GROUPS}
    d["containment"] = 0.55
    d.update(kw)
    return d


TARGETS = [
    _tgt(reactor=1, primary=1, containment=.8),
    _tgt(primary=.8, steamgen=1, secondary=.8, containment=.8),
    _tgt(secondary=.8, turbine=1, generator=1, line=1),
    _tgt(secondary=.6, condenser=1, cooling=1, tower=1),
    {g: 1.0 for g in GROUPS},
]
CAPTIONS = [
    "Im Reaktor erhitzt die Spaltung die Brennstäbe. Das Wasser im Primärkreislauf wird etwa 320 °C heiß – hoher Druck verhindert, dass es kocht.",
    "Im Dampferzeuger gibt das heiße Wasser seine Wärme an einen zweiten, getrennten Kreislauf ab. Dort entsteht Dampf.",
    "Der Dampf treibt die Turbine an. Der Generator wandelt die Drehung in elektrischen Strom um, der ins Stromnetz fließt.",
    "Im Kondensator wird der Dampf wieder zu Wasser. Der Kühlturm gibt die Restwärme an die Luft ab – die weiße Wolke ist nur Wasserdampf.",
    "Das Wasser fließt zurück zum Dampferzeuger und der Kreislauf beginnt von vorn – rund um die Uhr.",
]

PRIMARY = Path([(210, 720), (210, 450), (430, 450), (430, 720), (210, 720)])
SECOND = Path([(510, 600), (510, 330), (1030, 330), (1030, 850), (510, 850), (510, 600)])
COOLING = Path([(960, 610), (960, 580), (1290, 580), (1290, 760), (1420, 760),
                (1420, 830), (1230, 830), (1230, 610), (960, 610)])
LINE = Path([(1250, 330), (1760, 330)])


def galpha(t):
    i = max(0, sum(1 for p in PH if t >= p) - 1)
    cur = TARGETS[i]
    if i == 0:
        return dict(cur)
    w = ramp(t, PH[i], PH[i] + 0.8)
    return {g: lerp(TARGETS[i - 1][g], cur[g], w) for g in GROUPS}


def col_primary(d):
    f = d / PRIMARY.L
    if f < 0.2755:
        T = f / 0.2755
    elif f < 0.5:
        T = 1
    elif f < 0.7755:
        T = 1 - (f - 0.5) / 0.2755
    else:
        T = 0
    return mix(COLD_W, HOT, T), 6


def col_second(d):
    s = ramp(d, 60, 160) * (1 - ramp(d, 1000, 1100))
    return mix((80, 160, 245), STEAM, s), 6 + 2.5 * s


def col_cooling(d):
    if d < 540:
        T = 1
    elif d < 740:
        T = 1 - (d - 540) / 200
    elif d < 1150:
        T = 0
    else:
        T = (d - 1150) / 270
    return mix((90, 180, 255), (255, 170, 100), T), 5.5


def col_line(d):
    return (255, 225, 90), 5


def scene_schematic(c, t):
    A = galpha(t)
    pi = max(0, sum(1 for p in PH if t >= p) - 1)
    header(c, pi + 2)
    # Legende
    for x, col, txt in ((80, HOT, "heißes Wasser"), (360, STEAM, "Dampf"),
                        (520, (90, 170, 250), "kaltes Wasser"), (780, (255, 225, 90), "Strom")):
        c.circle(x, 180, 8, fill=col)
        c.text(x + 18, 180, txt, 22, (190, 210, 235), False, "lm")

    # Sicherheitsbehälter
    a = A["containment"]
    c.rect(70, 240, 590, 880, fill=dm((38, 52, 72), a) + (255,), outline=dm((150, 165, 185), a),
           w=8, r=140, corners=(True, True, False, False))
    c.text(330, 262, "Sicherheitsbehälter", 22, dm((200, 215, 235), max(a, .55)), False, "ma")

    # Reaktor
    a = A["reactor"]
    if a > 0.6:
        for k in range(3):
            c.circle(210, 595, 190 - k * 40, fill=(255, 140, 40, int(22 * a)))
    c.rect(140, 380, 280, 780, fill=dm((62, 74, 92), a), outline=dm((150, 165, 190), a), w=5, r=18)
    c.rect(150, 392, 270, 768, fill=dm((24, 70, 125), a), r=10)
    for x in (158, 176, 244, 262):
        glow = 0.5 + 0.5 * math.sin(t * 3 + x)
        c.rect(x - 5, 470, x + 5, 720, fill=dm(mix((200, 70, 30), (255, 190, 80), glow), a), r=3)
    ins = 100 + 90 * math.sin(t * 0.8)
    for x in (193, 227):
        c.rect(x - 4, 392, x + 4, 392 + ins, fill=dm((165, 172, 184), a), outline=dm((60, 66, 80), a), w=1)
    c.text(210, 345, "Reaktor", 22, dm((200, 220, 245), max(a, .5)), True, "ma")

    # Dampferzeuger
    a = A["steamgen"]
    c.rect(395, 380, 535, 780, fill=dm((62, 74, 92), a), outline=dm((150, 165, 190), a), w=5, r=18)
    c.rect(403, 388, 527, 772, fill=dm((24, 70, 125), a), r=10)
    c.rect(403, 388, 527, 520, fill=dm((110, 140, 175), a), r=10)
    c.text(495, 345, "Dampferzeuger", 22, dm((200, 220, 245), max(a, .5)), True, "ra")

    # Kühlturm (hinten)
    a = A["tower"]
    tower(c, 1500, 500, 880, 75, 140, dm((86, 100, 124), a), dm((62, 74, 96), a), 0.26)
    c.text(1500, 884, "Kühlturm", 22, dm((200, 220, 245), max(a, .5)), True, "ma")

    # Rohre + Teilchen
    flow(c, PRIMARY, max(A["primary"], A["reactor"] * .0 + A["primary"]), col_primary, t, speed=90)
    pump(c, 320, 720, A["primary"])
    flow(c, SECOND, A["secondary"], col_second, t, speed=110)
    pump(c, 760, 850, A["secondary"])

    # Turbine (über den Rohren: Dampf verschwindet im Gehäuse)
    a = A["turbine"]
    c.poly([(880, 312), (1080, 270), (1080, 390), (880, 348)],
           fill=dm((88, 100, 118), a), outline=dm((170, 185, 205), a))
    for i in range(5):
        x = 905 + i * 35
        h = 14 + i * 8
        c.line([(x, 330 - h), (x, 330 + h)], dm((60, 70, 86), a), 3)
        om = t * 5
        for k in range(6):
            y = 330 + h * math.sin(om + k * math.pi / 3 + i)
            c.line([(x - 9, y), (x + 9, y)], dm((200, 210, 225), a), 3)
    c.line([(1080, 330), (1151, 330)], dm((150, 160, 175), a), 10)
    c.text(980, 238, "Turbine", 22, dm((200, 220, 245), max(a, .5)), True, "ma")

    # Generator
    a = A["generator"]
    c.rect(1120, 280, 1250, 380, fill=dm((70, 95, 130), a), outline=dm((170, 190, 215), a), w=4, r=14)
    c.circle(1185, 330, 38, fill=dm((20, 30, 46), a), outline=dm((120, 140, 170), a), w=2)
    th = t * 5
    dx, dy = 32 * math.cos(th), 32 * math.sin(th)
    c.line([(1185, 330), (1185 + dx, 330 + dy)], dm((235, 70, 70), a), 16)
    c.line([(1185, 330), (1185 - dx, 330 - dy)], dm((80, 140, 240), a), 16)
    c.text(1185, 255, "Generator", 22, dm((200, 220, 245), max(a, .5)), True, "ma")

    # Kondensator
    a = A["condenser"]
    c.rect(930, 540, 1130, 640, fill=dm((62, 74, 92), a) + (110,), outline=dm((150, 165, 190), a), w=4, r=12)
    c.text(915, 590, "Kondensator", 22, dm((200, 220, 245), max(a, .5)), True, "rm")

    # Kühlwasser
    flow(c, COOLING, A["cooling"], col_cooling, t, speed=80, spacing=44, wall=16)

    # Stromleitung + Haus
    a = A["line"]
    for px in (1360, 1640):
        pc = dm((130, 142, 160), a)
        c.line([(px - 28, 430), (px, 338)], pc, 4)
        c.line([(px + 28, 430), (px, 338)], pc, 4)
        c.line([(px - 18, 392), (px + 18, 392)], pc, 3)
        c.line([(px - 22, 356), (px + 22, 356)], pc, 3)
    flow(c, LINE, a, col_line, t, speed=200, spacing=40, wall=6, pipe=False)
    c.line(LINE.pts, dm((150, 160, 175), a), 5)
    flow(c, LINE, a, col_line, t, speed=200, spacing=40, wall=6, pipe=False)
    c.poly([(1750, 330), (1805, 285), (1860, 330)], fill=dm((150, 70, 60), a))
    c.rect(1760, 330, 1850, 405, fill=dm((176, 164, 146), a))
    if a > 0.5:
        c.circle(1805, 368, 70, fill=(255, 220, 100, int(40 * a)))
    c.rect(1785, 348, 1825, 388, fill=dm(mix((70, 80, 95), (255, 225, 110), a), 1.0))
    c.text(1805, 420, "Haushalte", 22, dm((200, 220, 245), max(a, .5)), True, "ma")

    # Dampfwolke vorne
    plume(c, 1500, 500, 45, 0.9 * (0.35 + 0.65 * A["tower"]), t, 60)

    # Beschriftung/Untertitel
    idx = pi
    start = PH[idx] if idx > 0 else 17
    caption(c, idx + 2, CAPTIONS[idx], ramp(t, start, start + 0.4) if idx > 0 else 1.0)


# ----------------------------------------------------------- Szene 4: Fazit
CARDS = [
    ("Kernspaltung", "Beim Spalten von Uran-235 wird Wärme frei. Steuerstäbe regeln die Kettenreaktion."),
    ("Dampf & Turbine", "Das heiße Wasser erzeugt in einem zweiten Kreislauf Dampf, der die Turbine antreibt."),
    ("Strom", "Der Generator wandelt die Drehbewegung in elektrische Energie um."),
    ("Kühlung & Sicherheit", "Der Kühlturm gibt Restwärme ab. Mehrere Schutzbarrieren halten die Radioaktivität zurück."),
]
NOTE = ("Gut zu wissen: Im Betrieb setzt ein Atomkraftwerk kein CO₂ frei – dafür entsteht radioaktiver "
        "Abfall, der sehr lange sicher gelagert werden muss. In Deutschland wurden die letzten "
        "Kernkraftwerke im April 2023 abgeschaltet.")


def scene_outro(c, u):
    header(c, None, "Das Wichtigste in Kürze")
    for k, (title, body) in enumerate(CARDS):
        al = ramp(u, 0.3 + 0.8 * k, 0.8 + 0.8 * k)
        x0 = 100 + (k % 2) * 880
        y0 = 160 + (k // 2) * 270
        a255 = int(255 * al)
        c.rect(x0, y0, x0 + 840, y0 + 245, fill=(10, 24, 44, int(230 * al)),
               outline=(70, 110, 160, a255), w=2, r=24)
        c.circle(x0 + 62, y0 + 62, 36, fill=ACCENT + (a255,))
        c.text(x0 + 62, y0 + 62, str(k + 1), 40, (20, 30, 45, a255), True, "mm")
        c.text(x0 + 120, y0 + 62, title, 36, (255, 255, 255, a255), True, "lm")
        for j, l in enumerate(wrap(body, 28, 760)):
            c.text(x0 + 40, y0 + 130 + j * 40, l, 28, (215, 228, 245, a255))
    al = ramp(u, 4.0, 4.8)
    a255 = int(255 * al)
    c.rect(100, 725, 1820, 865, fill=(24, 44, 70, int(235 * al)), outline=(255, 196, 48, a255), w=2, r=24)
    for j, l in enumerate(wrap(NOTE, 28, 1640)):
        c.text(140, 760 + j * 42, l, 28, (240, 246, 255, a255))


# ------------------------------------------------------------------- Render
SCENES = [(0, 4), (4, 10), (10, 17), (17, 47), (47, 56)]


def draw(t):
    c = Canvas()
    if t < 4:
        scene_title(c, t)
    elif t < 10:
        scene_fission(c, t - 4)
    elif t < 17:
        scene_chain(c, t - 10)
    elif t < 47:
        scene_schematic(c, t)
    else:
        scene_outro(c, t - 47)
    return c


def render(i):
    t = i / FPS
    c = draw(t)
    # Überblendung zu Schwarz an Szenengrenzen
    dark = 0.0
    for s, e in SCENES:
        if s <= t < e or t == e:
            dark = max(dark, 1 - ramp(t, s, s + 0.4), ramp(t, e - 0.4, e))
    if dark > 0.003:
        c.rect(0, 0, W, H, fill=(0, 0, 0, int(255 * dark)))
    return c.finish()


# 8 Einzelclips (Name, Start, Ende in Sekunden des Gesamtvideos)
CLIPS = [
    ("01_kernspaltung", 4, 10),
    ("02_kettenreaktion", 10, 17),
    ("03_waerme_im_reaktor", 17.8, 23),
    ("04_dampferzeuger", 23.8, 29),
    ("05_strom_erzeugen", 29.8, 35),
    ("06_kuehlung", 35.8, 41),
    ("07_kreislauf", 41.8, 47),
    ("08_zusammenfassung", 47, 56),
]


def render_clip(arg):
    i, s, e = arg
    t = s + i / FPS
    c = draw(t)
    dark = max(1 - ramp(t, s, s + 0.4), ramp(t, e - 0.4, e))
    if dark > 0.003:
        c.rect(0, 0, W, H, fill=(0, 0, 0, int(255 * dark)))
    return c.finish()


def encode(out, frames_iter):
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
         "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
         "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-profile:v", "high",
         "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "96k", "-shortest",
         "-movflags", "+faststart", out],
        stdin=subprocess.PIPE)
    for b in frames_iter:
        ff.stdin.write(b)
    ff.stdin.close()
    ff.wait()


def main():
    if "--clips" in sys.argv:
        od = os.path.join(HERE, "clips")
        os.makedirs(od, exist_ok=True)
        with Pool(os.cpu_count() or 2) as p:
            for name, s, e in CLIPS:
                n = int(round((e - s) * FPS))
                encode(os.path.join(od, name + ".mp4"),
                       p.imap(render_clip, [(i, s, e) for i in range(n)], chunksize=4))
                print("fertig:", name, flush=True)
        return
    if "--stills" in sys.argv:
        secs = [float(x) for x in sys.argv[sys.argv.index("--stills") + 1:]]
        out = os.path.join(HERE, "stills")
        os.makedirs(out, exist_ok=True)
        for s in secs:
            b = render(int(s * FPS))
            Image.frombytes("RGB", (W, H), b).save(os.path.join(out, f"t{s:05.1f}.png"))
        return
    out = os.path.join(HERE, "atomkraftwerk.mp4")
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
         "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
         "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-profile:v", "high",
         "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "96k", "-shortest",
         "-movflags", "+faststart", out],
        stdin=subprocess.PIPE)
    with Pool(os.cpu_count() or 2) as p:
        for n, b in enumerate(p.imap(render, range(NF), chunksize=4)):
            ff.stdin.write(b)
            if n % 100 == 0:
                print(f"{n}/{NF}", flush=True)
    ff.stdin.close()
    ff.wait()
    print("fertig:", out)


if __name__ == "__main__":
    main()
