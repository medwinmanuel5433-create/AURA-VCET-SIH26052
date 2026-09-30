import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
INK, MUT, LINE, ACC, IN, OUT, BG, SURF = "#18211E", "#56625E", "#B9C3BE", "#A8620F", "#9B3B2B", "#1E6B61", "#FFFFFF", "#F4F7F5"
fig = plt.figure(figsize=(12, 16.03), dpi=150); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(8, 140); ax.axis("off")
fig.patch.set_facecolor(BG)
def box(x, y, w, h, title, lines=(), edge=INK, fill=SURF, tcol=None, lw=1.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.2", fc=fill, ec=edge, lw=lw))
    ax.text(x + 1.4, y + h - 1.9, title, fontsize=12.5, fontweight="bold", color=tcol or edge, va="top", family="DejaVu Sans")
    for i, l in enumerate(lines):
        ax.text(x + 1.4, y + h - 4.6 - i * 2.35, l, fontsize=9.6, color=INK, va="top", family="DejaVu Sans")
def arrow(x1, y1, x2, y2, col=INK):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", color=col, lw=1.6, mutation_scale=16))
def step(x, y, n):
    ax.text(x, y, str(n), fontsize=11, fontweight="bold", color="white", ha="center", va="center",
            bbox=dict(boxstyle="circle,pad=0.3", fc=ACC, ec=ACC))
ax.text(4, 136.5, "AURA  MARK7-DM v4  —  full pipeline", fontsize=20, fontweight="bold", color=INK, va="top")
ax.text(4, 132.6, "Real-time, frame by frame (8 ms hop, 16 ms lookahead). Two microphones in, clean loud voice out.", fontsize=11, color=MUT, va="top")
# inputs
box(6, 117, 40, 11, "Voice mic (boom)", ["your voice + the blast", "read RAW: gain is applied only after cleaning"], edge=IN)
box(54, 117, 40, 11, "Outer reference mic", ["the blast, voice 10–18 dB weaker", "close to the voice mic (hears the same blast)"], edge=IN)
arrow(26, 117, 26, 112.2); arrow(74, 117, 74, 112.2)
# STFT
step(3, 108.8, 1)
box(6, 104.5, 88, 7.3, "STFT", ["32 ms frames every 8 ms, 257 frequency bins, both mics  →  X (voice mic),  R (reference mic)"])
arrow(26, 104.5, 26, 99.3); arrow(74, 104.5, 74, 99.3)
# cues
step(3, 95.5, 2)
box(6, 88, 40, 11, "DSRA cues (8 per mic)", ["centroid, flatness, roll-off, low / speech / high", "band shares, loudness, loudness jump"], edge=ACC, tcol=ACC)
box(54, 88, 40, 11, "Speaker voiceprint", ["log-mel mean / std from frames that are", "blast-free (gate) AND voice (DSRA)"], edge=ACC, tcol=ACC)
arrow(26, 88, 38, 83.3); arrow(74, 88, 62, 83.3)
# network
step(3, 79, 3)
box(6, 63, 88, 20, "MARK7-DM v3 network  (MARK6.5 core, 1.7 M params, causal)",
    ["• sees X and R, plus 2 frames ahead (16 ms lookahead)",
     "• FiLM conditioning from the DSRA cues and the voiceprint: knows whose voice to keep",
     "• detector: speech / blast in every frame",
     "• deep filter on the voice mic + NEURAL CANCELLER: predicts the blast from R every frame, subtracts it",
     "• recoverability head: rebuilds voice where the blast destroyed it",
     "→  voice estimate  Y"], lw=2.2)
arrow(50, 63, 50, 58.3)
# reference post-filter
step(3, 54, 4)
box(6, 44, 88, 14, "Reference-mic residual post-filter",
    ["subband NLMS adaptive canceller on the network output Y: a 4-tap complex filter per bin",
     "maps the outer reference mic onto Y and estimates the residual blast",
     "adaptation is slowed where the network says voice is present",
     "→  attenuation-only gain  G = (|E| / |Y|)^0.5 ≤ 1"], edge=OUT, tcol=OUT, lw=2.2)
arrow(50, 44, 50, 37.3)
step(3, 34, 5)
box(6, 30.5, 88, 6.3, "Final clean-up", ["apply the gain to the voice estimate:  Z = G · Y   (never boosts, never adds noise)"])
arrow(27, 30.5, 27, 24.3)
step(3, 21, 6)
box(6, 17.5, 42, 6.3, "iSTFT", ["frames back to a waveform (overlap-add)"])
arrow(48, 20.6, 53.5, 20.6)
step(97, 20.6, 7)
box(54, 17.5, 40, 6.3, "Loudness  →  output", ["−18 LUFS + peak limiter (≈ +8 dB): clean, loud voice"], edge=OUT, tcol=OUT)
fig.savefig("/content/aura_home/deliver/AURA_MARK7_V4/pipeline_diagram.png", facecolor=BG)
print("saved")
