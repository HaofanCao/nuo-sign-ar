## Core Algorithm Design

This demo uses a lightweight rule-based recognizer on top of MediaPipe 21-point hand landmarks.

### 1) Inference Pipeline

`Frame -> Hand Landmarks (21x3) -> Feature Extraction -> Multi-gesture Scoring -> Decision Gate -> Temporal Smoothing -> AR Overlay`

### 2) Feature Extraction

Given normalized landmarks `lm`:

$$
\mathrm{palm\_size}
=
\max\!\Bigl(
d(\mathbf{l}_0,\mathbf{l}_9),\;
d(\mathbf{l}_5,\mathbf{l}_{17}),\;
10^{-6}
\Bigr)
$$

$$
\mathrm{finger\_ext}
=
\mathrm{clamp}\!\left(
\frac{y_{\mathrm{pip}}-y_{\mathrm{tip}}+0.01}{0.20},\;
0,\;
1
\right)
$$

$$
\mathrm{thumb\_ext}
=
\mathrm{clamp}\!\left(
\frac{
d(\mathbf{l}_{w},\mathbf{l}_{tt})-d(\mathbf{l}_{w},\mathbf{l}_{tj})
}{
0.30\cdot \mathrm{palm\_size}
},\;
0,\;
1
\right)
$$

where $\mathbf{l}_{w}$ is wrist, $\mathbf{l}_{tt}$ is thumb tip, and $\mathbf{l}_{tj}$ is thumb joint.

Core geometric features:

- `thumb_ext`, `index_ext`, `middle_ext`, `ring_ext`, `pinky_ext`
- `thumb_index` (thumb tip to index tip distance, normalized)
- `index_middle` (index tip to middle tip distance, normalized)
- `thumb_to_index_mcp` (thumb tip to index MCP distance, normalized)

### 3) Gesture Scoring (Weighted Rules)

For each gesture, score in $[0,1]$:

- `OPEN_PALM`: high extension across five fingers + finger spread
- `SWORD_SIGN`: index/middle extended, ring/pinky folded, thumb tucked
- `PINCH_SEAL`: thumb-index close (pinch), others mostly folded
- `FIST_GUARD`: all fingers folded

Implemented as weighted linear combinations in `hand_sign_ar/recognizer.py::_score_gestures`.

For example, the open-palm score is:

$$
s_{\mathrm{open}}
=
0.18\,e_{\mathrm{thumb}}
+
0.20\,e_{\mathrm{index}}
+
0.20\,e_{\mathrm{middle}}
+
0.20\,e_{\mathrm{ring}}
+
0.17\,e_{\mathrm{pinky}}
+
0.05\,s_{\mathrm{spread}}
$$

### 4) Decision Gate

Let `best` be the top score and `second` the second-highest:

$$
\hat{g}
=
\begin{cases}
\mathrm{UNKNOWN}, & s_{(1)} < \tau \\
\mathrm{UNKNOWN}, & s_{(1)} - s_{(2)} < m \\
g_{(1)}, & \text{otherwise}
\end{cases}
$$

where $\tau=\mathrm{threshold}$ and $m=\mathrm{margin}$.

This rejects low-confidence and ambiguous frames.

### 5) Temporal Smoothing

A sliding-window weighted vote is applied:

$$
w_i
=
0.65
+
0.35\cdot\frac{i+1}{n},
\quad
i=0,\ldots,n-1
$$

$$
\mathrm{stability}
=
\frac{
\sum\limits_{i:\, g_i=\hat{g}} w_i\,c_i
}{
\sum\limits_i w_i\,c_i
}
$$

This suppresses jitter and improves robustness in real-time webcam noise.

### 6) Runtime Confidence Fusion

Displayed confidence combines frame-level and temporal consistency.

For known gestures:

$$
\mathrm{conf}
=
\max\!\left(
c_{\mathrm{raw}},\;
0.75\cdot \mathrm{stability} + 0.25\cdot c_{\mathrm{raw}}
\right)
$$

For unknown gestures:

$$
\mathrm{conf}
=
\max\!\left(
c_{\mathrm{raw}},\;
0.60\cdot \mathrm{stability}
\right)
$$

### 7) Parameter-to-Logic Mapping

- `--threshold`: minimum accepted top score
- `--margin`: minimum gap between top-1 and top-2 scores
- `--smoothing`: temporal window size (larger = more stable, higher latency)
- `--min-detect`, `--min-track`: detector/tracker confidence in MediaPipe
