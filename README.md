# DistAE: Signal Distance Measurement in Latent Space via Autoencoders

> **A mathematically rigorous deep-learning approach for defining robust distance metrics between complex signals using convolutional autoencoder latent spaces.**

## Project Overview

This project develops an innovative convolutional autoencoder (CAE) designed to structure a latent space preserving precise notions of distance between temporal signals. The proposed method addresses the mathematical limitations and robustness issues inherent in traditional metrics like Euclidean distance and Dynamic Time Warping (DTW).

Key advantages:

- Robustness to noise, phase shifts, frequency and amplitude variations.
- Preservation of underlying geometric structures in signal spaces.
- Enhanced clustering, anomaly detection, and interpolation capabilities.

## Mathematical Motivation

Classical distance measures (`L2 norm`, `Fourier-based metrics`, `DTW`) often lack mathematical robustness or computational efficiency. By defining distances directly within a latent representation learned through CAEs, we achieve mathematical properties such as:

- **Continuity** and **smoothness** of distances with respect to input signal variations.
- **Invariant embeddings** respecting geometric transformations of signals.
- Preservation of intrinsic topology and distances between signals.

## Theorical Formulation

### Latent Space Definition via CAE

The autoencoder model is mathematically defined by two mappings:

- **Encoder** \( f_{\text{enc}}: \mathbb{R}^{n} \rightarrow \mathbb{R}^{d} \)
- **Decoder** \( f_{\text{dec}}: \mathbb{R}^{d} \rightarrow \mathbb{R}^{n} \)

such that for an input signal \( s(t) \in \mathbb{R}^{n} \):

\[
f_{\text{dec}}\bigl(f_{\text{enc}}(s(t))\bigr) \approx s(t)
\]

### Latent Space Distance Metrics

The mathematical construction of distances in latent space involves two metrics:

- **Euclidean Metric** (\( L^2 \)-norm):

\[
d(z_i, z_j) = \sqrt{\sum_{k=1}^{d}(z_{i,k} - z_{j,k})^2}
\]

- **Cosine Metric** (Inner-product derived metric):

\[
d(z_i, z_j) = 1 - \frac{\langle z_i, z_j \rangle}{\|z_i\|\cdot\|z_j\|}
\]

---

## Loss Function (Mathematical)

The CAE's training loss mathematically combines two crucial components: a **reconstruction loss** and a **distance-preserving loss**, defined explicitly as:

\[
\mathcal{L} = \alpha \underbrace{\frac{1}{N}\sum_{i=1}^{N}\|s_i(t)-\hat{s}_i(t)\|_2^2}_{\text{Reconstruction Loss}} 
+ \beta \underbrace{\frac{1}{N^2}\sum_{i=1}^{N}\sum_{j=1}^{N}\bigl(\|\theta_i - \theta_j\|_2^2 - \|z_i - z_j\|_2^2\bigr)^2}_{\text{Distance-Preserving Loss}}
\]

Here:

- \( s_i(t) \) : original signals
- \( \hat{s}_i(t) = f_{\text{dec}}(z_i) \) : reconstructed signals
- \( z_i = f_{\text{enc}}(s_i(t)) \) : latent vectors
- \( \theta_i \) : original parameter vectors of signals

## Experiments and Results

### Synthetic Signals

Synthetic signals, defined mathematically by polynomial and cosine combinations, provided controlled tests for latent space geometry validation:

\[
s(t) = \sum_{i=0}^{N_p} a_i t^i + \sum_{i=1}^{N_c} b_i \cos(w_i t + \varphi_i)
\]

The model demonstrated mathematically consistent trajectories in latent space when varying parameters independently.

### Real ECG Data Analysis

Validation on ECG signals confirmed practical mathematical robustness and performance consistency:

- Precise mathematical correspondence between latent distances and clinical ECG variations.
- Highly accurate reconstruction, validating latent space geometry.

---

## Latent Space Properties (Mathematical Insights)

Latent space analysis via PCA shows:

- Structured manifold preserving intrinsic signal geometry.
- Explicit preservation of linear and nonlinear parameter variations.

---

## 🛠️ Repository Structure

