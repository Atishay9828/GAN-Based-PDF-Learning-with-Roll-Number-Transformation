# GAN-Based-PDF-Learning-with-Roll-Number-Transformation

This repository implements probability density function (PDF) learning using a Generative Adversarial Network (GAN) on real-world environmental data. The distribution is personalized through a roll-number–dependent nonlinear transformation.

## Methodology
~~~
┌──────────────────────────┐
│ Dataset Loading          │
│ (India Air Quality CSV)  │
└─────────┬────────────────┘
          ↓
┌──────────────────────────┐
│ Feature Selection        │
│ (NO₂ Extraction)         │
└─────────┬────────────────┘
          ↓
┌──────────────────────────┐
│ Non-Linear Transformation│
│ z = x + ar·sin(br·x)     │
└─────────┬────────────────┘
          ↓
┌──────────────────────────┐
│ Normalization            │
│ (Zero Mean / Unit Var)   │
└─────────┬────────────────┘
          ↓
┌──────────────────────────┐
│ GAN Training (WGAN)      │
│ Generator vs             |
| Discriminator            │
└─────────┬────────────────┘
          ↓
┌──────────────────────────┐
│ Sample Generation        │
│ z_f = G(N(0,1))          │
└─────────┬────────────────┘
          ↓
┌──────────────────────────┐
│ KDE PDF Estimation       │
│ p̂(z) from GAN samples    │
└──────────────────────────┘
~~~
The workflow learns an unknown probability density purely from data using adversarial learning. No parametric PDF assumptions are made.

## Description
* Task Type: Probability Density Function Learning
* Problem Nature: Raw environmental data transformed via personalized nonlinear mapping and modeled using GANs
* Objective: Learn implicit PDF of transformed NO₂ concentration
* Core Components: Dataset ingestion, Feature extraction, Roll-number transformation, GAN training, KDE-based PDF estimation
* Architecture Type: Transform → Normalize → GAN → KDE

## Input / Output
### Input
* India Air Quality Dataset (CSV)
* Feature: NO₂ concentration
* University Roll Number r
### Transformation
z = x + a_r · sin(b_r · x)
a_r = 0.5 × (r mod 7)
b_r = 0.3 × ((r mod 5) + 1)
### Output
* Generated samples from learned distribution
* KDE-based PDF approximation
* Training stability curves
* Architecture summary

## GAN Architecture
### Generator
* Linear(1 → 64) + ReLU
* Linear(64 → 64) + ReLU
* Linear(64 → 1)
### Discriminator
* Linear(1 → 64) + LeakyReLU
* Linear(64 → 64) + LeakyReLU
* Linear(64 → 1)
Training uses Wasserstein GAN with weight clipping.

## Execution Environment
* Language: Python
* Notebook: Jupyter / Google Colab
* Libraries: PyTorch, NumPy, Pandas, Seaborn, Matplotlib, Scikit-learn
* Platform: CPU / GPU

## Results Summary
* Generator successfully approximates the transformed NO₂ distribution
* KDE-estimated PDF aligns closely with real data
* Stable convergence achieved using WGAN
* Personalized nonlinear transform introduces controlled variation
* GAN learns density implicitly from samples only

## Visual Results
### Training Stability
The generator and discriminator losses converge toward equilibrium, indicating stable adversarial training without mode collapse.
<img width="864" height="455" alt="image" src="https://github.com/user-attachments/assets/d48d2c11-c997-4fca-a812-c8959946ad3e" />

## PDF Comparison

The KDE-estimated PDF from GAN-generated samples closely aligns with the real transformed distribution.
The main peak and distribution shape are successfully captured.
<img width="859" height="532" alt="image" src="https://github.com/user-attachments/assets/8343c555-2ce3-42d2-9ebd-87c6c4945a0b" />

## Key Observations
* Main distribution mode is captured effectively
* Generator and discriminator reach equilibrium
* KDE smooths generated samples
* Normalization is critical for convergence
* Simple WGAN outperforms WGAN-GP for 1D data
* Large datasets require fewer epochs

## Transformation Parameters
| Parameter | Value     |
| --------- | --------- |
| a_r       | `2.0`     |
| b_r       | `0.6`     |

## Evaluation Strategy
| Metric               | Observation                             |
| -------------------- | --------------------------------------- |
| Mode Coverage        | Main distribution mode captured         |
| Training Stability   | Stable convergence observed             |
| Distribution Quality | KDE overlap indicates strong similarity |

## Conclusion
This project demonstrates end-to-end probability density learning using GANs on real-world environmental data.
* Nonlinear roll-number transformations personalize distributions
* GANs can learn implicit PDFs without analytical assumptions
* KDE provides smooth density approximation
* For low-dimensional problems, simpler adversarial setups outperform complex ones

## Applications
* Environmental data modeling
* Density estimation research
* Personalized statistical pipelines
* GAN-based distribution learning
* Academic experimentation
