# Recursive training and higher-density bias in diffusion models

IE 643 course project, Tensor Titans.

When a generative model is trained on images made by another generative model, its mistakes add up. Our
question is whether this makes a diffusion model drift towards the classes it already sees a lot of, and
whether we can stop that drift without throwing away any generated images.

This repository has three files:

- `main.ipynb`: the notebook that runs the whole project and produced every number
  in the report.
- `report.pdf`: the full report (method, experiments, results, the interface and what didn't work).
- `README.md`: this file.

## What the notebook does

1. **Builds an imbalanced CIFAR-10 split (D0).** Five common classes with about 4,900 images each (airplane,
   automobile, ship, truck, frog) and five rare ones going from 1,500 down to 50 (bird, deer, cat, dog,
   horse). That's 27,400 images and a 100:1 imbalance.
2. **Runs the recursive loop** D0 → M0 → D1 → M1 → D2 → M2 → D3. Every model is a small unconditional
   diffusion model (a 24.3M-parameter UNet) trained from scratch on the images the previous model generated.
3. **Runs two chains side by side.** The baseline uses the plain diffusion loss. CDR-W, our method, weights
   each training image's loss by how under- or over-represented its class is compared with the original
   data. Both chains get exactly the same images and compute.
4. **Measures every generated set** with a ResNet-18 classifier and DINOv2 features: class shares, tail
   retention, KL divergence to the original mix, Fréchet distance and recall.

All the code (`rdb/` package and `app.py`) is written out by the notebook itself, so the notebook is
all you need.

## Running it

1. On Kaggle: **Create → New Notebook → File → Import Notebook**, and pick the `.ipynb`.
2. In the settings, set **Accelerator: GPU T4 x2** and **Internet: On**.
3. Click **Save Version → Save & Run All**.

A full run takes about 9.5 hours. When it finishes, everything is in `/kaggle/working/run`: the split, all
generated datasets, all trained models (`ema.pt` files), the classifiers, metrics and figures. It also zips
them into `deliverables.zip`. If you want to see that the pipeline works first, the `smoke` preset finishes in
about 15 minutes with a tiny model.

## What we found

| | common-class share | tail retention (1 is ideal) | KL to original mix (0 is ideal) |
|---|---|---|---|
| original data | 89.4% | – | – |
| M0 output (D1) | 94.8% | 0.53 | 0.042 |
| baseline after two rounds (D3) | 94.9% | 0.52 | 0.284 |
| CDR-W after two rounds (D3) | 93.4% | 0.68 | 0.066 |

- **The bias is already there after one model.** M0 makes more common-class images than it saw.
- **Recursion makes the class mix drift much further.** In two rounds airplane goes from 23% to 41% of the
  output and truck drops from 13% to 4%, even though both had the same number of training images.
- **CDR-W removes about three quarters of that drift** and keeps more of the rare classes.
- **What it doesn't fix:** image quality still drops from round to round in both chains, and horse (only 50
  training images) can't be brought back by reweighting.
- **A partial fix for quality:** in a follow-up test, finishing the second-round images with the original
  model M0 brought the share of unrecognisable images down from about 36% to 22%. Details are in the report.

These come from one run with one seed, so treat small differences with some caution.

## Interface

After the run, the notebook can launch a Gradio interface (`app.py`) on the output folder. You can generate
images from any of the trained models, see how their class mix compares with the original data, and upload
your own images to check how biased they are.

## Built with

PyTorch, Hugging Face `diffusers` (UNet and DDIM sampling), torchvision, DINOv2 and Gradio. The data is
CIFAR-10, loaded from the Hugging Face Hub (`uoft-cs/cifar10`).
