# Notice

`flora-deploy` is licensed under the Apache License 2.0 (see `LICENSE`). That license covers
**only the code in this repository**.

## NVIDIA Flora models and code are not part of this repository

This framework deploys NVIDIA's Flora models:

- [`nvidia/Flora-TempSeg`](https://huggingface.co/nvidia/Flora-TempSeg)
- [`nvidia/Flora-Captioner`](https://huggingface.co/nvidia/Flora-Captioner)
- [`nvidia/Flora-Rewarder`](https://huggingface.co/nvidia/Flora-Rewarder)

Their weights, prompts and most of their inference code are governed by the **NVIDIA OneWay
Noncommercial License (22 March 2022)**. Under that license they may be used **for research or
evaluation purposes only**, and NVIDIA reserves all commercial rights. The Qwen3.5 base models are
licensed under Apache 2.0 by their authors.

What this means in practice:

- **No NVIDIA material is redistributed here.** The repository contains no NVIDIA weights, prompts
  or noncommercially licensed code. At runtime the framework downloads them from Hugging Face into
  *your own* Modal workspace and calls NVIDIA's code there by name.
- **Derived weights stay with you.** `flora prepare` builds the BF16 Captioner and the merged
  TempSeg inside your own Modal Volume. If you redistribute them, you must do so under the OneWay
  license, include its text and notices, and mark them as derivatives of NVIDIA's work.
- **Commercial use is not covered.** A paid or hosted labeling service built on these models would
  be commercial use.
- **Not for control or safety.** Flora outputs are annotations. They are not meant for robot
  control or as a safety monitor.
- **The status of generated labels is unresolved.** The license text does not settle whether
  labels Flora produces are themselves restricted. If you plan to use the labels commercially, get
  legal advice.

This notice is not legal advice.
