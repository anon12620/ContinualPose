# Continual Learning for Human Pose Estimation

## Installation

This is an MMPose-based project with the following prerequisites:

- CUDA 12.1
- Python 3.10
- PyTorch 2.1.0
- MMPose 1.3.0

Once the prerequisites are satisfied, the project can be installed as a Python package using the following command:

```bash
pip install [--editable] .
```

Use the `--editable` flag to install the package in development mode.

## Data Preparation

Create a 'data' directory and set up the following datasets:

- COCO
- MPII
- CrowdPose

For folder structure, please refer to the [MMPose documentation](https://mmpose.readthedocs.io/en/latest/dataset_zoo/2d_body_keypoint.html).

## Training

```bash
python tools/train.py <config_file> [optional arguments]
```

## License

This repository is provided solely for the purpose of transparency in the context of the related paper submission. All rights are reserved by the authors. No part of this repository, including but not limited to the code, data, and documentation, may be redistributed or used for any purposes other than reviewing its contents for the associated paper submission. Unauthorized use, reproduction, or distribution of any part of this repository is strictly prohibited.

