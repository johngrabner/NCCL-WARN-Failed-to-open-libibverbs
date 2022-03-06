# ddp across 2x A6000 locks up because IOMMU was enabled

This project is a docker containing a minimal pytorch-lightning  that does ddp across 2 gpus.  The host only needs to install 
NVIDIA drivers, docker, docker compose, nvidia for docker, and all other sw is contained in this docker image.

Use at your own risk since this project was slapped together during debugging to narrow the issue.

## References

[pytorch issue with logs and final solution](https://github.com/pytorch/pytorch/issues/73790)

[pytorch-lightning issues for this issue](https://github.com/PyTorchLightning/pytorch-lightning/issues/12235)

[Nvidia NCCL troubleshooting](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting.html) NCCL used when ddp 
is used. Turns out it says instability and lockups if IOMMU is enabled. It contains a bunch of other troubleshooting information.

