# DeepSeek V4.1 Flash

**Not implemented in gufo yet.** Capacity and design for running
[`deepseek-ai/DeepSeek-V4.1-Flash`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)
(763B MoE, CED + CSA2 + Engram) on Strix Halo, including NVMe weight
streaming and optional two-host TP2/RDMA.

See [PLAN.md](PLAN.md) for the full design. The **working reference engine**
for the VQ artifact is **[ds4-dfm-rs](https://github.com/Baekpica/ds4-dfm-rs)**
([V4.1 guide](https://github.com/Baekpica/ds4-dfm-rs/blob/main/docs/deepseek41-flash.md)).
For the supported DeepSeek line today, use
[DeepSeek V4 Flash](../deepseek-v4-flash/README.md).
