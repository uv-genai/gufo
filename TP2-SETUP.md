# Two-Host Qwen3.8-Flash-Next Setup — Complete Guide

From bare twin CachyOS Strix Halo hosts to TP=2 tensor-parallel serving of
Qwen3.8-Flash-Next over RDMA striped across **two** Thunderbolt/USB4 cables.

Everything here is what was actually done on `strix-halo-1` (rank 0) and
`strix-halo-2` (rank 1). Both hosts are identical: CachyOS, kernel
7.2.8-2-cachyos (Clang-built), gcc 16.2.1, ROCm via `rocm-gfx1151-bin`,
`rdma-core` 65, user `ugo` with sudo.

```
strix-halo-1 (rank 0)  ==USB4 cable A==  strix-halo-2 (rank 1)
        |          ==USB4 cable B==              |
   HTTP :8000                          worker (no public port)
   (OpenAI-compatible API)
```

Model files must be identical on both hosts:
`~/.cache/models/Qwen3.8-Flash-Next/` containing `Q8_0/` (6 shards),
`MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf`, `mmproj-BF16.gguf`.

---

## Part A — Build gufo with TP2 RDMA

Full details and rationale: [BUILD-LOCAL.md](BUILD-LOCAL.md) in this repo.
The short version (identical on both hosts):

```sh
cd ~/software/gufo
./build-gufo.sh          # configure + build + gufo diagnose
./build-gufo.sh --clean  # from scratch
```

The script encapsulates three toolchain workarounds you must not lose:

| Symptom | Cause | Fix (already in build-gufo.sh) |
|---|---|---|
| `expected an identifier for the attribute name` in `<format>` | HIP's `__noinline__` macro vs GCC 16 libstdc++ | `-DCMAKE_CXX_FLAGS="-include format" -DCMAKE_HIP_FLAGS="-include format"` |
| `Failed to find HIP root directory` | ROCm not on non-interactive PATH | `export PATH=/opt/rocm/bin:/opt/rocm/core/lib/llvm/bin:$PATH` |
| `cannot find ROCm device library` | `ROCM_PATH` + symlinked `/opt/rocm/lib` breaks clang | never set `ROCM_PATH`; pin `-DCMAKE_HIP_COMPILER=/opt/rocm/core/lib/llvm/bin/clang++` |

`-DGUFO_ENABLE_TP2_RDMA=ON` is the whole point of this build (RDMA TP2 is off
by default). Verify with `./build/release/gufo diagnose` and
`./build/release/gufo serve llm --help | grep tp-`.

Repo location on both hosts: `~/software/gufo`, branch `rdma`, same commit.
Build output: `~/software/gufo/build/release/gufo` (both hosts, byte-identical).

---

## Part B — Networking, Stage 1: IP over Thunderbolt + soft-RoCE (RXE)

This is the simple baseline. It works with stock modules only, needs no
third-party code, and gets ~860-930 tok/s prefill / ~22 tok/s decode.
Skip to Stage 2 if you want the fast path immediately.

1. **Plug both USB4 cables.** Verify training (ideally `20.0 Gb/s` × 2 lanes
   per cable; `10.0 Gb/s` × 2 = a 20 Gb/s cable or poor training):

```sh
for d in /sys/bus/thunderbolt/devices/*-*; do
  [ -e "$d/rx_speed" ] && echo "$(basename $d): rx $(cat $d/rx_speed) x $(cat $d/rx_lanes)"
done
```

2. **Give each cable an IP subnet.** `thunderbolt_net` auto-loads when a peer
   offers network service, creating `thunderbolt0`/`thunderbolt1`:

```sh
# strix-halo-1                          # strix-halo-2
sudo ip addr add 192.168.100.1/24 dev thunderbolt0
sudo ip addr add 192.168.101.1/24 dev thunderbolt1
sudo ip link set thunderbolt0 up        # 192.168.100.2/24, 192.168.101.2/24
sudo ip link set thunderbolt1 up
```

(Persist via NetworkManager dummy/ethernet connections if desired.)

3. **Create soft-RoCE (RXE) devices on those netdevs:**

```sh
sudo rdma link add rxe0 type rxe netdev thunderbolt0
sudo rdma link add rxe1 type rxe netdev thunderbolt1
ibv_devices    # rxe0, rxe1
```

`rdma link add` is runtime-only; recreate after boot (systemd unit or
NetworkManager dispatcher).

4. **Run gufo with `--tp-rdma-device rxe0`** (see Part C). Bootstrap/control
   TCP can ride the Thunderbolt IPs (`--tp-bootstrap-host 192.168.100.1`).
   Note gufo uses one QP on one device — one cable only, CPU-processed
   packets. Expect roughly half the throughput of Stage 2.

---

## Part C — Networking, Stage 2: RDMA over both cables (thunderbolt-ibverbs)

The fast path: kernel-module RDMA over the USB4 DMA rings, with **write
striping** spreading gufo's single QP over every ring of **every** cable.
~1260-1280 tok/s prefill / ~43 tok/s decode at current link speeds
(2 × 20 Gb/s aggregate), up to ~44 Gbit/s with 40 Gb/s cables.

Upstream: https://github.com/neuhaus/donnerkeule (branch
`feat/write-striping`, a fork of hellas-ai/thunderbolt-ibverbs). Research
driver: works great, taints the kernel, not for production.

### 1. Take a rollback point first

Snapper is active on these hosts (btrfs root). Take a manual snapshot
(automatic ones only fire for pacman):

```sh
sudo snapper create -d "pre thunderbolt-ibverbs install" --print-number
```

### 2. Install the userspace libibverbs provider

Needed so `ibv_devices`/gufo can enumerate `usb4_rdma*` devices.

```sh
cd /tmp
curl -sLO https://github.com/hellas-ai/thunderbolt-ibverbs/releases/download/v0.3.4/usb4-rdma-provider-0.3.4-1-x86_64.pkg.tar.zst
sudo pacman -U usb4-rdma-provider-0.3.4-1-x86_64.pkg.tar.zst
```

**Caveat:** the package ships `libusb4_rdma-rdmav59.so`, built against
rdma-core 59. Ours is rdma-core 65 — a symlink is *not* enough
(`version 'IBVERBS_PRIVATE_59' not found`). Build the provider against
rdma-core 65 instead (same source the package was built from; the fork's
userspace is byte-identical to upstream):

```sh
sudo pacman -S --needed python-pyelftools python-docutils   # rdma-core build deps
cd ~/software
git clone --depth=1 --branch v65.0 https://github.com/linux-rdma/rdma-core
cd rdma-core
patch -p1 < ~/software/donnerkeule/packaging/rdma-core-patches/0001-*.patch
patch -p1 < ~/software/donnerkeule/packaging/rdma-core-patches/0002-*.patch
mkdir build && cd build
cmake -GNinja -DNO_PYVERBS=1 ..
ninja
sudo cp lib/libusb4_rdma-rdmav65.so /usr/lib/libibverbs/
# /etc/libibverbs.d/usb4_rdma.driver already installed by the pacman package
```

### 3. Build the kernel module (fork, with write striping)

```sh
git clone -b feat/write-striping https://github.com/neuhaus/donnerkeule.git ~/software/donnerkeule
cd ~/software/donnerkeule
make LLVM=1 KVER="$(uname -r)"
sudo make LLVM=1 KVER="$(uname -r)" modules_install
sudo depmod -a
```

**`LLVM=1` is mandatory**: the CachyOS kernel is Clang-built; GCC chokes on
the kernel's Clang-only flags. The module is *not* DKMS-managed this way —
after a kernel upgrade repeat the three commands (or `sudo pacman -S dkms`
and use `make dkms-add/dkms-build/dkms-install` for automatic rebuilds).

### 4. Persistent configuration

```sh
# /etc/modprobe.d/thunderbolt-ibverbs.conf
#   thunderbolt_net must go: it steals DMA rings from the module (2 per
#   controller). IPoTB and RXE from Stage 1 stop working while blacklisted.
sudo tee /etc/modprobe.d/thunderbolt-ibverbs.conf <<'EOF'
blacklist thunderbolt_net
options thunderbolt_ibverbs profile=linux_perf tbnet=prefer_rdma lanes=2 bind_services=1 allocate_rings=1 start_rings=1 negotiate_native=1 enable_tunnels=1 register_verbs=1 roce_netdev=tbv0 native_write_striping=1
EOF

# /etc/modules-load.d/thunderbolt-ibverbs.conf
echo thunderbolt_ibverbs | sudo tee /etc/modules-load.d/thunderbolt-ibverbs.conf

# udev: keep the usb4_rdma names (rdma-core renames IB devices by PCI path
# otherwise, and the provider finds its devices by name)
sudo cp /usr/lib/udev/rules.d/60-rdma-persistent-naming.rules /etc/udev/rules.d/
sudo sed -i 's|^ACTION=="add", SUBSYSTEM=="infiniband", PROGRAM|ACTION=="add", SUBSYSTEM=="infiniband", KERNEL!="usb4_rdma*", PROGRAM|' \
  /etc/udev/rules.d/60-rdma-persistent-naming.rules
sudo udevadm control --reload
```

**The full option set matters.** `bind_services=1 allocate_rings=1
start_rings=1 negotiate_native=1 enable_tunnels=1` are all needed on this
module version (defaults are off); the shorter snippet in gufo's TP2.md is
outdated. `native_write_striping=1` is what stripes one QP across all cables.

### 5. RoCE GID netdev (`tbv0`)

The RDMA devices need a netdev with an IPv4 address for RoCE v2 GID metadata.
A dummy device is enough — one per host:

```sh
# strix-halo-1                          # strix-halo-2
sudo nmcli connection add type dummy ifname tbv0 con-name tbv0 \
  ipv4.method manual ipv4.addresses 10.77.0.1/24 ipv6.method disabled
#                                           10.77.0.2/24
```

### 6. Switch over

```sh
sudo rmmod thunderbolt_net        # drops thunderbolt0/1 + rxe0/rxe1
sudo modprobe thunderbolt_ibverbs # options come from modprobe.d
ibv_devices                       # usb4_rdma* — 2 rails per peer, spanning
                                  # both cables (path=1 and path=257)
sudo cat /sys/kernel/debug/thunderbolt_ibverbs/summary | grep striping
#   native_write_striping: 1  →  good
```

Verify the links:

```sh
sudo cat /sys/kernel/debug/thunderbolt_ibverbs/peers   # rails per path, speeds
```

### 7. Device naming caveat

`usb4_rdma<N>` numbers **shift on every cable re-plug** (and stale peers can
linger until the module is reloaded: `sudo rmmod thunderbolt_ibverbs &&
sudo modprobe thunderbolt_ibverbs`). The rank scripts pick the device
automatically (lowest GUID, identical choice on both hosts), so this is
normally invisible. Also give the links ~15 s to re-register after a re-plug
before starting gufo.

---

## Part D — Running the two-host server

Two scripts, one per host (same in content, mirrored roles):

| Role | Host | Script |
|---|---|---|
| rank 0 — public HTTP server on `:8000` | `strix-halo-1` | `~/opt/llama.cpp-cli/tp2-rank0.sh` |
| rank 1 — worker (no public port) | `strix-halo-2` | `~/opt/llama.cpp-cli/tp2-rank1.sh` |

```sh
# 1. rank 0 first
ssh strix-halo-1 '~/opt/llama.cpp-cli/tp2-rank0.sh'
# 2. rank 1 within 30 seconds
~/opt/llama.cpp-cli/tp2-rank1.sh
```

Both scripts use: Q8_0 model (6 shards) + MTP draft + mmproj, `--sessions 4`,
native 262k context, shared control token, bootstrap/control ports 18515/18516
over the LAN (192.168.1.22 — the Thunderbolt IPs don't exist in Stage 2),
RDMA over the auto-picked `usb4_rdma*` device.

Success looks like (both logs):

```
event=rdma_ready rdma_device=usb4_rdma5 port=1 link=roce_v2 gid_index=1 mtu=4096
event=load_completed ...
event=listening address=http://0.0.0.0:8000
```

Use the API (model id is **`Qwen3.8 Flash Next`**, with spaces):

```sh
curl http://192.168.1.22:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen3.8 Flash Next","messages":[{"role":"user","content":"Say something"}]}'
```

Stop: kill either rank; the peer notices at once and exits (status 1).
Restart needs both ranks again (rank 0 first).

---

## Part E — Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `error: expected an identifier for the attribute name` in `<format>` | GCC 16 vs HIP headers — use `build-gufo.sh` (Part A) |
| `cannot find ROCm device library` | `ROCM_PATH` is set — unset it (Part A) |
| module build fails with Clang-only flags passed to gcc | kernel is Clang-built: `make LLVM=1` |
| `ibv_devices` empty, kernel devices exist | provider mismatch: rebuild against rdma-core 65 (Part C.2) |
| `version 'IBVERBS_PRIVATE_59' not found` | same — the prebuilt provider is too old for rdma-core 65 |
| `no usb4_rdma device found` from the scripts | links re-negotiating after a re-plug (wait 15 s), or stale peers: reload the module |
| `text runner request-state claim exceeds state capacity` | single-host runs with 4 full-context sessions: add `--context 32768` (TP2 splits state, so 4 sessions are fine there) |
| `model_not_found` from the API | model id is `Qwen3.8 Flash Next` (spaces) |
| `TP worker response receive failed` / `tp_pair_lost` | the other rank died or was killed — expected; restart both |
| decode ~20 tok/s, prefill ~900 | you're on RXE Stage 1, not usb4_rdma — check `--tp-rdma-device` and that `thunderbolt_net` is blacklisted |
| links train at `10.0 Gb/s × 2` | per cable that's 20 Gb/s, not 40. Need 40 Gb/s-rated cables; otherwise try re-plugging (sometimes trains at 20 after several attempts); check BIOS Thunderbolt settings |
| handshake HTTP 500 naming a setting | the two ranks disagree on model/MTP/context/sessions/mmproj — make flags identical |

## Part F — Rollback

Piecewise (Stage 2 → Stage 1 or bare):

```sh
sudo rmmod thunderbolt_ibverbs
sudo rm /etc/modprobe.d/thunderbolt-ibverbs.conf
sudo rm /etc/modules-load.d/thunderbolt-ibverbs.conf
sudo rm /etc/udev/rules.d/60-rdma-persistent-naming.rules
sudo rm /lib/modules/$(uname -r)/updates/thunderbolt_ibverbs.ko.zst && sudo depmod -a
sudo rm /usr/lib/libibverbs/libusb4_rdma-rdmav65.so   # plus pacman -R usb4-rdma-provider
sudo nmcli connection delete tbv0
sudo modprobe thunderbolt_net                          # back to Stage 1
```

Full rollback: `sudo snapper rollback <N>` with the snapshot taken in
Part C.1 (or the automatic ones around each pacman step).

Caveat no snapshot can undo: a buggy kernel module can crash the kernel;
reboot with the module removed restores normal operation.

---

## Reference — measured performance (Q8_0 + MTP, 2,800-token prompts)

| Transport | prefill tok/s | decode tok/s |
|---|---|---|
| RXE over one cable (Stage 1) | 860–930 | ~21.7 |
| usb4_rdma striped, 2× 20 Gb/s cables (Stage 2, current) | 1260–1280 | ~43 |
| usb4_rdma striped, 2× 40 Gb/s cables (projected) | higher — link-bound | higher |

SMT on/off makes no measurable difference on these workloads (tested
single-host and TP2); BIOS SMT state need not match between hosts.
