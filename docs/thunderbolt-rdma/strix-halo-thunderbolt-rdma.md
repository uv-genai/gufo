# Thunderbolt/USB4 host-to-host RDMA — Strix Halo pair

Full setup, verification, and recovery runbook for the two-node
`strix-halo-1` ↔ `strix-halo-2` cluster running RDMA verbs over directly
connected Thunderbolt 5 / USB4 cables.

Driver: [`thunderbolt-ibverbs`](https://github.com/hellas-ai/thunderbolt-ibverbs)
(local tree: `/home/ugo/software/donnerkeule`, commit `e37ecaf`).

Verified state as of this writing: **37.4 Gb/s** striped WRITE, **6.0 µs**
round-trip WRITE latency, both cables active, kernel `7.2.9-2-cachyos`.

> ⚠️ Upstream warning, restated because it matters: this is a **research
> driver**. It is buggy and it is not secure. Do not expose these links to
> untrusted networks or peers.

---

## 1. Topology

```
strix-halo-1 (192.168.1.22)                strix-halo-2 (192.168.1.92)
  GMKtec NucBox EVO-X2, AMD Strix Halo       (identical machine)
  kernel 7.2.9-2-cachyos (clang-built)

  TB host router c7:00.5  ══ TB5 cable A ══>  TB host router (domain mirror)
  TB host router c7:00.6  ══ TB5 cable B ══>  TB host router (domain mirror)

  eno1  RTL8125 2.5GbE  — management / SSH / perftest rendezvous
  tbv0  10.77.0.1/24    — RoCE GID anchor only (NOT an IP link, see §7)
  wt0   wireguard tunnel  — unrelated
```

Two physical Thunderbolt cables give two **rails**; the driver stripes a
single QP across both (`native_write_striping=1`, `lanes=2`). Four IB
devices appear per host (2 rails × 2 USB4 domains):

| device | rail | domain |
|---|---|---|
| `usb4_rdma0` | peer/0 | 0 |
| `usb4_rdma1` | peer/1 | 0 |
| `usb4_rdma5` | peer/0 | 1 |
| `usb4_rdma6` | peer/1 | 1 |

A single device already uses **both** rails. Two contending devices do not
double the bandwidth — they split the same ~37 Gb/s ceiling. That is the
test that proves striping is live.

---

## 2. Prerequisites

### Hardware
- Two Thunderbolt 5 / USB4 hosts, **two** direct TB cables between them.
- Thunderbolt/USB4 enabled in BIOS, no IOMMU quirks blocking DMA.

### Kernel
Linux **6.14 or newer**. The driver needs Thunderbolt/USB4 subsystem
interfaces that older kernels do not export. CachyOS `7.2.x` is fine.

### Packages (Arch / CachyOS)

```sh
sudo pacman -S --needed \
    base-devel clang lld llvm \      # MUST match the kernel's compiler, see §4
    linux-headers \
    kmod make git \
    rdma-core \                     # libibverbs + ibv_* tools
    dkms                            # see §8
```

`perftest` for benchmarking (`ib_write_bw`, `ib_send_lat`, `ib_write_lat`).

**Check the kernel's compiler before building anything:**

```sh
grep LINUX_COMPILER /usr/lib/modules/$(uname -r)/build/include/generated/compile.h
# → #define LINUX_COMPILER  "clang version 23.1.1, LLD 23.1.1"
```

CachyOS kernels are built with **clang**. This is the single biggest gotcha
on this platform (§4).

---

## 3. Userspace libibverbs provider

The kernel module alone is not enough — `libibverbs` needs a provider so it
can enumerate `usb4_rdma*` devices.

Package `usb4-rdma-provider` installs:

```
/etc/libibverbs.d/usb4_rdma.driver
/usr/lib/libibverbs/libusb4_rdma-rdmav65.so
```

The `rdmavNN` suffix must match **rdma-core's** ABI. `rdma-core 65` needs
`libusb4_rdma-rdmav65.so`. A leftover `...-rdmav59.so` from an older
provider is inert but confusing; remove it:

```sh
sudo rm -f /usr/lib/libibverbs/libusb4_rdma-rdmav59.so
```

Verify:

```sh
ibv_devices
#  device            node GUID
#  ------            ----------------
#  usb4_rdma0        0200544258524253
#  usb4_rdma5        0200544257524253
#  usb4_rdma1        0200544258524254
#  usb4_rdma6        0200544257524254
```

For containers: the kernel module stays on the host; only the provider goes
into the image. Run with `--device=/dev/infiniband --cap-add=IPC_LOCK
--ulimit memlock=-1`.

---

## 4. Building the kernel module

Source tree: `/home/ugo/software/donnerkeule`

### ⚠️ The clang gotcha

A plain `make` **fails** against a CachyOS kernel:

```
gcc: error: unrecognized command-line option '-fms-anonymous-structs'
gcc: error: unrecognized command-line option '-mstack-alignment=8'
gcc: error: unrecognized command-line option '-mretpoline-external-thunk'
gcc: error: unrecognized command-line option '-fexperimental-late-parse-attributes'
gcc: error: unrecognized command-line option '-fsplit-lto-unit'
gcc: error: unrecognized command-line option '-fdebug-info-for-profiling'
gcc: error: unrecognized command-line option '-mllvm'
make: *** [.../scripts/Makefile.build:289: main.o] Error 1
```

The kernel records its build flags; gcc cannot consume clang-only flags.
**External modules must be built with the same toolchain as the kernel.**
Add `LLVM=1`:

```sh
cd /home/ugo/software/donnerkeule
make KVER=$(uname -r) LLVM=1 modules
```

Verify the result matches the running kernel:

```sh
modinfo kernel/thunderbolt_ibverbs.ko | grep -E 'vermagic|retpoline'
# vermagic: 7.2.9-2-cachyos SMP preempt mod_unload
# retpoline: Y
readelf -p .comment kernel/thunderbolt_ibverbs.ko | grep clang
# [ 1]  clang version 23.1.1
```

### Install

```sh
sudo make KVER=$(uname -r) LLVM=1 modules_install   # → /lib/modules/<KVER>/updates/
sudo depmod -a
```

### Load

```sh
sudo modprobe thunderbolt_ibverbs
ls /sys/class/infiniband/          # usb4_rdma0 1 5 6
```

Devices only appear once **both** hosts have the module loaded and the
cables connected — each side advertises a Thunderbolt service and binds the
peer's. Loading on one host alone yields a loaded module and **zero** IB
devices. This is expected, not a failure.

---

## 5. Module parameters

`/etc/modprobe.d/thunderbolt-ibverbs.conf`:

```
blacklist thunderbolt_net
options thunderbolt_ibverbs profile=linux_perf tbnet=prefer_rdma lanes=2 \
    bind_services=1 allocate_rings=1 start_rings=1 negotiate_native=1 \
    enable_tunnels=1 register_verbs=1 roce_netdev=tbv0 native_write_striping=1
```

`/etc/modules-load.d/thunderbolt-ibverbs.conf`:

```
thunderbolt_ibverbs
```

Key parameters (read them back live from
`/sys/module/thunderbolt_ibverbs/parameters/`):

| parameter | value | meaning |
|---|---|---|
| `lanes` | 2 | two rails per link |
| `native_write_striping` | Y | split each WRITE into one contiguous block per rail |
| `native_write_stripe_min_bytes` | 65536 | striping block granularity |
| `profile` | linux_perf | Linux↔Linux native transport (not Apple-compat) |
| `tbnet` | prefer_rdma | keep `thunderbolt_net` off the DMA rings |
| `roce_netdev` | tbv0 | netdev used to synthesise the RoCE GID table |
| `native_tx_max_inflight` | 32 | in-flight WRITE frames per QP |
| `nhi_interrupt_throttle_ns` | 0 | no NHI interrupt throttling |

**Why `blacklist thunderbolt_net`:** distributions auto-load
`thunderbolt_net` by modalias when a peer offers its network service. It
takes one DMA ring per USB4 controller — and there are only two for data —
so the RDMA driver gets fewer rails or none. Blacklisting it keeps both
rings available.

Confirm what is actually live:

```sh
for f in /sys/module/thunderbolt_ibverbs/parameters/*; do
  printf "%-34s %s\n" "$(basename $f)" "$(cat $f)"
done
sudo cat /sys/kernel/debug/thunderbolt_ibverbs/summary
```

Healthy summary shows `native_service_count: 2`, `services_registered: 1`,
`native_enabled: 1`, and in `dmesg`:

```
thunderbolt_ibverbs: bound native service id=0 key=tbverbs  native_lane=0 route=0x2 link_speed=10Gb/s width=0x2 rail=0x42f89564 path_id=1
thunderbolt_ibverbs: bound native service id=1 key=tbverb1 native_lane=1 route=0x2 link_speed=10Gb/s width=0x2 rail=0x12b84f1b path_id=257
thunderbolt_ibverbs: registered native ib_device usb4_rdma0 rail=peer2/0 domain=0 guid=0200544258524253
```

---

## 6. Verification

### Ports

```sh
for d in /sys/class/infiniband/usb4_rdma*; do
  echo "$(basename $d): $(cat $d/ports/1/state) phys=$(cat $d/ports/1/phys_state)"
done
# all: 4: ACTIVE phys=5: LinkUp
```

### GID table — pick the right index

```sh
for i in 0 1; do
  echo "idx=$i $(cat /sys/class/infiniband/usb4_rdma0/ports/1/gids/$i) \
$(cat /sys/class/infiniband/usb4_rdma0/ports/1/gid_attrs/types/$i)"
done
# idx=0  fe80:...:502b:4eff:fe5a:2b88   RoCE v2   (link-local)
# idx=1  0000:...:0000:ffff:0a4d:0001   RoCE v2   (IPv4-mapped 10.77.0.1)
```

**Always pass `-x 1` to perftest.** Index 0 is link-local and will not
resolve against the peer's IPv4 address; the test hangs silently.

### ⚠️ `tbv0` is not an IP link

`tbv0` exists only to give the RoCE GID table an IPv4 address. It does
**not** carry routable IP traffic between the hosts:

```sh
ping -I tbv0 10.77.0.2     # 100% packet loss — EXPECTED
```

perftest needs a TCP socket for its rendezvous handshake, so point it at
the **management** interface instead. The RDMA data still flows over
Thunderbolt; only the handshake uses `eno1`:

```sh
# server (strix-halo-2)
ib_write_bw -d usb4_rdma0 -x 1 -F -D 15 -s 1048576

# client (strix-halo-1)
ib_write_bw -d usb4_rdma0 -x 1 -F -D 15 -s 1048576 192.168.1.92
```

### ⚠️ Message size matters — do not trust the default

`ib_write_bw` defaults to **64 KiB**, which sits ~15% below the plateau.
Reporting the default number makes a healthy link look broken:

| message size | BW average | |
|---|---|---|
| 65 536 B (default) | 3 793 MiB/s | **31.8 Gb/s** ← misleading |
| 262 144 B | 4 314 MiB/s | 36.2 Gb/s |
| **1 048 576 B** | **4 457 MiB/s** | **37.4 Gb/s** ← report this |
| 4 194 304 B | 4 414 MiB/s | 37.0 Gb/s |

Use `-s 1048576` for headline bandwidth.

### ⚠️ `pkill -f ib_write_bw` over SSH kills your own shell

The pattern matches the remote `bash -c` command line, so the remote shell
terminates itself and you get empty output. Use a bracket class:

```sh
pkill -f "[i]b_write_bw"
```

---

## 7. Reference performance (measured, idle hosts)

### Bandwidth — RDMA WRITE, `usb4_rdma0`, 1 MiB, 15 s

```
#bytes     #iterations   BW average[MiB/sec]   MsgRate[Mpps]
1048576    40111         4456.52             0.004457
```

**4456.5 MiB/s = 37.38 Gb/s = 4.67 GB/s**

### Striping proof — concurrent devices

```
rdma0 + rdma1 @1MiB concurrent:  18.64 + 18.69 = 37.3 Gb/s total
rdma0         @1MiB alone:                          37.4 Gb/s
```

Same ceiling ⇒ one QP already spans both rails. If striping were *off*,
two devices would show ~18–19 Gb/s each and one device ~19 Gb/s.

### Latency — RDMA WRITE round-trip (ping-pong), 2 B, 20 000 iters

| metric | µs |
|---|---|
| t_min | 5.75 |
| t_typical | **6.03** |
| t_avg | 6.09 |
| t_stdev | 0.34 |
| p99 | 6.77 |
| p99.9 | 11.11 |
| t_max | 23.90 |

Round-trip; one-way ≈ **3 µs**. Tight distribution.

### Latency — RDMA SEND round-trip, same params

| metric | µs |
|---|---|
| t_min | 6.01 |
| t_typical | 7.79 |
| p99 | 16.47 |
| p99.9 | 331.02 |
| t_max | 36 318 |

SEND has a **fat tail**. Same wire as WRITE, so this is the receive path
(recv-side wakeup / scheduling), not the link. `nhi_interrupt_throttle_ns=0`,
so it is not throttling-induced — it is inherent to the SEND/recv
completion path. **RDMA WRITE is the verb to use** for latency-sensitive
traffic (gufo's TP exchanges and NCCL/RCCL both use WRITE).

Benchmark noise warning: running the peer over a live SSH session inflates
tail latency badly. Earlier SEND runs showed t_max 37 ms purely from
competing ssh/scheduling load. Idle the hosts for clean numbers.

---

## 8. DKMS setup (the durable fix)

### The failure this prevents

The module is **out-of-tree**. On a kernel upgrade it is not rebuilt unless
something rebuilds it. Without DKMS:

```
$ modprobe thunderbolt_ibverbs
modprobe: FATAL: Module thunderbolt_ibverbs not found in directory /lib/modules/7.2.9-2-cachyos
```

Symptom: `ib_core` loads, `/sys/class/infiniband/` is **empty**,
`ibv_devices` lists nothing, `lsmod` shows `thunderbolt_ibverbs` absent.
Both hosts must be fixed. This is exactly what happened on the
`7.2.8-2 → 7.2.9-2` upgrade:

```
/lib/modules/7.2.8-2-cachyos/updates/thunderbolt_ibverbs.ko.zst   ← present
/lib/modules/7.2.9-2-cachyos/updates/                             ← missing
```

### The DKMS gotcha

The repo's stock `dkms.conf` contains:

```
MAKE[0]="make KVER=${kernelver}"
```

That invokes **gcc** and dies on the clang-flag error from §4 — DKMS would
fail on every CachyOS kernel upgrade. `MAKE[0]` must pass `LLVM=1` when,
and only when, the *target* kernel was clang-built.

### Implementation

**1. Install DKMS**

```sh
sudo pacman -S --needed dkms        # 3.4.3-1 on cachyos
```

**2. Compiler-detecting build wrapper** — `/usr/local/sbin/tbv-dkms-build`

Inspects the target kernel's `compile.h` and adds `LLVM=1` if that kernel
was built with clang. Keeps `dkms.conf` a one-liner and makes the logic
testable on its own.

**3. Point `dkms.conf` at the wrapper**

```
MAKE[0]="/usr/local/sbin/tbv-dkms-build ${kernelver}"
```

**4. Register, build, install**

```sh
cd /home/ugo/software/donnerkeule
make KVER=$(uname -r) clean          # drop stale .o — DKMS copies the tree verbatim
sudo dkms add .
sudo dkms build  thunderbolt-ibverbs/0.4.0 -k $(uname -r)
sudo dkms install thunderbolt-ibverbs/0.4.0 -k $(uname -r)
dkms status
```

`AUTOINSTALL="yes"` in `dkms.conf` plus the DKMS pacman hook rebuilds the
module automatically for every newly installed kernel.

**5. Retire the hand-installed copy**

`make modules_install` put the module in `updates/`; DKMS installs to
`updates/dkms/`. Two copies of the same module name is an ambiguity — let
DKMS own it:

```sh
sudo rm -f /lib/modules/$(uname -r)/updates/thunderbolt_ibverbs.ko.zst
sudo depmod -a
modprobe --show-depends thunderbolt_ibverbs   # must resolve under updates/dkms/
```

Do **not** unload a live module just to test — it drops the RDMA links.
Verify by file path and `modprobe --show-depends` instead.

**6. Repeat on the peer host.**

### Secure Boot note

DKMS signs the module with its own generated MOK key (`signer: DKMS module
signing key`). This is harmless **only while Secure Boot is off** — which it
is on both hosts:

```
[    0.007365] Secure boot disabled
$ cat /sys/kernel/security/lockdown
[none] integrity confidentiality
```

If Secure Boot is ever enabled, the DKMS MOK must be enrolled
(`mokutil --import /var/lib/dkms/mok.pub`) or the module fails at load
with `Required key not available`.

### Autoinstall hook

The `dkms` package installs the pacman hooks that make this survive kernel
upgrades:

```
/usr/share/libalpm/hooks/70-dkms-install.hook
/usr/share/libalpm/hooks/70-dkms-upgrade.hook
/usr/share/libalpm/hooks/71-dkms-remove.hook
```

They trigger on `usr/lib/modules/*/build/include/` — i.e. on every
`linux-*-headers` install/upgrade — and run `dkms install`, which rebuilds
every module with `AUTOINSTALL="yes"` for the new kernel.

If a hook did not fire for a newly installed kernel:

```sh
sudo dkms autoinstall -k <new-kernel-version>
```

---

## 8b. Implementation status — DONE on both hosts

Applied to `strix-halo-1` and `strix-halo-2`, kernel `7.2.9-2-cachyos`:

| item | state |
|---|---|
| `dkms` | 3.4.3-1 installed (from `cachyos` repo) |
| `/usr/local/sbin/tbv-dkms-build` | installed, mode 0755, root:root |
| `dkms.conf` `MAKE[0]` | `"/usr/local/sbin/tbv-dkms-build ${kernelver}"` |
| DKMS registration | `thunderbolt-ibverbs/0.4.0, 7.2.9-2-cachyos, x86_64: installed` |
| module path | `/usr/lib/modules/<KVER>/updates/dkms/thunderbolt_ibverbs.ko.zst` |
| hand-installed copy in `updates/` | removed, 0 stale `modules.dep` refs |
| Secure Boot | disabled, lockdown `[none]` |

Verified `modprobe` resolution picks up the DKMS copy **with the correct
options** still applied from `/etc/modprobe.d/`:

```
$ modprobe --show-depends thunderbolt_ibverbs | grep thunderbolt_ibverbs.ko
insmod /lib/modules/7.2.9-2-cachyos/updates/dkms/thunderbolt_ibverbs.ko.zst \
  profile=linux_perf tbnet=prefer_rdma lanes=2 bind_services=1 allocate_rings=1 \
  start_rings=1 negotiate_native=1 enable_tunnels=1 register_verbs=1 \
  roce_netdev=tbv0 native_write_striping=1
```

Post-migration regression test — no change in performance:

```
before DKMS:  4456.52 MiB/s  (37.38 Gb/s)
after  DKMS:  4457.87 MiB/s  (37.39 Gb/s)
```

All four ports `4: ACTIVE` on both hosts. The module was never unloaded
during the migration, so the RDMA links stayed up throughout.

### To verify after the next kernel upgrade

```sh
dkms status
# expect a line for the NEW kernel version, "installed"
ls /lib/modules/$(uname -r)/updates/dkms/thunderbolt_ibverbs.ko.zst
ls /sys/class/infiniband/        # expect: usb4_rdma0 1 5 6
```

---

## 9. Recovery cheat-sheet

| Symptom | Cause | Fix |
|---|---|---|
| `Module thunderbolt_ibverbs not found in directory /lib/modules/<new>` | kernel upgraded, module not rebuilt | §8, or rebuild manually §4 |
| `unrecognized command-line option '-fms-anonymous-structs'` | built with gcc against clang kernel | add `LLVM=1` |
| Module loads, `/sys/class/infiniband/` empty | peer has no module loaded / no cables | load on **both** hosts |
| `ib_write_bw` hangs with no output | wrong GID index | `-x 1` |
| `Couldn't connect to 10.77.0.x:18515` | rendezvous over `tbv0` | use `eno1` IP |
| ~32 Gb/s instead of ~37 Gb/s | default 64 KiB message | `-s 1048576` |
| Empty output from remote `pkill`/start | pattern matched own shell | `pkill -f "[i]b_write_bw"` |
| `ibv_devices` empty but `/sys/class/infiniband` populated | provider ABI mismatch | match `libusb4_rdma-rdmavNN.so` to rdma-core |
| Fewer rails than expected | `thunderbolt_net` grabbed a ring | keep `blacklist thunderbolt_net` |

---

## 10. Files touched on each host

```
/etc/modprobe.d/thunderbolt-ibverbs.conf       module options + thunderbolt_net blacklist
/etc/modules-load.d/thunderbolt-ibverbs.conf   load at boot
/etc/libibverbs.d/usb4_rdma.driver            provider registration (from package)
/usr/lib/libibverbs/libusb4_rdma-rdmav65.so   provider (from package)
/usr/local/sbin/tbv-dkms-build                compiler-detecting build wrapper  [DKMS]
/usr/src/thunderbolt-ibverbs-0.4.0/           DKMS source copy               [DKMS]
/lib/modules/<KVER>/updates/dkms/thunderbolt_ibverbs.ko.zst   built module  [DKMS]
docs/strix-halo-thunderbolt-rdma.md           this runbook
docs/tbv-bench.sh                             benchmark harness                [new]
docs/tbv-verify.sh                            one-shot stack verification    [new]
```

---

## 11. Quick reference — full rebuild, one host

```sh
cd /home/ugo/software/donnerkeule
grep LINUX_COMPILER /usr/lib/modules/$(uname -r)/build/include/generated/compile.h
make KVER=$(uname -r) clean
make KVER=$(uname -r) LLVM=1 modules
sudo make KVER=$(uname -r) LLVM=1 modules_install
sudo depmod -a
sudo modprobe thunderbolt_ibverbs
ls /sys/class/infiniband/          # expect: usb4_rdma0 1 5 6
```

## 12. One-shot verification

After a reboot, a kernel upgrade, or any doubt — run **one command** on
either host:

```sh
/home/ugo/software/donnerkeule/docs/tbv-verify.sh
```

It probes the local machine **and** the peer over SSH, asserts every layer,
and prints `PASS` / `WARN` / `FAIL` per check with a final tally. Exit code
is `0` only when nothing failed.

```
── Kernel & DKMS (both hosts) ──
  PASS  local dkms installed for kernel      thunderbolt-ibverbs/0.4.0, 7.2.9-2-cachyos, x86_64: installed
  PASS  local module in updates/dkms         yes
  PASS  local no stale hand-installed module no
  PASS  local modprobe -> updates/dkms       updates/dkms/thunderbolt_ibverbs.ko.zst
  PASS  local thunderbolt_net blacklisted    yes
  ...
── Module & IB devices (both hosts) ──
  PASS  local 4 IB devices                  usb4_rdma0 usb4_rdma1 usb4_rdma5 usb4_rdma6
  PASS  local all ports ACTIVE/LinkUp       usb4_rdma0=4:ACTIVE/LinkUp ...
── Traffic test ──
  PASS  striped WRITE bandwidth             4457.96 MiB/s = 37.40 Gb/s (min 34)
        concurrent usb4_rdma0+usb4_rdma1    2221 + 2233 MiB/s = 37.36 Gb/s (1.00x single)
  PASS  striping proof (shared ceiling)     1.00x — one QP spans both rails
  PASS  WRITE round-trip latency            6.05 us typical (max 12)

 RESULT: 42 pass, 2 warn
```

### What it checks

| Layer | Checks (per host) |
|---|---|
| DKMS | registered + installed for the *running* kernel; module present in `updates/dkms/`; no stale hand-installed copy; wrapper present |
| Persistence | `modules-load.d` and `modprobe.d` configs present; `thunderbolt_net` blacklisted |
| Params | `lanes=2` + `native_write_striping=1` actually applied by modprobe |
| Runtime | module loaded; 4 IB devices; all ports `ACTIVE`/`LinkUp` |
| Addressing | `tbv0` present, administratively UP, in `10.77.0.0/24`, addresses differ across hosts |
| GID | index 1 is RoCE v2 and IPv4-mapped |
| Userspace | `libusb4_rdma-rdmav65.so` present; `ibv_devices` enumerates 4; stale provider libs flagged |
| Posture | lockdown state; `thunderbolt_net` not holding a DMA ring |
| Traffic | striped WRITE bandwidth, striping proof, WRITE latency |

### Options

```sh
docs/tbv-verify.sh --no-bench      # config only, no traffic (fast, ~5s)
docs/tbv-verify.sh --verbose       # dump raw probe output for both hosts
docs/tbv-verify.sh --peer-ssh strix-halo-1.local --peer-ip 192.168.1.22
docs/tbv-verify.sh --dur 15 --iters 20000    # longer, more accurate
MIN_GBPS=36 MAX_TYP_LAT_US=8 docs/tbv-verify.sh   # tighten thresholds
```

Tunable thresholds: `MIN_GBPS` (default 34), `MAX_TYP_LAT_US` (default 12).

### The striping proof is a real assertion

The concurrent test launches **both** clients simultaneously and compares the
sum against the single-device result:

- **~1.0x** → PASS. One QP already spans both rails, so two devices share
  the same ceiling. This is the correct, striped behaviour.
- **>1.35x** → FAIL. The devices are running independently, meaning
  write striping is *not* active.

> Beware a false pass here: if the two clients run *sequentially* instead of
> concurrently, each gets full bandwidth and the sum reads ~2x, which looks
> like a doubling that does not exist. The script backgrounds both clients
> and waits for both specifically to avoid this.


## 13. Quick reference — benchmark pair

Use the bundled script; it already handles the GID index, the `eno1`
rendezvous, and the message-size trap from §6:

```sh
/home/ugo/software/donnerkeule/docs/tbv-bench.sh
```

Run it on the client host. It starts the perftest server on the peer over
SSH, runs the sweep, the striping proof, and both latency tests, and prints
Gb/s. Idle both hosts first for clean latency tails.

```sh
# override for the reverse direction
docs/tbv-bench.sh --peer-ssh strix-halo-1.local --peer-ip 192.168.1.22

# long-run bandwidth only, single size
docs/tbv-bench.sh --dur 30 --sizes 1048576 --no-latency

docs/tbv-bench.sh --help    # all flags; every value is also an env var
```

Equivalent manual commands:

```sh
# strix-halo-2 (server), idle the machine first
ib_write_bw -d usb4_rdma0 -x 1 -F -D 15 -s 1048576
ib_write_lat -d usb4_rdma0 -x 1 -F -n 20000

# strix-halo-1 (client)
ib_write_bw  -d usb4_rdma0 -x 1 -F -D 15 -s 1048576 192.168.1.92   # → ~37.4 Gb/s
ib_write_lat -d usb4_rdma0 -x 1 -F -n 20000 192.168.1.92            # → ~6.0 µs RTT
```
