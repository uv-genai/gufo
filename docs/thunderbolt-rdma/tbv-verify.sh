#!/usr/bin/env bash
#
# tbv-verify.sh — one-shot health check for the thunderbolt-ibverbs RDMA pair.
#
# Companion to docs/strix-halo-thunderbolt-rdma.md
#
# Run on ONE host. It probes the local machine and the peer over SSH, asserts
# every layer of the stack, runs a short benchmark, and prints PASS/WARN/FAIL.
#
#   ./tbv-verify.sh                    # full check, strix-halo-1 -> strix-halo-2
#   ./tbv-verify.sh --no-bench        # config only, skip traffic (fast)
#   ./tbv-verify.sh --peer-ssh strix-halo-1.local --peer-ip 192.168.1.22
#   ./tbv-verify.sh --verbose         # show raw probe output too
#
# Exit code: 0 = all pass (warnings allowed), 1 = one or more FAIL.
#
set -uo pipefail

PEER_SSH="${PEER_SSH:-strix-halo-2.local}"
PEER_IP="${PEER_IP:-192.168.1.92}"     # peer MANAGEMENT ip (eno1), NOT tbv0
GID="${GID:-1}"
DEV="${DEV:-usb4_rdma0}"
DEV2="${DEV2:-usb4_rdma1}"
BW_DUR="${BW_DUR:-8}"
LAT_ITERS="${LAT_ITERS:-5000}"
MIN_GBPS="${MIN_GBPS:-34}"             # fail below this striped WRITE bandwidth
MAX_TYP_LAT_US="${MAX_TYP_LAT_US:-12}" # fail above this WRITE round-trip typical
RUN_BENCH=1
VERBOSE=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --peer-ssh) PEER_SSH="$2"; shift 2 ;;
        --peer-ip)  PEER_IP="$2"; shift 2 ;;
        --gid)      GID="$2"; shift 2 ;;
        --dev)      DEV="$2"; shift 2 ;;
        --dev2)     DEV2="$2"; shift 2 ;;
        --dur)      BW_DUR="$2"; shift 2 ;;
        --iters)    LAT_ITERS="$2"; shift 2 ;;
        --no-bench) RUN_BENCH=0; shift ;;
        --verbose)  VERBOSE=1; shift ;;
        -h|--help)
            sed -n '2,20p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

# ---------------------------------------------------------------- output
if [[ -t 1 ]]; then
    C_G=$'\e[32m'; C_Y=$'\e[33m'; C_R=$'\e[31m'; C_B=$'\e[1m'; C_0=$'\e[0m'
else
    C_G=""; C_Y=""; C_R=""; C_B=""; C_0=""
fi

N_PASS=0; N_WARN=0; N_FAIL=0
ok()   { N_PASS=$((N_PASS+1)); printf "  ${C_G}PASS${C_0}  %-46s %s\n" "$1" "${2:-}"; }
warn() { N_WARN=$((N_WARN+1)); printf "  ${C_Y}WARN${C_0}  %-46s %s\n" "$1" "${2:-}"; }
bad()  { N_FAIL=$((N_FAIL+1)); printf "  ${C_R}FAIL${C_0}  %-46s %s\n" "$1" "${2:-}"; }
info() { printf "        %-46s %s\n" "$1" "${2:-}"; }
section() { printf "\n${C_B}── %s ──${C_0}\n" "$1"; }

# assert <label> <expected> <actual> [warn-only]
assert() {
    local label="$1" exp="$2" act="$3" soft="${4:-0}"
    if [[ "$act" == "$exp" ]]; then ok "$label" "$act"
    elif [[ "$soft" == "1" ]]; then warn "$label" "want '$exp' got '${act:-<empty>}'"
    else bad "$label" "want '$exp' got '${act:-<empty>}'"
    fi
}

# assert_nonempty <label> <actual>
assert_nonempty() {
    local label="$1" act="$2"
    if [[ -n "$act" && "$act" != "0" && "$act" != "no" ]]; then ok "$label" "$act"
    else bad "$label" "<empty/none>"
    fi
}

# ---------------------------------------------------------------- probe
# Runs identically on local and remote; emits KEY=value for the assertions.
read -r -d '' PROBE <<'PROBE_EOF'
K=$(uname -r)
echo "KERNEL=$K"

# DKMS
ds=$(dkms status 2>/dev/null | grep "installed" | grep "$K" | head -1)
echo "DKMS_STATUS=${ds:-NONE}"

# module file locations
mf="/usr/lib/modules/$K/updates/dkms/thunderbolt_ibverbs.ko.zst"
[[ -f "$mf" ]] && echo "DKMS_MODULE=yes" || echo "DKMS_MODULE=no"
stale=$(ls /usr/lib/modules/$K/updates/thunderbolt_ibverbs.ko.zst 2>/dev/null)
[[ -n "$stale" ]] && echo "STALE_MODULE=yes" || echo "STALE_MODULE=no"

# modprobe resolution + options actually applied
mp=$(modprobe --show-depends thunderbolt_ibverbs 2>/dev/null | grep 'thunderbolt_ibverbs.ko')
echo "MODPROBE_PATH=$(echo "$mp" | grep -o 'updates/dkms/thunderbolt_ibverbs.ko.zst' | head -1)"
echo "MODPROBE_OPTS=$(echo "$mp" | sed 's/.*\.ko\.zst //')"

# persistence configs
[[ -f /etc/modules-load.d/thunderbolt-ibverbs.conf ]] && echo "LOAD_CONF=yes" || echo "LOAD_CONF=no"
[[ -f /etc/modprobe.d/thunderbolt-ibverbs.conf ]] && echo "OPTS_CONF=yes" || echo "OPTS_CONF=no"
grep -q 'blacklist thunderbolt_net' /etc/modprobe.d/thunderbolt-ibverbs.conf 2>/dev/null \
    && echo "TBNET_BLACKLIST=yes" || echo "TBNET_BLACKLIST=no"
[[ -x /usr/local/sbin/tbv-dkms-build ]] && echo "WRAPPER=yes" || echo "WRAPPER=no"

# loaded?
if lsmod | grep -q '^thunderbolt_ibverbs '; then echo "LOADED=yes"; else echo "LOADED=no"; fi

# devices
devs=$(ls /sys/class/infiniband/ 2>/dev/null | tr '\n' ' ')
echo "DEVICES=${devs}"
echo "DEVICE_COUNT=$(ls /sys/class/infiniband/ 2>/dev/null | wc -l | tr -d ' ')"

# port states
st=""
for d in /sys/class/infiniband/usb4_rdma*; do
    [[ -d $d ]] || continue
    s=$(cat "$d/ports/1/state" 2>/dev/null | tr -d ' ')
    p=$(cat "$d/ports/1/phys_state" 2>/dev/null | tr -d ' ' | sed 's/^[0-9]*://')
    st="$st$(basename $d)=${s}/${p} "
done
echo "PORT_STATES=$st"

# tbv0
tbip=$(ip -4 -br addr show tbv0 2>/dev/null | awk '{print $3}' | head -1)
echo "TBV0_IP=${tbip:-NONE}"
echo "TBV0_FLAGS=$(ip -br link show tbv0 2>/dev/null | awk '{print $NF}')"

# GID index 1 must be the IPv4-mapped one
g1=$(cat /sys/class/infiniband/usb4_rdma0/ports/1/gids/1 2>/dev/null)
gt1=$(cat /sys/class/infiniband/usb4_rdma0/ports/1/gid_attrs/types/1 2>/dev/null)
echo "GID1=${g1}"
echo "GID1_TYPE=${gt1}"

# userspace provider
echo "IBV_COUNT=$(ibv_devices 2>/dev/null | grep -c usb4_rdma)"
[[ -e /usr/lib/libibverbs/libusb4_rdma-rdmav65.so ]] && echo "PROVIDER65=yes" || echo "PROVIDER65=no"
stalep=$(ls /usr/lib/libibverbs/libusb4_rdma-rdmav5*.so 2>/dev/null | tr '\n' ' ')
echo "PROVIDER_STALE=${stalep:-none}"

# informational
if [[ -r /sys/kernel/security/lockdown ]]; then
    echo "LOCKDOWN=$(cat /sys/kernel/security/lockdown | awk -F'[][]' '{print $2}')"
else
    echo "LOCKDOWN=unknown"
fi
echo "KCOMPILER=$(sed -n 's/^#define LINUX_COMPILER[[:space:]]*"\(.*\)"/\1/p' /usr/lib/modules/$K/build/include/generated/compile.h 2>/dev/null)"
echo "TBNET_LOADED=$(lsmod | grep -c '^thunderbolt_net ' | tr -d ' ')"
PROBE_EOF

probe_local() { bash -c "$PROBE"; }
probe_peer()  { ssh -o BatchMode=yes -o ConnectTimeout=10 "$PEER_SSH" 'bash -s' <<< "$PROBE"; }

LOCAL_OUT="$(probe_local 2>/dev/null)"
PEER_OUT="$(probe_peer  2>/dev/null)"
PEER_OK=$?

L() { echo "$LOCAL_OUT" | sed -n "s/^${1}=//p" | head -1; }
P() { echo "$PEER_OUT"  | sed -n "s/^${1}=//p" | head -1; }

printf "${C_B}==============================================================${C_0}\n"
printf "${C_B} thunderbolt-ibverbs stack verification${C_0}\n"
printf "${C_B} local : %s${C_0}\n" "$(hostname)"
printf "${C_B} peer  : %s (%s)${C_0}\n" "$PEER_SSH" "$PEER_IP"
printf "${C_B}==============================================================${C_0}\n"

if [[ $VERBOSE -eq 1 ]]; then
    echo; echo "--- raw local probe ---"; echo "$LOCAL_OUT"
    echo "--- raw peer probe ---";  echo "$PEER_OUT"
fi

# ---------------------------------------------------------------- 1. kernel + dkms
section "Kernel & DKMS (both hosts)"
for side in local peer; do
    if [[ $side == local ]]; then g=L; else g=P; fi
    [[ $side == peer && $PEER_OK -ne 0 ]] && { bad "peer reachable via $PEER_SSH" "unreachable"; continue; }
    k="$($g KERNEL)"
    assert "$side kernel"        "$(L KERNEL)" "$k" 1
    dk="$($g DKMS_STATUS)"
    if [[ -n "$dk" && "$dk" == *"$k"* && "$dk" == *"installed"* ]]; then
        ok "$side dkms installed for kernel" "$dk"
    else
        bad "$side dkms installed for kernel" "${dk:-NONE}"
    fi
    assert "$side module in updates/dkms" "yes" "$($g DKMS_MODULE)"
    assert "$side no stale hand-installed module" "no" "$($g STALE_MODULE)"
    assert "$side modprobe -> updates/dkms" "updates/dkms/thunderbolt_ibverbs.ko.zst" "$($g MODPROBE_PATH)"
    assert "$side modules-load.d present" "yes" "$($g LOAD_CONF)"
    assert "$side modprobe.d present" "yes" "$($g OPTS_CONF)"
    assert "$side thunderbolt_net blacklisted" "yes" "$($g TBNET_BLACKLIST)"
    assert "$side tbv-dkms-build wrapper" "yes" "$($g WRAPPER)"
done
info "kernel compiler" "$(L KCOMPILER)"

# striping param must be live on both
for side in local peer; do
    [[ $side == peer && $PEER_OK -ne 0 ]] && continue
    if [[ $side == local ]]; then o="$(L MODPROBE_OPTS)"; else o="$(P MODPROBE_OPTS)"; fi
    if [[ "$o" == *"native_write_striping=1"* && "$o" == *"lanes=2"* ]]; then
        ok "$side striping params live" "lanes=2 native_write_striping=1"
    else
        bad "$side striping params live" "missing in: ${o:-<none>}"
    fi
done

# ---------------------------------------------------------------- 2. module + devices
section "Module & IB devices (both hosts)"
for side in local peer; do
    [[ $side == peer && $PEER_OK -ne 0 ]] && continue
    if [[ $side == local ]]; then g=L; else g=P; fi
    assert "$side module loaded" "yes" "$($g LOADED)"
    cnt="$($g DEVICE_COUNT)"
    if [[ "$cnt" == "4" ]]; then ok "$side 4 IB devices" "$($g DEVICES)"
    else bad "$side 4 IB devices" "found $cnt: $($g DEVICES)" ; fi
    ps="$($g PORT_STATES)"
    nact=$(echo "$ps" | grep -o '4:ACTIVE' | wc -l | tr -d ' ')
    if [[ "$nact" == "4" ]]; then ok "$side all ports ACTIVE/LinkUp" "$ps"
    else bad "$side all ports ACTIVE/LinkUp" "$nact/4 active — $ps"; fi
done

# ---------------------------------------------------------------- 3. addressing
section "Addressing & GID (both hosts)"
lip="$(L TBV0_IP)"; pip="$(P TBV0_IP)"
for side in local peer; do
    [[ $side == peer && $PEER_OK -ne 0 ]] && continue
    if [[ $side == local ]]; then g=L; else g=P; fi
    ip="$($g TBV0_IP)"
    if [[ "$ip" == 10.77.0.* ]]; then ok "$side tbv0 has 10.77.0.0/24 address" "$ip"
    else bad "$side tbv0 has 10.77.0.0/24 address" "${ip:-NONE}"; fi
    fl="$($g TBV0_FLAGS)"
    if [[ ",${fl//[<>]/}," == *",UP,"* ]]; then ok "$side tbv0 administratively UP" "$fl"
    else bad "$side tbv0 administratively UP" "${fl:-<missing>}"; fi
    assert "$side GID1 is RoCE v2" "RoCE v2" "$($g GID1_TYPE)"
    g1="$($g GID1)"
    if [[ "$g1" == *"ffff:"* ]]; then ok "$side GID1 IPv4-mapped" "$g1"
    else bad "$side GID1 IPv4-mapped" "$g1"; fi
done
if [[ -n "$lip" && -n "$pip" && "$lip" != "$pip" ]]; then
    ok "tbv0 addresses differ across hosts" "$lip vs $pip"
else
    bad "tbv0 addresses differ across hosts" "$lip vs $pip"
fi

# ---------------------------------------------------------------- 4. userspace
section "Userspace provider (both hosts)"
for side in local peer; do
    [[ $side == peer && $PEER_OK -ne 0 ]] && continue
    if [[ $side == local ]]; then g=L; else g=P; fi
    assert "$side provider rdmav65 present" "yes" "$($g PROVIDER65)"
    c="$($g IBV_COUNT)"
    if [[ "$c" == "4" ]]; then ok "$side ibv_devices enumerates 4" "$c"
    else bad "$side ibv_devices enumerates 4" "found $c"; fi
    if [[ "$($g PROVIDER_STALE)" == "none" ]]; then ok "$side no stale provider libs" "clean"
    else warn "$side no stale provider libs" "$($g PROVIDER_STALE)"; fi
done

# ---------------------------------------------------------------- 5. security posture
section "Security posture (informational)"
ld="$(L LOCKDOWN)"
if [[ "$ld" == "none" ]]; then
    info "kernel lockdown" "$ld — DKMS MOK signing not enforced"
else
    warn "kernel lockdown" "$ld — if Secure Boot is on, enroll /var/lib/dkms/mok.pub"
fi
if [[ "$(L TBNET_LOADED)" == "0" ]]; then
    info "thunderbolt_net" "not loaded — both DMA rings free for RDMA"
else
    bad "thunderbolt_net" "LOADED — it steals a DMA ring, breaks a rail"
fi

# ---------------------------------------------------------------- 6. benchmark
if [[ $RUN_BENCH -eq 1 ]]; then
    section "Traffic test — RDMA WRITE over ${DEV}"
    if [[ ! -d /sys/class/infiniband/$DEV ]]; then
        bad "local device $DEV present" "missing"
    else
        timeout $((BW_DUR + 60)) ssh -o BatchMode=yes "$PEER_SSH" \
            "ib_write_bw -d $DEV -x $GID -F -D $BW_DUR -s 1048576" >/tmp/tbv_v_bw.log 2>&1 &
        SP=$!
        sleep 4
        bw=$(timeout $((BW_DUR + 40)) ib_write_bw -d "$DEV" -x "$GID" -F -D "$BW_DUR" \
                -s 1048576 "$PEER_IP" 2>/dev/null | awk '/^ 1048576/ {print $4}')
        wait $SP 2>/dev/null || true
        if [[ -z "$bw" ]]; then
            bad "bandwidth measured" "no result — check peer server / rendezvous"
        else
            gbps=$(awk -v v="$bw" 'BEGIN{printf "%.2f", v*8*1.048576/1000}')
            if awk -v a="$gbps" -v m="$MIN_GBPS" 'BEGIN{exit !(a>=m)}'; then
                ok "striped WRITE bandwidth" "${bw} MiB/s = ${gbps} Gb/s (min ${MIN_GBPS})"
            else
                bad "striped WRITE bandwidth" "${bw} MiB/s = ${gbps} Gb/s (min ${MIN_GBPS})"
            fi
        fi

        # striping proof: two devices running TRULY concurrently should SUM
        # to the same ceiling a single device reaches, because one QP already
        # spans both rails. Independent devices would show ~2x instead.
        timeout $((BW_DUR + 60)) ssh -o BatchMode=yes "$PEER_SSH" \
            "ib_write_bw -d $DEV -x $GID -F -D $BW_DUR -s 1048576 -p 18611" >/tmp/tbv_v_c1.log 2>&1 &
        timeout $((BW_DUR + 60)) ssh -o BatchMode=yes "$PEER_SSH" \
            "ib_write_bw -d $DEV2 -x $GID -F -D $BW_DUR -s 1048576 -p 18612" >/tmp/tbv_v_c2.log 2>&1 &
        sleep 4
        # launch BOTH clients at the same time, then wait — sequential runs
        # would each get full bandwidth and fake a 2x sum
        timeout $((BW_DUR + 40)) ib_write_bw -d "$DEV"  -x "$GID" -F -D "$BW_DUR" \
            -s 1048576 -p 18611 "$PEER_IP" >/tmp/tbv_v_s1.txt 2>/dev/null &
        cc1=$!
        timeout $((BW_DUR + 40)) ib_write_bw -d "$DEV2" -x "$GID" -F -D "$BW_DUR" \
            -s 1048576 -p 18612 "$PEER_IP" >/tmp/tbv_v_s2.txt 2>/dev/null &
        cc2=$!
        wait $cc1 2>/dev/null; wait $cc2 2>/dev/null
        s1=$(awk '/^ 1048576/ {print $4}' /tmp/tbv_v_s1.txt)
        s2=$(awk '/^ 1048576/ {print $4}' /tmp/tbv_v_s2.txt)
        if [[ -n "$s1" && -n "$s2" && -n "${bw:-}" ]]; then
            tot=$(awk -v a="$s1" -v b="$s2" 'BEGIN{printf "%.2f", (a+b)*8*1.048576/1000}')
            ratio=$(awk -v a="$s1" -v b="$s2" -v s="$bw" 'BEGIN{printf "%.2f", (a+b)/s}')
            info "concurrent ${DEV}+${DEV2}" \
                "$(awk -v a="$s1" -v b="$s2" 'BEGIN{printf "%.0f + %.0f MiB/s", a, b}') = ${tot} Gb/s (${ratio}x single)"
            if awk -v r="$ratio" 'BEGIN{exit !(r>=0.8 && r<=1.35)}'; then
                ok "striping proof (shared ceiling)" "${ratio}x — one QP spans both rails"
            elif awk -v r="$ratio" 'BEGIN{exit !(r>1.35)}'; then
                bad "striping proof (shared ceiling)" "${ratio}x — independent, striping NOT active"
            else
                warn "striping proof (shared ceiling)" "${ratio}x — below expected"
            fi
        else
            warn "concurrent striping test" "no result"
        fi

        # WRITE latency
        timeout 120 ssh -o BatchMode=yes "$PEER_SSH" \
            "ib_write_lat -d $DEV -x $GID -F -n $LAT_ITERS" >/tmp/tbv_v_lat.log 2>&1 &
        LP=$!
        sleep 4
        lat=$(timeout 90 ib_write_lat -d "$DEV" -x "$GID" -F -n "$LAT_ITERS" "$PEER_IP" 2>/dev/null \
              | awk '/^     2 |^ 2 / {print $5}')
        wait $LP 2>/dev/null || true
        if [[ -z "$lat" ]]; then
            warn "WRITE round-trip latency" "no result"
        elif awk -v a="$lat" -v m="$MAX_TYP_LAT_US" 'BEGIN{exit !(a<=m)}'; then
            ok "WRITE round-trip latency" "${lat} us typical (max ${MAX_TYP_LAT_US})"
        else
            bad "WRITE round-trip latency" "${lat} us typical (max ${MAX_TYP_LAT_US})"
        fi
    fi
else
    section "Traffic test"
    info "skipped" "--no-bench"
fi

# ---------------------------------------------------------------- summary
printf "\n${C_B}==============================================================${C_0}\n"
printf "${C_B} RESULT: ${C_G}${N_PASS} pass${C_0}"
[[ $N_WARN -gt 0 ]] && printf ", ${C_Y}${N_WARN} warn${C_0}"
[[ $N_FAIL -gt 0 ]] && printf ", ${C_R}${N_FAIL} fail${C_0}"
printf "${C_B}${C_0}\n"
if [[ $N_FAIL -eq 0 ]]; then
    printf " ${C_G}Stack healthy.${C_0} Reference: 37.4 Gb/s striped WRITE, ~6.0 us RTT\n"
    exit 0
else
    printf " ${C_R}${N_FAIL} check(s) failed.${C_0} See docs/strix-halo-thunderbolt-rdma.md §9 recovery table\n"
    exit 1
fi
