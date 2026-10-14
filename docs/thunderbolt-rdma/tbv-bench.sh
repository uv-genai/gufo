#!/usr/bin/env bash
#
# tbv-bench.sh — benchmark thunderbolt-ibverbs RDMA between two hosts.
#
# Companion to docs/strix-halo-thunderbolt-rdma.md
#
# Run this on the CLIENT host. It starts the perftest server on the peer over
# SSH, runs the client locally, and reports Gb/s.
#
# Defaults are for strix-halo-1 -> strix-halo-2. Override with flags or env.
#
# IMPORTANT — the three traps this script already handles for you:
#   1. GID index: always -x 1 (the IPv4-mapped GID). Index 0 is link-local
#      and the test hangs silently.
#   2. Rendezvous: perftest needs a TCP socket for its handshake. tbv0 does
#      NOT carry IP traffic (it is only a RoCE GID anchor), so the peer
#      address must be the MANAGEMENT interface (eno1), not 10.77.0.x.
#      The RDMA data still flows over Thunderbolt.
#   3. Message size: perftest defaults to 64 KiB, which sits ~15% below the
#      plateau. Use 1 MiB for headline bandwidth.
#
# Idle both hosts first. A live SSH session on the peer inflates latency tails
# badly (observed t_max 37 ms purely from competing scheduling load).
#
set -uo pipefail

PEER_SSH="${PEER_SSH:-strix-halo-2.local}"
PEER_IP="${PEER_IP:-192.168.1.92}"     # peer MANAGEMENT ip (eno1), NOT tbv0
DEV="${DEV:-usb4_rdma0}"
DEV2="${DEV2:-usb4_rdma1}"
GID="${GID:-1}"
BW_DUR="${BW_DUR:-15}"
LAT_ITERS="${LAT_ITERS:-20000}"
SIZES="${SIZES:-65536 262144 1048576 4194304}"
RUN_CONCURRENT=1
RUN_LATENCY=1
PORT_BASE="${PORT_BASE:-18515}"

usage() {
    cat <<EOF
Usage: $(basename "$0") [options]

  --peer-ssh HOST     ssh target for the peer        (default: $PEER_SSH)
  --peer-ip  IP       peer management IP, eno1     (default: $PEER_IP)
  --dev      DEV      primary IB device            (default: $DEV)
  --dev2     DEV      secondary device for striping proof (default: $DEV2)
  --gid      N        RoCE GID index, use 1        (default: $GID)
  --dur      SEC      bandwidth test duration      (default: $BW_DUR)
  --iters    N        latency iterations          (default: $LAT_ITERS)
  --sizes    "..."    space separated byte sizes  (default: "$SIZES")
  --no-concurrent     skip the two-device striping proof
  --no-latency        skip latency tests
  -h, --help

Examples:
  $(basename "$0")                                  # full suite, strix-halo-1 -> -2
  $(basename "$0") --peer-ssh strix-halo-1.local --peer-ip 192.168.1.22
  $(basename "$0") --sizes 1048576 --dur 30
EOF
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --peer-ssh)     PEER_SSH="$2"; shift 2 ;;
        --peer-ip)      PEER_IP="$2"; shift 2 ;;
        --dev)          DEV="$2"; shift 2 ;;
        --dev2)         DEV2="$2"; shift 2 ;;
        --gid)          GID="$2"; shift 2 ;;
        --dur)          BW_DUR="$2"; shift 2 ;;
        --iters)        LAT_ITERS="$2"; shift 2 ;;
        --sizes)        SIZES="$2"; shift 2 ;;
        --no-concurrent) RUN_CONCURRENT=0; shift ;;
        --no-latency)    RUN_LATENCY=0; shift ;;
        -h|--help)      usage ;;
        *) echo "unknown option: $1" >&2; usage ;;
    esac
done

# mib/s -> Gb/s
gbps() { awk -v v="$1" 'BEGIN{printf "%.2f", v*8*1.048576/1000}'; }

# start_server <dev> <port> <extra perftest args...>
# Holds an open SSH session in the background running the perftest server.
start_server() {
    local dev="$1" port="$2"; shift 2
    timeout $((BW_DUR + 90)) ssh -o BatchMode=yes "$PEER_SSH" \
        "ib_write_bw -d $dev -x $GID -F -p $port $*" \
        >"/tmp/tbv_srv_${port}.log" 2>&1 &
    SERVER_PIDS+=("$!")
}

start_lat_server() {
    local tool="$1" dev="$2" port="$3"; shift 3
    timeout 120 ssh -o BatchMode=yes "$PEER_SSH" \
        "$tool -d $dev -x $GID -F -p $port $*" \
        >"/tmp/tbv_srv_${port}.log" 2>&1 &
    SERVER_PIDS+=("$!")
}

SERVER_PIDS=()
cleanup() { for p in "${SERVER_PIDS[@]:-}"; do kill "$p" 2>/dev/null || true; done; }
trap cleanup EXIT INT TERM

echo "=============================================================="
echo " thunderbolt-ibverbs RDMA benchmark"
echo "   client : $(hostname)"
echo "   peer   : ${PEER_SSH} (${PEER_IP}, management iface)"
echo "   device : ${DEV}   gid: ${GID}   bw: ${BW_DUR}s   lat: ${LAT_ITERS} iters"
echo "=============================================================="

# ---------------------------------------------------------------- preflight
echo
echo "### Preflight"
if [[ ! -d /sys/class/infiniband/$DEV ]]; then
    echo "ERROR: /sys/class/infiniband/$DEV missing — module not loaded or no peer." >&2
    echo "       Load thunderbolt_ibverbs on BOTH hosts." >&2
    exit 1
fi
echo "local ports:"
for d in /sys/class/infiniband/usb4_rdma*; do
    printf "  %-14s %s\n" "$(basename "$d")" "$(cat "$d/ports/1/state" 2>/dev/null)"
done
echo "peer devices:"
timeout 20 ssh -o BatchMode=yes "$PEER_SSH" \
    'ls /sys/class/infiniband/ | tr "\n" " "; echo' || {
    echo "ERROR: cannot ssh to ${PEER_SSH}" >&2; exit 1; }

# ---------------------------------------------------------------- sweep
echo
echo "### Bandwidth sweep — RDMA WRITE, ${DEV}"
printf "%-12s %14s %10s\n" "size" "MiB/s" "Gb/s"
for size in $SIZES; do
    start_server "$DEV" "$PORT_BASE" "-D $BW_DUR -s $size"
    sleep 4
    timeout $((BW_DUR + 40)) ib_write_bw -d "$DEV" -x "$GID" -F -D "$BW_DUR" \
        -s "$size" -p "$PORT_BASE" "$PEER_IP" 2>/dev/null \
    | awk -v s="$size" '/^ [0-9]/ {printf "%-12s %14.2f\n", s, $4}' \
    | while read -r sz mib; do printf "%-12s %14s %10s\n" "$sz" "$mib" "$(gbps "$mib")"; done
    wait "${SERVER_PIDS[-1]}" 2>/dev/null || true
done

# --------------------------------------------------- striping proof
if [[ "$RUN_CONCURRENT" == "1" ]]; then
    echo
    echo "### Striping proof — ${DEV} and ${DEV2} concurrently @1MiB"
    echo "### (if striping is live these SUM to the single-device ceiling,"
    echo "###  rather than each reaching it independently)"
    p1=$((PORT_BASE + 1)); p2=$((PORT_BASE + 2))
    timeout $((BW_DUR + 90)) ssh -o BatchMode=yes "$PEER_SSH" \
        "ib_write_bw -d $DEV -x $GID -F -D $BW_DUR -s 1048576 -p $p1" >/tmp/tbv_c1.log 2>&1 &
    SERVER_PIDS+=("$!")
    timeout $((BW_DUR + 90)) ssh -o BatchMode=yes "$PEER_SSH" \
        "ib_write_bw -d $DEV2 -x $GID -F -D $BW_DUR -s 1048576 -p $p2" >/tmp/tbv_c2.log 2>&1 &
    SERVER_PIDS+=("$!")
    printf "%-12s %14s %10s\n" "device" "MiB/s" "Gb/s"
    sleep 5
    for pair in "$DEV:$p1" "$DEV2:$p2"; do
        dev="${pair%%:*}"; port="${pair##*:}"
        timeout $((BW_DUR + 40)) ib_write_bw -d "$dev" -x "$GID" -F -D "$BW_DUR" \
            -s 1048576 -p "$port" "$PEER_IP" 2>/dev/null \
        | awk -v s="$dev" '/^ [0-9]/ {printf "%-12s %14.2f\n", s, $4}' \
        | while read -r sz mib; do
              printf "%-12s %14s %10s\n" "$sz" "$mib" "$(gbps "$mib")"
          done &
    done
    wait
    echo "(sum the two lines above for aggregate striped bandwidth)"
fi

# ---------------------------------------------------------------- latency
if [[ "$RUN_LATENCY" == "1" ]]; then
    echo
    echo "### Latency — RDMA WRITE round-trip (ping-pong), 2 B"
    echo "### (one-way is roughly half; this is the verb TP inference uses)"
    p=$((PORT_BASE + 3))
    timeout 120 ssh -o BatchMode=yes "$PEER_SSH" \
        "ib_write_lat -d $DEV -x $GID -F -n $LAT_ITERS -p $p" >/tmp/tbv_wlat.log 2>&1 &
    SERVER_PIDS+=("$!")
    sleep 4
    timeout 90 ib_write_lat -d "$DEV" -x "$GID" -F -n "$LAT_ITERS" -p "$p" \
        "$PEER_IP" 2>/dev/null | tail -4

    echo
    echo "### Latency — RDMA SEND round-trip, 2 B"
    echo "### (expect a fat tail: receive-path wakeup, not the link)"
    p=$((PORT_BASE + 4))
    timeout 120 ssh -o BatchMode=yes "$PEER_SSH" \
        "ib_send_lat -d $DEV -x $GID -F -n $LAT_ITERS -p $p" >/tmp/tbv_slat.log 2>&1 &
    SERVER_PIDS+=("$!")
    sleep 4
    timeout 90 ib_send_lat -d "$DEV" -x "$GID" -F -n "$LAT_ITERS" -p "$p" \
        "$PEER_IP" 2>/dev/null | tail -4
fi

echo
echo "=============================================================="
echo " Reference (measured, idle hosts, kernel 7.2.9-2-cachyos):"
echo "   WRITE 1MiB  : 4457 MiB/s = 37.4 Gb/s"
echo "   WRITE lat   : t_min 5.75  t_typ 6.03  p99 6.77  p99.9 11.11 us"
echo "   SEND  lat   : t_min 6.01  t_typ 7.79  p99 16.47 p99.9 331.02 us"
echo "=============================================================="
