#!/usr/bin/env bash
watch -n1 'tail -2000 ~/gufo-tp2-swap.log | grep -oE "request=r[0-9]+ event=(received|completed)" \
     | awk "{split(\$1,a,\"=\"); if (\$2==\"event=received\") open[a[2]]=1; else delete open[a[2]]} \
     END {for (k in open) print \"in-flight:\", k}"'
