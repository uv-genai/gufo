# Monitoring requests

To monitor the number of concurrent requests do:
```
```
watch -n1 'curl -s http://strix-halo-1.local:${PORT}/metrics | grep -E "requests_(processing|deferred) "'
```
```

Slots alternative:
```
```
curl -s http://strix-halo-1.local:${PORT}/slots | jq 'map(.state) | group_by(.) | map({state: .[0], n: length})'
```
```
