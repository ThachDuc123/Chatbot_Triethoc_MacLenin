param(
  [int]$Port = 5002,
  [string]$HostName = "127.0.0.1"
)

$ErrorActionPreference = "Stop"

try {
  $r = Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 -Uri "http://$HostName`:$Port/"
  "OK status=$($r.StatusCode)"
  $r.Content
} catch {
  "FAIL: $($_.Exception.Message)"
}
