param(
  [int]$Port = 5002
)

$ErrorActionPreference = "Stop"

try {
  $r = Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 -Uri "http://localhost:$Port/"
  "OK status=$($r.StatusCode)"
} catch {
  "FAIL: $($_.Exception.Message)"
}
