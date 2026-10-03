param(
  [int]$Port = 5002
)

# Ensure UTF-8 output/input so Vietnamese isn't mangled on Windows PowerShell 5.1
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
try { chcp 65001 | Out-Null } catch { }

$ErrorActionPreference = "Stop"

function Invoke-Chat([string]$text) {
  $payload = @(
    @{ role = "user"; content = $text }
  )
  # Ensure we send a JSON ARRAY (not an object) and UTF-8 bytes.
  $body = ($payload | ConvertTo-Json -Depth 5)
  $bytes = [System.Text.Encoding]::UTF8.GetBytes($body)

  $r = Invoke-RestMethod -Method Post -Uri "http://localhost:$Port/api/search" -ContentType "application/json; charset=utf-8" -Body $bytes
  "USER: $text"
  "BOT : $($r.content)"
  ""
}

Invoke-Chat "Triết học có chức năng cơ bản nào ?"
Invoke-Chat "Câu 3: Trong xã hội có giai cấp, triết học:"
Invoke-Chat "Cho mình câu hỏi về quy luật biện chứng"
Invoke-Chat "Chủ đề Vật chất & Ý thức"
