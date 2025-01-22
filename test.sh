#!/bin/bash

# Convert image to base64
BASE64_IMAGE=$(base64 -w 0 demo.jpg)

# Create temporary file for JSON payload
TEMP_FILE=$(mktemp)

# Write JSON payload to temporary file
cat > "$TEMP_FILE" << EOF
{
  "base64Image": "$BASE64_IMAGE",
  "prompt": "<image>Analyze this document image and provide a complete, accurate transcription of all text.\nThe image is the firt page and subsequent pages will be transcribed and appended to your transcript of this first page without modification.\nPay special attention to:\nMaintain line breaks.\nTranscribe any handwritten texts along with typed text.\nIgnore any signatures, or stamps.\nIgnore any text that is unclear, blurry, or partially obscured.\nRespond with only the transcription and nothing else. Do not comment on your response.\nDo not add your own labels, annotations or any other text."
}
EOF

# Send request using the temporary file
curl -X POST \
  -H "Content-Type: application/json" \
  --data "@$TEMP_FILE" \
  http://localhost:3303/api

echo "sent"

# Clean up temporary file
rm "$TEMP_FILE"
