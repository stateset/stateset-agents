#!/bin/bash
# Pod-side guard: the running launcher decodes `tail -c 300 phase1.log` as UTF-8 and
# crashes when the byte cut splits a multi-byte character. Keep an ASCII-only copy at
# the path it reads; the job keeps writing to the renamed raw log (same open fd).
cd /workspace || exit 1
if [ ! -f phase1.raw.log ]; then
  mv phase1.log phase1.raw.log
fi
while true; do
  python3 -c "d=open('/workspace/phase1.raw.log','rb').read(); open('/workspace/phase1.log.tmp','w').write(d.decode('ascii','ignore'))" \
    && mv -f /workspace/phase1.log.tmp /workspace/phase1.log
  sleep 5
done
