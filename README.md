to recover from .jpg.jpg or .jpg on .zip.(numberic go here),
  use ``for f in *.zip.*.jpg*; do mv -- "$f" "$(awk -v f="$f" 'BEGIN{sub(/\.jpg(\.jpg)?$/, "", f); print f}')"; done`` to recover your files.


to turn them into image file backed extensions so it uploads as 5mb slices of .zip, 
   use `` for f in (zipfilename go here).zip.*; do cp -- "$f" "$(awk -v f="$f" 'BEGIN{print f ".jpg"}')" && rm "$f"; done``
