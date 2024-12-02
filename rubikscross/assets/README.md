
Bash command to convert all wav files of the folder to mp3:
```bash
for i in *.wav; do ffmpeg -i "$i" -vn -ar 44100 -ac 2 -b:a 192k "${i%.*}.mp3"; done
```
Renaming all *.mp3 files in *.mp3.py:
```bash
for file in *.mp3; do mv "$file" "${file}.py"; done
```