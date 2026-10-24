# Mombe farm day: capture protocol

This is the capture protocol for a future field session; the README results come from public datasets and synthetic takes.

Goal: 30 to 50 cattle, each with noseprints from two sessions, one walk-around video with the board, and a measured weight. This day calibrates weight on local breeds and gives identity its first Zimbabwean test set.

## Kit

- The scale board: `mombe_scale_board_A3.pdf`, printed at 100% on A3, glued flat to stiff card. Measure the black bar under it: it reads 100 mm. Write the measured length on the back.
- A livestock girth tape (weigh band) and a 3 m builder's tape.
- A scale, if the farm, a feedlot or the veterinary department has one. Scale weights turn the day from good to decisive.
- One phone recording at 1080p or higher, plus a second person to hold the board.
- Chalk or temporary tags to number each animal, and a notebook or spreadsheet.
- Owner consent for every animal, written down with the owner's name.

## Per animal

1. **Number it.** Chalk the number on the flank. This number is the ground truth for every image and measurement.
2. **Nose, session A.** Take 6 to 8 photos with the nose filling the frame: straight on, then slightly left and slightly right. Shade the nose from direct sun. Wipe away mucus or grass first.
3. **Walk-around video.** Record one continuous take of 30 to 45 seconds:
   - start on the nose close-up, then pull back;
   - walk slowly around the animal at 2 to 3 metres, keeping the whole body in frame;
   - the helper holds the board flat against the left shoulder, facing the camera, for the first half; then against the right shoulder for the second half;
   - the animal stands square on level ground.
4. **Tape.** Heart girth just behind the front legs, in cm. Body length from the point of the shoulder to the pin bone, in cm. Height at the withers if possible.
5. **Weight.** Scale weight in kg and the time it was taken. Note the time since the animal last drank or grazed.
6. **Record.** Breed (Mashona, Nguni, Tuli, Brahman cross, other), sex, approximate age, body condition score from 1 to 5, owner.
7. **Nose, session B.** Later the same day, or the next morning, take 4 to 6 more nose photos. Two sessions show identity holds across time.

## File names

`<number>_noseA_<n>.jpg`, `<number>_noseB_<n>.jpg`, `<number>_walk.mp4`. One spreadsheet row per number with the measurements.

## Upload

Copy everything to one folder per animal number and share the folder. The pipeline reads the numbers, splits by animal, and keeps owner names out of the public results.
