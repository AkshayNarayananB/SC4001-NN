# Real-sensor capture protocol (Samsung Galaxy S25 Ultra)

Goal: real low-light captures in which we control exposure time, burst length and camera motion, to validate the
simulator's sensing-vs-compute trade-off on real hardware. Only the main (wide, 1x) camera is used.

## Step 1: pilot (about 15 minutes, do this first)

We need to know which app writes a **single-frame** RAW. Samsung's Expert RAW merges several frames into one DNG,
which would hide exactly the trade-off we measure. Take the shots below in a dim room (one small lamp, or a
room lit only by a hallway), pointing at a table with a few everyday objects on it.

1. **Stock Camera, Pro mode.** Camera > Settings > Advanced picture options > Picture formats > turn on
   "RAW copies (Pro)". In Pro mode set ISO 800, white balance auto, focus auto. Phone resting on the table:
   take one shot each at shutter 1/500, 1/125, 1/30, 1/8.
2. **Expert RAW app**, same scene, same ISO and shutter speeds (1/500 and 1/30 are enough).
3. **Open Camera** (free, Play Store). Settings > Camera API: "Camera2 API". Settings > Photo settings >
   RAW: "JPEG and DNG". Then set the photo mode to "Fast burst" (8 images) with manual ISO 800 and exposure 1/500.
   Take one burst. Note whether it saves 8 DNG files.
4. Copy all DNG files (and the JPEGs) into one folder, name the folder after the method
   (`pro/`, `expertraw/`, `opencamera/`), and put them on the GPU box under
   `/mnt/Data/akshay/phone_pilot/` (or a Google Drive folder you share with yourself).

I will check each file's metadata and noise statistics and confirm which method gives true single frames and
true bursts, then fix the full protocol below.

## Step 2: full capture (after the pilot; about 3 to 4 hours, can be split over evenings)

About 100 scenes: rooms at night, corridors, a dim kitchen, a street or car park at dusk. Each scene should hold
2 to 6 objects from the 80 COCO categories (person, chair, bottle, cup, laptop, tv, car, bicycle, potted plant,
book, ...). Only photograph people who agree to it; scenes with objects only are fine.

Per scene (ISO fixed at 800):
- **Reference:** one well-exposed shot (room lights on, or phone resting and a long exposure). Used only to label
  the scene, never as a test image.
- **Static exposure ladder:** phone resting on a surface, shutter 1/500, 1/250, 1/125, 1/60, 1/30, 1/15, 1/8.
- **Static burst:** 8 frames at 1/500, and 8 frames at 1/125.
- **Moving exposure ladder:** the same 7 shutter speeds while slowly panning the phone (about one phone-width per
  second), and the same two bursts while panning.

That is about 30 files per scene. Labels: after the session I will propose the objects present in each reference
shot, and you confirm or correct them (about 1 minute per scene).

## Privacy

The photos stay on the GPU box and are not uploaded anywhere. The paper would show a few example scenes without
people unless they consent.
