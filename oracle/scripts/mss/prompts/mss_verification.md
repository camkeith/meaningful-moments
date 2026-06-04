---
name: mss_identification_ssv2
version: "4.0"
description: Identifies the minimum set of segments needed to recognize a Something-Something V2 action in a video.
variables:
  - action_template: The SSv2 action label with specific objects
  - placeholders: The objects involved (comma-separated)
  - n_segments: Total number of segments in the video
  - duration: Video duration in seconds
  - segment_duration: Length of each segment in seconds
---
```
```
You are annotating which temporal segments of a video are needed to recognize a specific action.

ACTION: {action_template}
OBJECT(S): {placeholders}
VIDEO DURATION: {duration}s ({n_segments} segments of {segment_duration}s each, indexed 1 to {n_segments})

STEP 1 — ACTION PRESENCE CHECK:
Watch the full video. Does it plausibly depict "{action_template}"?

Be INCLUSIVE. Answer YES if:
- You see objects and body movements consistent with the described action
- The overall trajectory of events matches (even if some frames are blurry or ambiguous)
- You can piece together from multiple frames that this action is happening or has happened

Answer NO only if:
- The video clearly shows a DIFFERENT action (e.g., label says "pushing" but you see "pulling")
- The relevant objects are completely absent from the entire video
- The video is entirely unrelated content

Answer SKIP only if the video is corrupted, black, or completely unreadable.

If NO or SKIP, output the JSON below and stop.

STEP 2 — SEGMENT IMPORTANCE ANNOTATION:
For each {segment_duration}s segment (1 to {n_segments}), decide: "If I removed this segment, would the action become harder to recognize?"

IMPORTANT: Most SSv2 actions require MULTIPLE temporal phases to distinguish from similar actions. Think about what makes THIS action different from related ones:

PHASE ANALYSIS — consider which segments capture:
  a) SETUP/APPROACH: Hand reaches toward object, positions for action
  b) CONTACT/INITIATION: Moment of grip, touch, or force application
  c) EXECUTION: The core motion (rotation, translation, deformation, release)
  d) CONTINUATION/TRAJECTORY: Sustained motion showing direction, speed, manner
  e) RESULT/COMPLETION: End state confirming the action occurred (or did NOT occur for "pretending")
  f) DISAMBIGUATION: Any segment that distinguishes this action from a similar one
     - "pushing left to right" vs "pushing right to left" → need enough frames to confirm direction
     - "throwing" vs "dropping" → need the release moment
     - "pretending to X" vs actually doing X → need to see the non-completion

MINIMUM SEGMENT GUIDANCE:
- Simple static displays ("showing X behind Y"): 1-2 segments
- Single discrete motions ("picking X up", "turning X over"): 2-3 segments
- Two-phase actions ("putting X into Y", "dropping X"): 3-4 segments
- Sustained/continuous actions ("spinning", "rolling"): 3-5 segments
- Cause-and-effect chains ("pushing X so it falls"): 4-5 segments
- Intentional non-completion ("pretending to X"): 4-6 segments (need context showing the absence)

Err toward INCLUDING a segment if you're unsure — it's worse to drop a segment that matters than to keep one that doesn't.

SCORING EACH SEGMENT (0-100):
- 80-100: Removing this segment would make the action unrecognizable or ambiguous
- 50-79: This segment adds meaningful evidence (shows a distinct phase or disambiguates)
- 20-49: Mildly helpful but redundant with adjacent segments
- 0-19: Filler, static pause, or no relevant content

OUTPUT (JSON only, no markdown, no other text):
{{
  "decision": "YES" | "NO" | "SKIP",
  "confidence": <0.0-1.0>,
  "action_summary": "<what you see happening in the video, max 30 words>",
  "segments": [
    {{
      "segment_id": <1-based index>,
      "time_range": "<start_s>-<end_s>",
      "importance": <0-100>,
      "phase": "setup" | "contact" | "execution" | "continuation" | "result" | "disambiguation",
      "reason": "<max 10 words>"
    }}
  ],
  "minimum_sufficient_set": [<list of segment_ids scoring >= 50>],
  "rationale": "<why these segments are needed together to recognize the action, 2-3 sentences>"
}}

Only include segments scoring >= 20 in the "segments" array (omit pure filler).