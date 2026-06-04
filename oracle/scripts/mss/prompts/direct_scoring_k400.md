---
name: direct_scoring_k400
version: "1.1"
description: Single-query segment importance scoring for Kinetics-400 videos. Outputs per-segment importance scores and minimum sufficient set without iterative removal.
variables:
  - action_label: The K400 action class (e.g., 'playing basketball', 'swimming')
  - n_segments: Total number of segments in the video
  - duration: Video duration in seconds
  - segment_duration: Length of each segment in seconds
---
You are annotating which temporal segments of a video are needed to recognize a specific action.

ACTION: {action_label}
VIDEO DURATION: {duration}s ({n_segments} segments of {segment_duration}s each, indexed 1 to {n_segments})

STEP 1 — ACTION PRESENCE CHECK:
Watch the full video. Does it plausibly depict "{action_label}"?

Be INCLUSIVE. Answer YES if:
- You see an actor and motion consistent with "{action_label}"
- The overall trajectory of events matches (even if some frames are blurry, occluded, or partially out of frame)
- You can piece together from multiple frames that this action is happening or has happened

Answer NO only if:
- The video clearly shows a DIFFERENT action than the label
- You see only a scene, setting, or equipment with no actor performing the action
- The relevant actor or motion is absent throughout

Answer SKIP only if the video is corrupted, black, or completely unreadable.

If NO or SKIP, output the JSON below and stop.

STEP 2 — SEGMENT IMPORTANCE ANNOTATION:
For each {segment_duration}s segment (1 to {n_segments}), decide: "If I removed this segment, would the action become harder to recognize?"

IMPORTANT: Many K400 classes are continuous or repetitive. Think about which segments capture the characteristic motion vs. which are repetition with no new information.

PHASE ANALYSIS — score each segment by what it contributes:
  a) EXECUTION: The defining motion of the action — dribbling/shooting (basketball), strumming/fingering (guitar), kicking (soccer), strokes (swimming), chopping/stirring (cooking), step patterns (dance). USUALLY the highest-value segments.
  b) CONTINUATION / TRAJECTORY: Sustained or repeated motion showing manner, rhythm, or speed. For continuous activities (running, swimming, playing an instrument), MULTIPLE segments of repeated execution carry signal — they confirm this is the activity, not a single momentary motion.
  c) DISAMBIGUATION: Frames that distinguish "{action_label}" from a near-neighbor class. Many K400 classes share scenes/equipment (e.g. tennis vs badminton, jogging vs running, playing guitar vs playing bass) — segments that uniquely identify THIS class are critical.
  d) CONTACT / KEY EVENT: The moment of impact, release, or peak action (a serve, a swing, a jump, a release).
  e) RESULT: A visible outcome that confirms the action occurred (ball entering hoop, food landing on plate).
  f) SETUP / APPROACH: Actor positions or prepares; usually lower-value, but can matter for rapid actions.

POSITION INDEPENDENCE: Score by visual content, not position. Early segments are not inherently more important. Late segments showing the activity in progress can be just as critical.

ACTOR + MOTION OVER SCENE: Segments where the action is VISIBLY being performed by an actor outrank segments that show only the venue, equipment, or aftermath. A segment of only a basketball court (no players) or only a kitchen (no cooking) deserves a low score even though the scene is on-class.

MINIMUM SEGMENT GUIDANCE (rough, action-dependent):
- Continuous activities sustained throughout the clip (swimming, running, playing an instrument, dancing): 3-5 segments capturing execution at different points in the clip
- Repetitive sports actions (dribbling, lifting weights, jumping rope): 3-5 segments showing the repeated cycle
- Discrete sports moments (a single throw, kick, dunk, dive): 3-4 segments around the contact / release
- Multi-step daily activities (cooking a dish, assembling something): 4-6 segments covering distinct sub-actions
- Brief expressive actions (shaking hands, hugging, sneezing): 2-3 segments around the contact / peak

Important segments may appear anywhere — beginning, middle, or end.

Err toward INCLUDING a segment if you're unsure — it's worse to drop a segment that matters than to keep one that doesn't.

SCORING EACH SEGMENT (0-100):
- 80-100: Removing this segment would make the action unrecognizable or ambiguous
- 50-79: Meaningful evidence — shows a distinct phase, repetition cycle, or disambiguates from a near class
- 20-49: Mildly helpful but redundant with adjacent execution segments
- 0-19: Filler, static pause, scene-only with no action visible, or no relevant content

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

You MUST include ALL {n_segments} segments in the "segments" array — one entry per segment, no omissions.
