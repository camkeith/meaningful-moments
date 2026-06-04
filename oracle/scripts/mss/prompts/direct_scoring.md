---
name: direct_scoring
version: "3.1"
description: Single-query segment importance scoring for general video action recognition. Outputs per-segment importance scores and minimum sufficient set without iterative removal.
variables:
  - action_label: The action class label
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
- You see body movements, objects, or environments consistent with the described action
- The overall trajectory of events matches (even if some frames are blurry or ambiguous)
- You can piece together from multiple frames that this action is happening or has happened

Answer NO only if:
- The video clearly shows a DIFFERENT action than the label
- The relevant objects, actors, or motions are absent across the entire video
- The video is entirely unrelated content

Answer SKIP only if the video is corrupted, black, or completely unreadable.

If NO or SKIP, output the JSON below and stop.

STEP 2 — SEGMENT IMPORTANCE ANNOTATION:
For each {segment_duration}s segment (1 to {n_segments}), decide: "If I removed this segment, would the action become harder to recognize?"

PHASE ANALYSIS — score each segment by what it contributes:
  a) EXECUTION: The defining motion of the action (the swing, the step, the strum, the throw, the lift) — usually the highest-value content.
  b) DISAMBIGUATION: Frames that distinguish this action from a near-neighbor. If multiple actions involve similar setup or environment, the segments that uniquely identify THIS action are critical.
  c) CONTACT / INITIATION: The moment force is applied, an object is grasped, or the action begins.
  d) RESULT / COMPLETION: End state that confirms the action occurred.
  e) CONTINUATION / TRAJECTORY: Sustained motion showing direction, speed, or manner — important for repetitive or continuous actions.
  f) SETUP / APPROACH: The actor positions for the action; useful but often redundant.

Not every action exhibits every phase. Continuous actions (e.g. sustained activities) may be dominated by EXECUTION/CONTINUATION; discrete actions may have a clear CONTACT → EXECUTION → RESULT arc.

POSITION INDEPENDENCE: Score by visual content, not segment position. Early segments are not inherently more important. Late segments showing results, completion, or disambiguation can be just as critical.

ACTOR + MOTION OVER SCENE: Segments where the action is VISIBLY being performed by an actor outrank segments that show only the setting, equipment, or aftermath without the action happening.

MINIMUM SEGMENT GUIDANCE (rough, action-dependent):
- Brief discrete events (single contact, single throw, single step): 2-3 segments
- Two-phase actions (approach → execute, execute → result): 3-4 segments
- Continuous / repetitive activities (running, swimming, dancing, playing an instrument): 3-5 segments capturing characteristic motion across the clip
- Cause-and-effect chains (action causes a downstream visible effect): 4-5 segments
- Multi-stage activities with distinct phases: 4-6 segments

Important segments may appear anywhere — beginning, middle, or end.

Err toward INCLUDING a segment if you're unsure — it's worse to drop a segment that matters than to keep one that doesn't.

SCORING EACH SEGMENT (0-100):
- 80-100: Removing this segment would make the action unrecognizable or ambiguous
- 50-79: This segment adds meaningful evidence (shows a distinct phase or disambiguates)
- 20-49: Mildly helpful but redundant with adjacent segments
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
