---
name: direct_scoring_diving48
version: "1.0"
description: Single-query segment importance scoring for Diving-48 videos. Encodes FINA-rule diving phases (approach/takeoff/flight/entry) and the 4-attribute label structure (takeoff group × somersaults × twists × body position).
variables:
  - action_label: Diving-48 action label (e.g., 'forward 3.5 somersaults in pike position')
  - n_segments: Total number of segments in the video
  - duration: Video duration in seconds
  - segment_duration: Length of each segment in seconds
---
You are annotating which temporal segments of a video are needed to recognize a specific competitive dive.

ACTION: {action_label}
VIDEO DURATION: {duration}s ({n_segments} segments of {segment_duration}s each, indexed 1 to {n_segments})

This is a competitive dive, scored under FINA rules. The action label encodes a
4-attribute classification:

- Takeoff group: forward, back, reverse, inward, or armstand. The takeoff group
  determines the diver's facing direction and the relationship between body
  orientation and rotation direction.
- Number of somersaults: in half-rotation increments (1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5).
- Number of twists: in half-rotation increments (0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5).
- Body position during flight: tuck (knees to chest), pike (legs straight,
  hip fold), straight/layout (fully extended), or free (combination position).

A standard dive proceeds in four phases (FINA-defined judging components):
  APPROACH: setup at the end of the board (running for forward springboard,
            standing for back/inward, armstand for platform armstand dives).
  TAKEOFF:  push off the board or platform. Sets the angular momentum and
            initial rotation direction; brief but discriminative for the takeoff
            group.
  FLIGHT:   somersault and twist rotations executed in the announced body
            position. This phase carries most of the class-discriminative
            information: the number of somersaults, the number of twists, and
            the body position are all observed here.
  ENTRY:    head-first or feet-first water entry. Confirms the announced body
            position transitioning to a vertical line for scoring.

Discriminating between Diving-48 classes typically requires counting somersaults
and twists during FLIGHT, identifying the body position during FLIGHT, and
identifying the takeoff group from the diver's orientation during TAKEOFF.
ENTRY is also important for confirming body position. APPROACH is largely
shared across many classes and rarely class-discriminative.

STEP 1 — ACTION PRESENCE CHECK:
Watch the full video. Does it plausibly depict "{action_label}"?

Be INCLUSIVE. Answer YES if:
- You can see a diver executing a dive with takeoff group, rotation count, body position,
  and twist count broadly consistent with the label
- Motion blur during fast rotation is normal — count rotations from peak-to-peak
  body orientation changes rather than expecting crisp frames
- Some attributes may be hard to verify exactly (e.g., 4 vs 4.5 somersaults can be
  visually similar) — accept reasonable consistency, not exact pixel-level proof

Answer NO only if:
- The takeoff group is wrong (e.g., a forward takeoff for a "back" label)
- The body position is clearly wrong (e.g., visibly straight/layout for a "tuck" label,
  or vice-versa) throughout FLIGHT
- The dive is missing entirely (no diver visible, water-only footage, or unrelated content)
- The somersault count is off by ≥ 1 full somersault from the label (e.g., a single
  somersault for a "3.5 somersault" label)

Answer SKIP only if the video is corrupted, black, or completely unreadable.

If NO or SKIP, output the JSON below and stop.

STEP 2 — SEGMENT IMPORTANCE ANNOTATION:
For each {segment_duration}s segment (1 to {n_segments}), decide: "If I removed this segment, would
the dive become harder to recognize as '{action_label}' specifically?"

DIVING PHASE → SEGMENT PHASE MAPPING (use these phase labels in the output JSON):
  APPROACH segments → "setup"      (low priority unless armstand or unusual stance)
  TAKEOFF segments  → "contact"    (HIGH priority — encodes takeoff group)
  FLIGHT segments   → "execution"  (HIGHEST priority — encodes somersaults, twists, body position)
                       or "continuation" if the rotation is sustained across many segments
  ENTRY segments    → "result"     (HIGH priority — confirms body position and rotation completion)

Use "disambiguation" for segments that uniquely separate this dive from a
near-neighbor class (e.g., the moment that distinguishes 3 vs 3.5 somersaults
by the entry orientation, or pike vs tuck by visible knee position mid-flight).

PRIORITY GUIDANCE FOR DIVING:
- FLIGHT segments are almost always HIGH (60-100). The number of somersaults and
  twists, and the body position, can only be verified during flight.
- TAKEOFF is brief (often 1 segment) but HIGH (70-95). It anchors the takeoff group.
  Drop it only if the takeoff group is unambiguously visible elsewhere.
- ENTRY segments are MEDIUM-HIGH (50-80). The entry confirms body-position transition
  to vertical and is essential for verifying the announced body position.
- APPROACH segments are usually LOW (10-30) for forward/back/reverse/inward dives where
  the approach is generic. EXCEPTION: armstand dives — the armstand setup IS the
  takeoff group signature, so APPROACH for armstand is HIGH.
- Empty pre-dive footage (diver not yet on the platform, just water, post-splash water
  rings) is FILLER (0-15).

POSITION INDEPENDENCE: Score by visual content, not segment position. The dive may
start late or end early in the clip — find the actual TAKEOFF → FLIGHT → ENTRY span
and score those highest regardless of where they fall in the timeline.

MINIMUM SEGMENT GUIDANCE FOR DIVING:
- Most dives need 3-5 keep segments: 1 takeoff + 2-3 flight + 1 entry
- High-rotation dives (3.5, 4, 4.5 somersaults) may need 4-6 flight segments to
  count rotations confidently
- Multi-twist dives (1.5, 2, 2.5+ twists) similarly benefit from extra flight
  coverage to count twist axes
- Armstand dives need APPROACH segments (the armstand) plus TAKEOFF + FLIGHT + ENTRY,
  so 4-6 keep segments total

Err toward INCLUDING a segment if it shows any portion of the rotating body in flight
or the entry — it's worse to drop a flight segment that disambiguates somersault count
than to keep one that's slightly redundant.

SCORING EACH SEGMENT (0-100):
- 80-100: Removing this segment would make the dive's class unrecognizable or ambiguous
  (e.g., the only flight segment showing a particular rotation, the takeoff that
  identifies the takeoff group, the entry that confirms body position).
- 50-79: Meaningful evidence — a flight segment among several, the entry, or a
  disambiguating frame between two near-neighbor classes.
- 20-49: Mildly helpful but redundant with adjacent flight segments showing the
  same rotation phase.
- 0-19: Pre-dive footage, post-splash water-only frames, or other filler with no
  diver in flight.

OUTPUT (JSON only, no markdown, no other text):
{{
  "decision": "YES" | "NO" | "SKIP",
  "confidence": <0.0-1.0>,
  "action_summary": "<what dive you see, including takeoff group, rotation count, twist count, body position; max 30 words>",
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
  "rationale": "<why these segments are needed together to recognize this specific dive class — call out which segments verify takeoff group, somersault count, twist count, and body position; 2-3 sentences>"
}}

You MUST include ALL {n_segments} segments in the "segments" array — one entry per segment, no omissions.
