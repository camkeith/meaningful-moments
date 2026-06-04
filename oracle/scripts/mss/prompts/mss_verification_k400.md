---
name: mss_verification_k400
version: "3.0"
description: Verifies whether a video contains sufficient visual evidence for a Kinetics-400 action class. Requires visible action execution, not just scene recognition.
variables:
  - action_label: The K400 action class (e.g., 'playing basketball', 'swimming')
---

You are verifying whether a video shows a specific action from the Kinetics-400 dataset.

ACTION CLASS: {action_label}

ABOUT KINETICS-400:
This dataset contains 400 human action classes covering:
- Sports (playing basketball, swimming, skateboarding)
- Daily activities (cooking, cleaning, eating)
- Interactions (shaking hands, hugging, fighting)
- Instrument playing (playing guitar, playing piano)
- Dance and movement (dancing, yoga, stretching)

Actions are typically 10 seconds long and focus on the MAIN activity being performed.

TASK: Determine if this video shows someone performing "{action_label}".

CRITICAL REQUIREMENT — VISIBLE ACTION:
You must SEE the action happening in the video frames. Recognizing a scene or setting where the action COULD happen is NOT sufficient. The action itself must be visible.

Ask yourself: "Can I see someone DOING '{action_label}', or do I just see a setting where it might happen?"
- A basketball court with no players moving is NOT "playing basketball"
- A kitchen with no cooking activity is NOT "cooking"
- A guitar visible but not being played is NOT "playing guitar"

KEY JUDGMENT CRITERIA:
1. MAIN ACTION identification:
   - What is the PRIMARY activity VISIBLE in the video?
   - Ignore brief/incidental actions, focus on the dominant activity
   - The person performing the action must be visible AND actively doing it

2. ACTION EXECUTION (not just scene):
   - The defining motion/activity of "{action_label}" must be VISIBLE across frames
   - Seeing just a setting, equipment, or result state is NOT sufficient
   - "playing basketball" = you must see dribbling, shooting, passing — not just a court
   - "cooking" = you must see food preparation happening — not just a kitchen

3. REASONABLE MATCHING:
   - Accept natural variations of the action class
   - "playing basketball" = dribbling, shooting, passing a basketball
   - "cooking" = chopping, stirring, frying in kitchen context

4. MASKED OR PARTIAL VIDEO:
   - If parts of the video are masked, blurred, or removed, judge from the VISIBLE portions
   - But the action itself must still be recognizable in the remaining frames
   - If the remaining frames show only the setting/scene without the action happening, choose NO

CONFIDENCE SCALE (only applies when decision is YES):
- 0.8-1.0: Action is clearly and unambiguously visible — you can see it happening
- 0.5-0.8: Action is recognizable but some details are obscured or uncertain
- 0.3-0.5: Action appears to be happening but evidence is limited or partially visible

DECISION:
- YES: The action "{action_label}" is VISIBLY being performed — you can see the motion/activity happening in the video frames.
- NO: The action is NOT visible — you see only a scene/setting, only the result, or a DIFFERENT action entirely.
- SKIP: Video is corrupted, completely black, entirely unrelated content, or so degraded that NO action of ANY kind can be identified. SKIP is a last resort.

OUTPUT (JSON only, no other text):
{{
  "decision": "YES" | "NO" | "SKIP",
  "confidence": <0.0-1.0>,
  "evidence": "<what action you observed happening, max 20 words>",
  "rationale": "<explain: can you see the action being performed? what motion/activity is visible? 2-3 sentences>"
}}
