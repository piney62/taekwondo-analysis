"""Layer 5 LLM feedback generation.

Converts MovementIssueSummary objects into natural-language corrective feedback
by calling the configured LLM client.

Typical usage:
    from itf_analysis.feedback.llm_client import get_llm_client
    from itf_analysis.feedback.llm_feedback import generate_full_feedback
    from itf_analysis.segmentation.keypose_matcher import load_master_keyposes

    keyposes = load_master_keyposes("master_data/chon_ji/keyposes_angles.json")
    client   = get_llm_client()
    feedback = generate_full_feedback(summaries, client, keyposes)
    # feedback[1] → "Your left elbow is too bent during the high block ..."
"""

from typing import Dict, List, Optional

from itf_analysis.feedback.difference_analyzer import MovementIssueSummary
from itf_analysis.feedback.llm_client import LLMClient

# ---------------------------------------------------------------------------
# Human-readable angle labels
# ---------------------------------------------------------------------------

_JOINT_LABEL: Dict[str, str] = {
    "right_knee":          "right knee",
    "left_knee":           "left knee",
    "right_elbow":         "right elbow",
    "left_elbow":          "left elbow",
    "right_hip":           "right hip",
    "left_hip":            "left hip",
    "shoulder_line_angle": "shoulder rotation",
    "hip_line_angle":      "hip rotation",
}

# Directional descriptions tuned per joint type.
_DIRECTION_LABEL: Dict[str, Dict[str, str]] = {
    "right_elbow":         {"too_small": "too bent",             "too_large": "too straight"},
    "left_elbow":          {"too_small": "too bent",             "too_large": "too straight"},
    "right_knee":          {"too_small": "too deeply bent",      "too_large": "not bent enough"},
    "left_knee":           {"too_small": "too deeply bent",      "too_large": "not bent enough"},
    "right_hip":           {"too_small": "too closed",           "too_large": "too open"},
    "left_hip":            {"too_small": "too closed",           "too_large": "too open"},
    "shoulder_line_angle": {"too_small": "not rotated enough",   "too_large": "over-rotated"},
    "hip_line_angle":      {"too_small": "not rotated enough",   "too_large": "over-rotated"},
}

def _build_system_prompt(poomsae_name: str) -> str:
    return (
        f"You are an ITF Taekwondo instructor reviewing a student's {poomsae_name}.\n"
        "You receive motion-analysis data for one movement and write concise corrective feedback.\n"
        "\n"
        "Rules:\n"
        '- Mention only what the student needs to fix, never what they did correctly.\n'
        '- Name the specific body part in plain English (e.g. "right elbow", "shoulder rotation").\n'
        "- Do not include any numeric angles or percentages in your response.\n"
        "- Keep feedback to 2-3 sentences.\n"
        '- Write directly to the student using "your" (e.g. "Your right elbow is ...").\n'
        "- Be specific enough that the student knows exactly what to adjust.\n"
    )

# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _build_user_prompt(
    summary: MovementIssueSummary,
    movement_name: str,
    itf_name: str,
) -> str:
    """Format a MovementIssueSummary as a structured user prompt."""
    name_line = f"{itf_name}" if itf_name else f"Movement {summary.movement_number}"
    if movement_name:
        name_line = f"{movement_name} - {name_line}"

    lines = [
        f"Movement {summary.movement_number}: {name_line}",
        f"Frames analysed: {summary.frame_count}  |  Mean error: {summary.mean_rms:.1f} degrees",
        "",
        "Issues to correct (worst first):",
    ]
    for i, issue in enumerate(summary.issues, 1):
        label = _JOINT_LABEL.get(issue.joint, issue.joint.replace("_", " "))
        direction = (
            _DIRECTION_LABEL.get(issue.joint, {}).get(issue.direction, issue.direction)
        )
        lines.append(f"  {i}. {label}: {direction}  (severity {issue.severity:.1f}×)")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_movement_feedback(
    summary: MovementIssueSummary,
    client: LLMClient,
    movement_name: str = "",
    itf_name: str = "",
    poomsae_name: str = "",
    max_tokens: int = 200,
) -> str:
    """Generate natural-language feedback for a single movement.

    Args:
        summary: MovementIssueSummary from summarize_movement_issues().
        client: Configured LLMClient instance.
        movement_name: Internal movement name, e.g. "left_walking_high_block".
            Optional — used only to enrich the prompt context.
        itf_name: Official ITF technique name, e.g. "Ap Kubi Olgul Makgi (L)".
            Optional — used only to enrich the prompt context.
        poomsae_name: Human-readable poomsae name, e.g. "Chon-Ji poomsae".
            Used in the system prompt so the LLM has technique context.
        max_tokens: Maximum tokens in the generated response.

    Returns:
        Natural-language feedback string. Returns a fixed "no issues" message
        when summary.issues is empty, without calling the LLM.
    """
    if not summary.issues:
        return "No significant issues detected for this movement."

    user_prompt = _build_user_prompt(summary, movement_name, itf_name)
    system_prompt = _build_system_prompt(poomsae_name or "ITF poomsae")
    return client.generate(system_prompt, user_prompt, max_tokens=max_tokens)


def generate_full_feedback(
    summaries: Dict[int, MovementIssueSummary],
    client: LLMClient,
    keyposes: Optional[List[Dict]] = None,
    poomsae_name: str = "",
    max_tokens: int = 200,
) -> Dict[int, str]:
    """Generate feedback for every movement in *summaries*.

    Args:
        summaries: Dict mapping movement_number to MovementIssueSummary.
        client: Configured LLMClient instance.
        keyposes: Optional keypose list from load_master_keyposes(). When
            provided, movement_name and itf_name are extracted per movement
            and included in the prompt for richer context.
        max_tokens: Maximum tokens per movement response.

    Returns:
        Dict mapping movement_number to feedback string.
    """
    kp_map: Dict[int, Dict] = {}
    if keyposes:
        kp_map = {kp["movement_index"]: kp for kp in keyposes}

    feedback: Dict[int, str] = {}
    for mov_num in sorted(summaries):
        kp = kp_map.get(mov_num, {})
        feedback[mov_num] = generate_movement_feedback(
            summaries[mov_num],
            client,
            movement_name=kp.get("movement_name", ""),
            itf_name=kp.get("itf_name", ""),
            poomsae_name=poomsae_name,
            max_tokens=max_tokens,
        )
    return feedback
