"""Export the active JSON model into a C header."""

import json
from pathlib import Path


MODEL_FILE = "models/model.json"
HEADER_FILE = "include/voice_model.h"


def _sanitize_identifier(name):
    out = []
    for ch in name.lower():
        if ch.isalnum():
            out.append(ch)
        else:
            out.append("_")
    ident = "".join(out).strip("_")
    return ident or "cmd"


def _format_float(value):
    text = f"{float(value):.8g}"
    if "." not in text and "e" not in text.lower():
        text += ".0"
    return f"{text}f"


def _format_vector(values):
    return "{" + ", ".join(_format_float(value) for value in values) + "}"


def _format_stats(stats_by_feature, feature_order):
    values = []
    for feature in feature_order:
        stats = stats_by_feature.get(feature, {})
        values.append(
            "{"
            + _format_float(stats.get("mean", 0.0))
            + ", "
            + _format_float(stats.get("std", 0.0))
            + "}"
        )
    return "{" + ", ".join(values) + "}"


def export_voice_model_header(model_path=MODEL_FILE, header_path=HEADER_FILE):
    with open(model_path, "r") as f:
        model = json.load(f)

    meta = model.get("meta", {})
    feature_order = meta.get("feature_order", [])
    feature_weights = meta.get("feature_weights", {})
    feature_floors = meta.get("feature_floors", {})
    commands = model.get("commands", {})
    command_names = list(commands.keys())

    feature_count = len(feature_order)
    command_count = len(command_names)

    lines = [
        "#ifndef VOICE_MODEL_H",
        "#define VOICE_MODEL_H",
        "",
        f"/* Generated from {Path(model_path).as_posix()}. Keep feature extraction in sync with features.py. */",
        f"#define VOICE_MODEL_FEATURE_COUNT {feature_count}",
        f"#define VOICE_MODEL_COMMAND_COUNT {command_count}",
        f"#define VOICE_MODEL_K_NEIGHBORS {int(meta.get('k_neighbors', 3))}",
        "#define VOICE_MODEL_UNKNOWN_THRESHOLD "
        + _format_float(meta.get("unknown_threshold", 0.0)),
        "#define VOICE_MODEL_MIN_MARGIN " + _format_float(meta.get("min_margin", 0.0)),
        "",
        "typedef struct {",
        "    float values[VOICE_MODEL_FEATURE_COUNT];",
        "} VoiceFeatureVector;",
        "",
        "typedef struct {",
        "    float mean;",
        "    float std;",
        "} VoiceFeatureStat;",
        "",
        "typedef struct {",
        "    const char* command;",
        "    VoiceFeatureVector centroid;",
        "    const VoiceFeatureVector* samples;",
        "    unsigned int sample_count;",
        "    VoiceFeatureStat command_stats[VOICE_MODEL_FEATURE_COUNT];",
        "} VoiceCommandModel;",
        "",
        "static const char* g_voice_model_feature_order[VOICE_MODEL_FEATURE_COUNT] = {",
    ]

    for feature in feature_order:
        lines.append(f'    "{feature}",')
    lines.extend(
        [
            "};",
            "",
        "static const float g_voice_model_feature_weights[VOICE_MODEL_FEATURE_COUNT] = {",
    ]
    )
    for feature in feature_order:
        lines.append(f"    {_format_float(feature_weights.get(feature, 1.0))},")
    lines.extend(
        [
            "};",
            "",
            "static const float g_voice_model_feature_floors[VOICE_MODEL_FEATURE_COUNT] = {",
        ]
    )
    for feature in feature_order:
        lines.append(f"    {_format_float(feature_floors.get(feature, 0.0))},")
    lines.extend(
        [
            "};",
            "",
            "static const VoiceFeatureStat g_voice_model_feature_stats[VOICE_MODEL_FEATURE_COUNT] = "
            + _format_stats(model.get("feature_stats", {}), feature_order)
            + ";",
            "",
        ]
    )

    table_entries = []
    used_names = set()
    for command in command_names:
        ident = _sanitize_identifier(command)
        if ident in used_names:
            ident = f"{ident}_{len(used_names)}"
        used_names.add(ident)

        array_name = f"g_voice_model_samples_{ident}"
        table_entries.append((command, ident, array_name))

        lines.append(f"static const VoiceFeatureVector {array_name}[] = {{")
        for sample in commands[command].get("samples", []):
            lines.append(f"    {{{_format_vector(sample)}}},")
        lines.extend(["};", ""])

    lines.append("static const VoiceCommandModel g_voice_model_commands[VOICE_MODEL_COMMAND_COUNT] = {")
    command_stats = model.get("command_stats", {})
    for command, ident, array_name in table_entries:
        info = commands[command]
        centroid = _format_vector(info.get("centroid", [0.0] * feature_count))
        stats = _format_stats(command_stats.get(command, {}), feature_order)
        lines.append("    {")
        lines.append(f'        "{command}",')
        lines.append(f"        {{{centroid}}},")
        lines.append(f"        {array_name},")
        lines.append(
            f"        (unsigned int)(sizeof({array_name}) / sizeof({array_name}[0])),"
        )
        lines.append(f"        {stats},")
        lines.append("    },")
    lines.extend(["};", "", "#endif", ""])

    out_path = Path(header_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


if __name__ == "__main__":
    path = export_voice_model_header()
    print(f"Generated {path}")
