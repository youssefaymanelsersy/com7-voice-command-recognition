import json
from pathlib import Path

from core.features import FRAME_COUNT


DATASET_FILE = "data/samples.json"
HEADER_FILE = "include/samplesZteAndZcr.h"


def _sanitize_identifier(name: str) -> str:
    out = []
    for ch in name.lower():
        if ch.isalnum():
            out.append(ch)
        else:
            out.append("_")
    ident = "".join(out).strip("_")
    return ident or "cmd"


def _format_float(value: float) -> str:
    text = f"{float(value):.8g}"
    if "." not in text and "e" not in text.lower():
        text += ".0"
    return f"{text}f"


def export_samples_zte_and_zcr_header(
    dataset_path: str = DATASET_FILE,
    header_path: str = HEADER_FILE,
) -> Path:
    """Export zcr/energy pairs from data/samples.json into a C header file."""
    with open(dataset_path, "r") as f:
        sample_database = json.load(f)

    commands = sorted(sample_database.keys())
    lines = []
    lines.append("#ifndef SAMPLES_ZTE_AND_ZCR_H")
    lines.append("#define SAMPLES_ZTE_AND_ZCR_H")
    lines.append("")
    lines.append("#include <stdint.h>")
    lines.append("")
    lines.append("typedef struct {")
    lines.append("    int32_t zcr;")
    lines.append("    int64_t energy;")
    lines.append(f"    int32_t zcr_features[{FRAME_COUNT}];")
    lines.append(f"    int64_t ste_features[{FRAME_COUNT}];")
    lines.append("} SampleZteAndZcr;")
    lines.append("")
    lines.append("typedef struct {")
    lines.append("    const char* command;")
    lines.append("    const SampleZteAndZcr* samples;")
    lines.append("    unsigned int count;")
    lines.append("} CommandSamplesZteAndZcr;")
    lines.append("")

    table_entries = []
    for command in commands:
        samples = sample_database.get(command, [])
        ident = _sanitize_identifier(command)
        array_name = f"g_samples_{ident}"

        lines.append(f"static const SampleZteAndZcr {array_name}[] = {{")
        for sample in samples:
            zcr = int(sample.get("zcr", 0))
            energy = int(sample.get("energy", 0))
            zcr_features = ", ".join(str(int(value)) for value in sample.get("zcr_features", [0] * FRAME_COUNT)[:FRAME_COUNT])
            ste_features = ", ".join(str(int(value)) for value in sample.get("ste_features", [0] * FRAME_COUNT)[:FRAME_COUNT])
            lines.append(
                "    {"
                + f"{zcr}, {energy}, "
                + "{"
                + zcr_features
                + "}, {"
                + ste_features
                + "}"
                + "},"
            )
        lines.append("};")
        lines.append("")

        table_entries.append((command, array_name))

    lines.append("static const CommandSamplesZteAndZcr g_command_samples_zte_and_zcr[] = {")
    for command, array_name in table_entries:
        lines.append(
            f"    {{\"{command}\", {array_name}, (unsigned int)(sizeof({array_name}) / sizeof({array_name}[0]))}},"
        )
    lines.append("};")
    lines.append("")
    lines.append(
        "static const unsigned int g_command_samples_zte_and_zcr_count = (unsigned int)(sizeof(g_command_samples_zte_and_zcr) / sizeof(g_command_samples_zte_and_zcr[0]));"
    )
    lines.append("")
    lines.append("#endif")
    lines.append("")

    out_path = Path(header_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


if __name__ == "__main__":
    path = export_samples_zte_and_zcr_header()
    print(f"Generated {path}")
