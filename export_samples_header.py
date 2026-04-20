import json
from pathlib import Path


DATASET_FILE = "samples.json"
HEADER_FILE = "samplesZteAndZcr.h"


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
    return f"{float(value):.8g}f"


def export_samples_zte_and_zcr_header(
    dataset_path: str = DATASET_FILE,
    header_path: str = HEADER_FILE,
) -> Path:
    """Export zcr/energy pairs from samples.json into a C header file."""
    with open(dataset_path, "r") as f:
        sample_database = json.load(f)

    commands = sorted(sample_database.keys())
    lines = []
    lines.append("#ifndef SAMPLES_ZTE_AND_ZCR_H")
    lines.append("#define SAMPLES_ZTE_AND_ZCR_H")
    lines.append("")
    lines.append("typedef struct {")
    lines.append("    float zcr;")
    lines.append("    float energy;")
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
            zcr = _format_float(sample.get("zcr", 0.0))
            energy = _format_float(sample.get("energy", 0.0))
            lines.append(f"    {{{zcr}, {energy}}},")
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
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


if __name__ == "__main__":
    path = export_samples_zte_and_zcr_header()
    print(f"Generated {path}")
