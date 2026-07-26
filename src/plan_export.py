import csv
import io


def build_plan_csv(plan_rows):
    output = io.StringIO()
    if plan_rows:
        writer = csv.DictWriter(output, fieldnames=list(plan_rows[0]))
        writer.writeheader()
        writer.writerows(plan_rows)
    return output.getvalue().encode("utf-8")


def _pdf_escape(value):
    return str(value).replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def build_plan_pdf(primary_plan, validation, plan_rows, context):
    lines = [
        "IPL Strategy Engine - Bowling Plan",
        f"Context: {context}",
        f"Plan A: {' -> '.join(primary_plan.get('sequence', []))}",
        (
            f"Expected runs: {primary_plan.get('expected_runs', 0):.1f} | "
            f"Expected wickets: {primary_plan.get('expected_wickets', 0):.2f}"
        ),
    ]
    if validation:
        lines.append(
            f"Monte Carlo simulations: {validation.get('simulations', 0)} | "
            f"Preferred plan: {validation.get('preferred_plan', 'A')}"
        )
    lines.append("")
    for row in plan_rows:
        lines.append(
            f"Over {row['Planned Over']}: {row['Bowler']} | "
            f"runs {row['Expected Runs']:.1f} | wickets {row['Expected Wickets']:.2f} | "
            f"boundary {row['Boundary in Over %']:.1f}%"
        )

    text_commands = ["BT", "/F1 11 Tf", "48 790 Td", "14 TL"]
    for index, line in enumerate(lines[:48]):
        if index:
            text_commands.append("T*")
        text_commands.append(f"({_pdf_escape(line)}) Tj")
    text_commands.append("ET")
    stream = "\n".join(text_commands).encode("latin-1", errors="replace")

    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 842] "
            b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>"
        ),
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for object_number, pdf_object in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{object_number} 0 obj\n".encode())
        pdf.extend(pdf_object)
        pdf.extend(b"\nendobj\n")
    xref_offset = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode())
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode())
    pdf.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF"
        ).encode()
    )
    return bytes(pdf)
