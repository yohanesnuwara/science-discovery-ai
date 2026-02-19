"""
Convert markdown report to PDF with LaTeX-style equation formatting using Unicode
"""

import markdown
import re

# Read the markdown file
with open(".output/scientific_report.md", "r") as f:
    md_content = f.read()


# Replace equations with Unicode-based LaTeX-style formatting
def format_latex_style(text):
    """Replace equation patterns with Unicode-formatted versions"""

    # Binary crisis onset - create centered equation
    text = text.replace(
        "**y(c,t) = 1{crisisJST(c,t+1) = 1 AND crisisJST(c,t) = 0}**",
        """<div style="text-align: center; margin: 20px 0; font-size: 13pt; font-family: 'Times New Roman', serif;">
<strong>y<sub>c,t</sub> = 𝟙{crisisJST<sub>c,t+1</sub> = 1 ∧ crisisJST<sub>c,t</sub> = 0}</strong>
</div>""",
    )

    # Multi-year ahead - centered display
    text = text.replace(
        "**y(c,t,h) = 1{ (Σ(τ=1 to h) crisisJST(c,t+τ) ≥ 1) AND (crisisJST(c,t) = 0) }**",
        """<div style="text-align: center; margin: 20px 0; font-size: 13pt; font-family: 'Times New Roman', serif;">
<strong>y<sub>c,t</sub><sup>(h)</sup> = 𝟙{( Σ<sub>τ=1</sub><sup>h</sup> crisisJST<sub>c,t+τ</sub> ≥ 1 ) ∧ ( crisisJST<sub>c,t</sub> = 0 )}</strong>
</div>""",
    )

    # Multi-task loss
    text = text.replace(
        "**L_multi = Σ(k=1 to 4) BCE(ŷ_k, y_k)**",
        """<div style="text-align: center; margin: 20px 0; font-size: 13pt; font-family: 'Times New Roman', serif;">
<strong>ℒ<sub>multi</sub> = Σ<sub>k=1</sub><sup>4</sup> BCE(ŷ<sub>k</sub>, y<sub>k</sub>)</strong>
</div>""",
    )

    # Predicted probability
    text = text.replace(
        "**p̂(c,t) = Pr(y(c,t) = 1 | X(c,t))**",
        """<div style="text-align: center; margin: 20px 0; font-size: 13pt; font-family: 'Times New Roman', serif;">
<strong>p̂<sub>c,t</sub> = Pr( y<sub>c,t</sub> = 1 | 𝐗<sub>c,t</sub> )</strong>
</div>""",
    )

    return text


# Process the markdown
md_content = format_latex_style(md_content)

# Convert markdown to HTML
html_content = markdown.markdown(md_content, extensions=["tables", "fenced_code"])

# Add comprehensive CSS
css_content = """
@page {
    size: A4;
    margin: 2.5cm;
    @bottom-center {
        content: counter(page);
        font-size: 9pt;
    }
}

body {
    font-family: 'DejaVu Serif', 'Times New Roman', serif;
    font-size: 11pt;
    line-height: 1.6;
    color: #000;
}

h1 {
    font-size: 18pt;
    font-weight: bold;
    margin-top: 24pt;
    margin-bottom: 12pt;
    page-break-after: avoid;
}

h2 {
    font-size: 14pt;
    font-weight: bold;
    margin-top: 18pt;
    margin-bottom: 10pt;
    page-break-after: avoid;
}

h3 {
    font-size: 12pt;
    font-weight: bold;
    margin-top: 14pt;
    margin-bottom: 8pt;
    page-break-after: avoid;
}

p {
    margin-bottom: 10pt;
    text-align: justify;
}

table {
    width: 100%;
    border-collapse: collapse;
    margin: 15pt 0;
    font-size: 10pt;
}

th, td {
    border: 1px solid #000;
    padding: 6pt;
    text-align: left;
}

th {
    background-color: #f0f0f0;
    font-weight: bold;
}

ul, ol {
    margin-left: 20pt;
    margin-bottom: 10pt;
}

li {
    margin-bottom: 4pt;
}

hr {
    border: none;
    border-top: 1px solid #000;
    margin: 20pt 0;
}

strong {
    font-weight: bold;
}

em {
    font-style: italic;
}

sub, sup {
    font-size: 8pt;
}
"""

# Create full HTML document
full_html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Financial Crisis Forecasting Using Temporal Convolutional Networks</title>
</head>
<body>
{html_content}
</body>
</html>
"""

# Save HTML first
with open(".output/scientific_report_formatted.html", "w") as f:
    f.write(full_html)

print("✓ HTML with formatted equations saved: .output/scientific_report_formatted.html")

# Convert to PDF
try:
    from weasyprint import HTML, CSS
    from weasyprint.text.fonts import FontConfiguration

    font_config = FontConfiguration()
    HTML(string=full_html).write_pdf(
        ".output/scientific_report.pdf",
        stylesheets=[CSS(string=css_content)],
        font_config=font_config,
    )
    print("✓ PDF generated successfully: .output/scientific_report.pdf")
    print(
        "  Equations are now rendered with LaTeX-style formatting (Unicode subscripts/superscripts)"
    )
except Exception as e:
    print(f"✗ PDF generation failed: {e}")
