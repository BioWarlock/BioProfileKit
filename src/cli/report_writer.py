import shutil
from pathlib import Path

from jinja2 import Environment, FileSystemLoader
from importlib_resources import files

TEMPLATE_DIR = files("templates").joinpath()
STATIC_DIR = files("static").joinpath()
env = Environment(loader=FileSystemLoader(str(TEMPLATE_DIR)), autoescape=True)

PAGE_MAP = {
    "overview":      "numeric_data.html",
    "columns":       "columns.html",
    "multivariate":  "general_statistics.html",
}

def _resolve_detail_links(quality):
    if quality is None:
        return
    for category in quality.categories:
        for check in category.checks:
            if not check.detail_link:
                continue
            anchor = check.detail_link.lstrip("#")
            check.detail_link = PAGE_MAP.get(anchor)

def write_report(output_path: Path, general, plots, dup_groups,
                 column_overviews, numeric_overviews, categorical_overviews,
                 top_n, quality=None):
    output_path.mkdir(parents=True, exist_ok=True)
    shutil.copytree(str(STATIC_DIR), str(output_path / "static"), dirs_exist_ok=True)

    _resolve_detail_links(quality)

    _render_to_file(output_path / "index.html", 'LandingPage.jinja')
    _render_to_file(output_path / "numeric_data.html", 'numeric_overview.jinja',
                    general=general, dup_groups=dup_groups, quality=quality)
    _render_to_file(output_path / "columns.html", 'columns.jinja',
                    columns=column_overviews, overview=numeric_overviews,
                    categorical=categorical_overviews, top_n=top_n)
    _render_to_file(output_path / "general_statistics.html", 'general_statistics.jinja',
                    plots=plots)


def _render_to_file(filepath: Path, template_name: str, **context):
    template = env.get_template(template_name)
    with open(filepath, 'w', encoding="utf-8") as f:
        f.write(template.render(**context))