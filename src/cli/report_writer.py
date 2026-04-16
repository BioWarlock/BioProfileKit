import shutil
from pathlib import Path

from jinja2 import Environment, FileSystemLoader
from importlib_resources import files

TEMPLATE_DIR = files("templates").joinpath()
STATIC_DIR = files("static").joinpath()
env = Environment(loader=FileSystemLoader(str(TEMPLATE_DIR)), autoescape=True)


def write_report(output_path: Path, general, plots, duplicates_table,
                 column_overviews, numeric_overviews, categorical_overviews, top_n):
    output_path.mkdir(parents=True, exist_ok=True)
    shutil.copytree(str(STATIC_DIR), str(output_path / "static"), dirs_exist_ok=True)

    _render_to_file(output_path / "index.html", 'LandingPage.jinja')
    _render_to_file(output_path / "numeric_data.html", 'numeric_overview.jinja',
                    general=general, dups=duplicates_table)
    _render_to_file(output_path / "columns.html", 'columns.jinja',
                    columns=column_overviews, overview=numeric_overviews,
                    categorical=categorical_overviews, top_n=top_n)
    _render_to_file(output_path / "general_statistics.html", 'general_statistics.jinja',
                    plots=plots)


def _render_to_file(filepath: Path, template_name: str, **context):
    template = env.get_template(template_name)
    with open(filepath, 'w', encoding="utf-8") as f:
        f.write(template.render(**context))