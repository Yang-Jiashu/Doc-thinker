"""Offline checks for the shared workspace's pages and packaged assets."""

import json
import re
import shutil
import subprocess
from html.parser import HTMLParser
from pathlib import Path

import pytest
from jinja2 import Environment, FileSystemLoader
from flask import url_for

from docthinker.ui.app import app

ROOT = Path(__file__).parents[1]
UI = ROOT / "docthinker/ui"


class PageElements(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = []
        self.labels = []
        self.local_styles = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids.append(attrs["id"])
        if tag == "label":
            self.labels.append(attrs)
        if tag == "link" and attrs.get("href", "").startswith("/static/"):
            self.local_styles.append(attrs["href"].split("?", 1)[0][1:])


@pytest.mark.parametrize(
    "template,endpoint",
    [
        ("query_modern.html", "query_page"),
        ("config_modern.html", "config_page"),
        ("kg_viz_modern.html", "knowledge_graph_page"),
        ("upload_modern.html", "upload_page"),
    ],
)
def test_workspace_pages_have_real_assets_and_valid_scripts(template, endpoint):
    env = Environment(loader=FileSystemLoader(UI / "templates"))
    env.globals["url_for"] = url_for
    with app.test_request_context():
        html = env.get_template(template).render(
            request={"endpoint": endpoint}, api_config={"api_prefix": "/api/v1"}, config={}
        )
    page = PageElements()
    page.feed(html)
    assert "static/workspace.css" in page.local_styles
    for asset in page.local_styles:
        assert (UI / asset).is_file(), asset
    assert len(page.ids) == len(set(page.ids)), "Duplicate control IDs"
    if template == "config_modern.html":
        assert page.labels
        assert all(label.get("for") in page.ids for label in page.labels)
    node = shutil.which("node")
    if node:
        scripts = re.findall(r"<script\b[^>]*>([\s\S]*?)</script>", html)
        result = subprocess.run(
            [node, "-e", "const vm=require('vm'); const fs=require('fs'); "
             "JSON.parse(fs.readFileSync(0,'utf8')).forEach(s=>new vm.Script(s));"],
            input=json.dumps(scripts), text=True, capture_output=True, timeout=10,
        )
        assert result.returncode == 0, result.stderr


def test_distribution_includes_ui_template_and_style_patterns():
    manifest = (ROOT / "MANIFEST.in").read_text()
    assert "recursive-include docthinker/ui/templates *.html" in manifest
    assert "recursive-include docthinker/ui/static *.css" in manifest
