#!/usr/bin/env python
import sys

from jinja2 import Template


print(Template(sys.stdin.read(),trim_blocks=True).render())

