"""
Generate config.yaml.default from the configuration schema.

The shipped default configuration is not edited by hand.  Its settings, their
values and their comments all come from the models in substation.config: each
live value is the field's default, and each comment is the field's docstring.
The hand-written band library in band_library.yaml follows them unchanged.

Run from a source checkout after changing a setting or a band:

	python -m substation.generate_default_config

tests/test_generate_default_config.py fails if the committed file differs from
what this writes.
"""

import inspect
import json
import math
import pathlib
import re
import textwrap
import typing

import pydantic
import yaml

import substation.config


PACKAGE_DIR = pathlib.Path(__file__).parent
DEFAULT_CONFIG_PATH = PACKAGE_DIR / "config.yaml.default"
BAND_LIBRARY_PATH = PACKAGE_DIR / "band_library.yaml"
REGENERATE_COMMAND = "python -m substation.generate_default_config"

# Comment lines wrap at this many columns, including indentation and '# '
LINE_WIDTH = 79
INDENT = "    "

# A number and its unit stay on one line when comments are wrapped
NUMBER_BEFORE_UNIT = re.compile(r"(\d) (?=(?:k|M|G)?Hz\b|dBFS\b|dB\b|ms\b|PPM\b|seconds\b)")
UNBREAKABLE_SPACE = "\u00a0"

HEADER = f"""\
# Substation configuration, with every setting at its default value.
#
# `substation --init` copies this file to start a config.yaml.  A setting
# removed from config.yaml falls back to the value shown here.
#
# Substation generates this file from substation/config.py and
# substation/band_library.yaml, with:
#
#     {REGENERATE_COMMAND}
"""

BAND_SETTINGS_INTRO = """\
Band settings

`band_defaults` holds templates of settings, and `bands` holds the bands to
scan; the band library that follows fills both.  The settings each template,
band, and device override accepts are listed first, each with its default."""


def _comment_lines (text: str, prefix: str) -> list[str]:

	"""
	Turn prose into wrapped YAML comment lines, each starting with prefix.

	Each paragraph of the text is re-flowed to fit LINE_WIDTH, because docstrings
	are wrapped for the source file, not for the depth they appear at here.
	Paragraphs are separated by a line holding the prefix alone.
	"""

	width = LINE_WIDTH - len(prefix)
	lines: list[str] = []

	for paragraph in inspect.cleandoc(text).split("\n\n"):

		if lines:
			lines.append(prefix.rstrip())

		joined = " ".join(line.strip() for line in paragraph.splitlines())
		joined = NUMBER_BEFORE_UNIT.sub(rf"\1{UNBREAKABLE_SPACE}", joined)

		# textwrap breaks only on ASCII whitespace, so the unbreakable space holds
		wrapped = textwrap.wrap(joined, width=width, break_long_words=False, break_on_hyphens=False)
		lines.extend(f"{prefix}{line}".replace(UNBREAKABLE_SPACE, " ") for line in wrapped)

	return lines


def _float_text (value: float) -> str:

	"""Write a float compactly: whole numbers without a point, large ones in engineering form such as 93.7e+6."""

	if value.is_integer() and abs(value) < 1e4:
		return str(int(value))

	if abs(value) >= 1e4:

		exponent = 3 * (int(math.floor(math.log10(abs(value)))) // 3)
		mantissa = f"{value / 10 ** exponent:.6g}"

		if "." not in mantissa:
			mantissa += ".0"

		candidate = f"{mantissa}e+{exponent}"

		if float(candidate) == value:
			return candidate

	return repr(value)


def _yaml_text (value: typing.Any) -> str:

	"""Write one value as YAML, in the style a person would type it."""

	if value is None:
		return "null"

	if isinstance(value, bool):
		return "true" if value else "false"

	if isinstance(value, int):
		return str(value)

	if isinstance(value, float):
		return _float_text(value)

	if isinstance(value, str):

		# A plain string needs quotes only if YAML would read it as something else
		if yaml.safe_load(value) == value:
			return value

		return json.dumps(value)

	if isinstance(value, dict):
		return "{" + ", ".join(f"{_yaml_text(key)}: {_yaml_text(item)}" for key, item in value.items()) + "}"

	if isinstance(value, list):
		return "[" + ", ".join(_yaml_text(item) for item in value) + "]"

	raise TypeError(f"No YAML form for {type(value).__name__} value {value!r}")


def yaml_value (value: typing.Any) -> str:

	"""
	Write one value as YAML that reads back as the same value.

	Raises:
		ValueError: If Substation's own YAML loader would read the text back as a
			different value, so a misformatted default can never be shipped.
	"""

	text = _yaml_text(value)
	loaded = yaml.load(f"value: {text}", Loader=substation.config._YamlLoader)["value"]

	if loaded != value:
		raise ValueError(f"{value!r} would be written as {text!r} and read back as {loaded!r}")

	return text


def _nested_model (field: pydantic.fields.FieldInfo) -> type[pydantic.BaseModel] | None:

	"""Return the model a field holds directly, or None if it holds a plain value."""

	annotation = field.annotation

	if inspect.isclass(annotation) and issubclass(annotation, pydantic.BaseModel):
		return annotation

	return None


def _live_settings (model: type[pydantic.BaseModel], indent: str) -> list[str]:

	"""
	Write every setting of a model as a live key at its default, under its description.

	Raises:
		ValueError: If a setting has no default, since a live value must always
			equal the default the schema declares.
	"""

	lines: list[str] = []

	for name, field in model.model_fields.items():

		if field.is_required():
			raise ValueError(f"{model.__name__}.{name} has no default, so it cannot be written as a live value")

		lines.append("")
		lines.extend(_comment_lines(field.description or "", f"{indent}# "))

		for example in field.examples or []:
			lines.append(f"{indent}# Example: {yaml_value(example)}")

		nested = _nested_model(field)

		if nested is None:
			lines.append(f"{indent}{name}: {yaml_value(field.get_default(call_default_factory=True))}")
			continue

		lines.append(f"{indent}{name}:")
		lines.extend(_live_settings(nested, indent + INDENT))

	return lines


def _reference_settings (title: str, model: type[pydantic.BaseModel]) -> list[str]:

	"""
	Write every setting of a model as a commented-out entry with its default.

	Used for settings that belong to each template, band, or device override,
	which have no single place in the file to be live.
	"""

	lines = ["#", f"# {title}", "#"]
	lines.extend(_comment_lines(model.__doc__ or "", "# "))

	for name, field in model.model_fields.items():

		shown = "(required)" if field.is_required() else yaml_value(field.get_default(call_default_factory=True))

		lines.append("#")
		lines.append(f"#   {name}: {shown}")
		lines.extend(_comment_lines(field.description or "", "#     "))

		for example in field.examples or []:
			lines.append(f"#     Example: {yaml_value(example)}")

	return lines


def render () -> str:

	"""Return the full text of config.yaml.default: header, generated settings, then the band library."""

	lines = HEADER.splitlines()

	for section in ("scanner", "recording"):

		model = _nested_model(substation.config.AppConfig.model_fields[section])
		assert model is not None

		lines.append("")
		lines.extend(_comment_lines(model.__doc__ or "", "# "))
		lines.append(f"{section}:")
		lines.extend(_live_settings(model, INDENT))

	lines.append("")
	lines.extend(_comment_lines(BAND_SETTINGS_INTRO, "# "))
	lines.extend(_reference_settings("Settings for each template in `band_defaults`", substation.config.BandTypeConfig))
	lines.extend(_reference_settings("Settings for each band in `bands`", substation.config.BandConfig))
	lines.extend(_reference_settings("Settings for each device family in a band's `device_overrides`", substation.config.DeviceOverrideConfig))

	lines.append("")
	lines.append(BAND_LIBRARY_PATH.read_text(encoding="utf-8").rstrip("\n"))

	return "\n".join(lines) + "\n"


def main () -> int:

	"""Write config.yaml.default, after checking that the result loads as a valid configuration."""

	text = render()
	substation.config.validate_config(yaml.load(text, Loader=substation.config._YamlLoader))

	DEFAULT_CONFIG_PATH.write_text(text, encoding="utf-8")
	print(f"Wrote {DEFAULT_CONFIG_PATH}")

	return 0


if __name__ == "__main__":
	raise SystemExit(main())
