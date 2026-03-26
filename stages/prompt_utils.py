"""Safe prompt rendering — avoids str.format() conflicts with JSON braces in templates."""


def render(template: str, **kwargs) -> str:
    """Replace {key} placeholders without interpreting unrelated braces."""
    for key, value in kwargs.items():
        template = template.replace("{" + key + "}", str(value))
    return template
