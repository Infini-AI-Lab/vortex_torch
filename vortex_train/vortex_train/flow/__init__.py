"""Frontend: the selection-policy contract users write against."""
from .spec import Budget, Field, Selection, get_selection, register  # noqa: F401
from . import ops  # noqa: F401
from . import recipes  # noqa: F401  - importing registers the catalogue
