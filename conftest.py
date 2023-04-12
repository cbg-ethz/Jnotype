"""PyTest unit tests configuration file."""
import dataclasses

import pytest


@dataclasses.dataclass
class TurnOnTestSuiteArgument:
    """CLI argument together with a keyword used to mark tests turned off by default.

    Attrs:
        cli_flag: CLI flag to be used to turn on an optional test suite, e.g., --run-r
        help_message: CLI help message, describing the flag
        mark_keyword: keyword used to mark the unit tests to be skipped
            if the flag is not specified.
            Note that it should be a valid Python name, e.g., requires_r
        inivalue_line_description: a short description of the marker `mark_keyword`.
            See also: `inivalue_line`
    """

    cli_flag: str
    help_message: str
    mark_keyword: str
    inivalue_line_description: str

    def reason(self) -> str:
        """Reason for skipping, generated when the tests are run."""
        return f"Need {self.cli_flag} option to run."

    def inivalue_line(self) -> str:
        """Generates human-readable description describing the argument."""
        return f"{self.mark_keyword}: {self.inivalue_line_description}"


# Example usage:
# TURN_ON_ARGUMENTS = [
#     TurnOnTestSuiteArgument(
#         cli_flag="--run-r",
#         help_message="Run tests requiring R dependencies.",
#         mark_keyword="requires_r",
#         inivalue_line_description="mark test as requiring R dependencies to run.",
#     ),
# ]
TURN_ON_ARGUMENTS = []


def pytest_addoption(parser):
    """Adds CLI options."""
    for argument in TURN_ON_ARGUMENTS:
        parser.addoption(
            argument.cli_flag,
            action="store_true",
            default=False,
            help=argument.help_message,
        )

    parser.addoption(
        "--save-artifact",
        action="store_true",
        help="Auxiliary artifact will be generated.",
    )


def pytest_configure(config):
    """I do not know how this works."""
    for argument in TURN_ON_ARGUMENTS:
        config.addinivalue_line("markers", argument.inivalue_line())


@pytest.fixture
def save_artifact(request):
    """Boolean fixture representing a CLI flag.

    True if generated artifacts should be saved, False if not."""
    return request.config.getoption("--save-artifact")


def add_skipping_markers(
    argument: TurnOnTestSuiteArgument,
    config,
    items,
) -> None:
    """Basic function to skip options
    marked with a specific marker,
    unless a CLI argument is provided."""
    if not config.getoption(argument.cli_flag):
        skip = pytest.mark.skip(reason=argument.reason())
        for item in items:
            if argument.mark_keyword in item.keywords:
                item.add_marker(skip)


def pytest_collection_modifyitems(config, items):
    """PyTest function modifying markers.

    It consists of:
      - Adds `add_skipping_markers`
    """
    for argument in TURN_ON_ARGUMENTS:
        add_skipping_markers(argument=argument, config=config, items=items)
