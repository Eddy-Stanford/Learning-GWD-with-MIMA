import sys

import click

from lrgwd.extractor.__main__ import main as extractor
from lrgwd.ingestor.__main__ import main as ingestor
from lrgwd.split.__main__ import main as split
from lrgwd.train.__main__ import main as train
from lrgwd.evaluate.__main__ import main as evaluate


@click.group()
def main():
    pass


main.add_command(ingestor)
main.add_command(extractor)
main.add_command(split)
main.add_command(train)
main.add_command(evaluate)


if __name__ == "__main__":
    main()
