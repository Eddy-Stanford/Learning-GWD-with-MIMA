import sys

import click
from lrgwd.extractor.__main__ import main as extractor
from lrgwd.ingestor.__main__ import main as ingestor
from lrgwd.performance.compare.__main__ import main as compare
from lrgwd.performance.evaluate.__main__ import main as evaluate
from lrgwd.split.__main__ import main as split
from lrgwd.train.__main__ import main as train


@click.group()
def main():
    pass

main.add_command(ingestor)
main.add_command(extractor)
main.add_command(split)
main.add_command(train)
main.add_command(evaluate)
main.add_command(compare)


if __name__ == "__main__":
    main()
