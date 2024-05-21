
class Logger:
    def __init__(self):
        self.output = ""

    def write(self, what):
        self.output += str(what) + '\n'

    def print(self):
        print(self.output)

import inquarting

for name in inquarting.__dict__:
    if name.startswith("test_") and callable(inquarting.__dict__[name]):
        logger = Logger();
        print(f"Running {name}...")
        success = inquarting.__dict__[name](logger)
        if not success:
            print(f"\033[91m{name} FAILED\033[0m")
            print(name + " log:")
            logger.print()
