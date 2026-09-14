#!/usr/bin/env python3
"""A new rented box becomes the evaluator with one command, and the tokens never touch a command line.

The script talks to a real box, so it is not run here. What is checked is what would be expensive
to find out on a box: that it parses, how the tokens travel, and that a new box continues the
published ledger instead of starting a second history that could never be pushed.
"""
from __future__ import annotations

import re
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT = ROOT / "eval" / "provision_box.sh"


class TestProvisioningABox(unittest.TestCase):
    SRC = SCRIPT.read_text()

    def test_it_parses(self):
        subprocess.run(["bash", "-n", str(SCRIPT)], check=True)

    def test_the_tokens_travel_over_stdin_not_arguments(self):
        self.assertIn('cat > /var/lib/burnish/secrets.env\' < "$SECRETS"', self.SRC)
        for line in self.SRC.splitlines():
            if re.search(r"\bssh\b|\"\$\{SSH\[@\]\}\"", line):
                self.assertNotRegex(line, r"\$(GH_TOKEN|BURNISH_LEDGER_TOKEN)", line)

    def test_a_secrets_file_other_accounts_can_read_is_refused(self):
        self.assertIn("600|400)", self.SRC)

    def test_a_new_box_clones_the_published_ledger_before_any_round(self):
        clone = self.SRC.index('git clone -q "$BURNISH_LEDGER_REMOTE" "$BURNISH_LEDGER"')
        self.assertLess(clone, self.SRC.index("crontab -"))

    def test_the_box_is_checked_before_rounds_are_scheduled(self):
        self.assertLess(self.SRC.index("--check-box"), self.SRC.index("crontab -"))

    def test_the_cron_line_is_the_one_the_wrapper_documents(self):
        wrapper = (ROOT / "eval" / "run_round_cron.sh").read_text()
        self.assertIn("0 */2 * * *", wrapper)
        self.assertIn('line="0 */2 * * * $R/eval/run_round_cron.sh >> /var/log/burnish-round.log 2>&1"',
                      self.SRC)


if __name__ == "__main__":
    unittest.main(verbosity=2)
