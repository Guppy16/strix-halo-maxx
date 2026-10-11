#!/bin/bash
# pass 21: freeze research CP-SAT (SIGSTOP) while main has a WINDOW OPEN in RESULTS_FROM_MAIN, resume (SIGCONT) when it closes.
# usage: r26/cpgate.sh REGEX   (matched against process command lines; exits when none is left)
R=$HOME/git/app/strix-halo-maxx/tapnext/research/RESULTS_FROM_MAIN.md
pat=$1
while pids=$(ps -eo pid=,args= | grep -E "$pat" | grep -v -e grep -e cpgate | awk '{print $1}'); [ -n "$pids" ]; do
  last=$(grep -E "^- WINDOW (OPEN|CLOSED)" $R | tail -1)
  if echo "$last" | grep -q "WINDOW OPEN" && ! echo "$last" | grep -q "research"; then kill -STOP $pids 2>/dev/null; else kill -CONT $pids 2>/dev/null; fi
  sleep 3
done
