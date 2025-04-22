#!/usr/bin/env bash
for truefreq in 1 3 10 30 100 300 1000 10000; do
  for readnoise in 1 3 10 30 100 300 1000 10000; do
    python3 compare_fits.py --truefreq "$truefreq" --readnoise "$readnoise"
  done
done

