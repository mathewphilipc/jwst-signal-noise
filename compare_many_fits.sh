#!/usr/bin/env bash
for truefreq in 1 10; do
  for readnoise in 1 10; do
    python3 compare_fits.py --truefreq "$truefreq" --readnoise "$readnoise"
  done
done

