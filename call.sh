#!/bin/bash
for i in {0..255}; do
  curl "https://api.airplanes.live/v2/icao/45$(printf '%02x' $i)"
done

