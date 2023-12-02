#!/bin/sh
set -e
original="$1" target="$2"
if [ -d "$target" ]; then
  target="$target/${original##*/}"
fi
mv -- "$original" "$target"
case "$original" in
  */*)
    case "$target" in
      /*) :;;
      *) target="$(cd -- "$(dirname -- "$target")" && pwd)/${target##*/}"
    esac
esac
ln -s -- "$target" "$original"