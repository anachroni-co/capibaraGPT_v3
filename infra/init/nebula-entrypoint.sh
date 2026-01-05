#!/bin/sh
set -e

HOST=nebula-graphd
PORT=9669

while ! nc -z  ; do
  echo [nebula-console]
