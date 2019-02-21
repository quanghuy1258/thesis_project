#!/bin/bash

echo "Warning: Please check that you have already installed C/C++ compiler"
echo "To install C/C++ compiler: Install gcc and g++"
read -r -p "Continue? [y/N] " response;
if [ -z "$(echo $response | grep -E "^([yY][eE][sS]|y)$")" ]; then
  exit;
fi;

git submodule update --init --recursive;
cd build >/dev/null 2>&1 && cmake .. "$@";
