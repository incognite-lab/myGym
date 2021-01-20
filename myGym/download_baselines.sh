#!/bin/bash

wget -r --no-parent --reject "index.html*" https://data.ciirc.cvut.cz/public/groups/incognite/myGym/baselines/

mkdir -p trained_models

mv -T data.ciirc.cvut.cz/public/groups/incognite/myGym/baselines ./trained_models/baselines

rm -r data.ciirc.cvut.cz