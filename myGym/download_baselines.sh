#!/bin/bash

wget -r --no-parent --reject "index.html*" https://data.ciirc.cvut.cz/public/groups/incognite/myGym/baselines
mv -Tf data.ciirc.cvut.cz/public/groups/incognite/myGym/baselines ./trained_models/baselines