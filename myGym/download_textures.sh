#!/bin/bash

wget -r --no-parent --reject "index.html*" https://data.ciirc.cvut.cz/public/groups/incognite/myGym/dtdseamless/warpedImages/

mv -T data.ciirc.cvut.cz/public/groups/incognite/myGym/dtdseamless/warpedImages ./envs/dtdseamless

rm -r data.ciirc.cvut.cz
