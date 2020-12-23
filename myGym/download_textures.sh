#!/bin/bash

wget -r --no-parent --reject "index.html*" https://data.ciirc.cvut.cz/public/groups/incognite/myGym/dtdseamless/warpedImages/
mv -Tf data.ciirc.cvut.cz/public/groups/incognite/myGym/dtdseamless/warpedImages ./envs/dtdseamless
