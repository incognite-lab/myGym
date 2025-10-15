#!/bin/bash

wget -r --no-parent --reject "index.html*" https://data.ciirc.cvut.cz/public/groups/incognite/myGym/trained_models/

mkdir -p trained_models

mv -T data.ciirc.cvut.cz/public/groups/incognite/myGym/trained_models ./trained_models

rm -r data.ciirc.cvut.cz