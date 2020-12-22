#!/bin/bash

wget -r --no-parent --reject "index.html*" https://data.ciirc.cvut.cz/public/groups/incognite/myGym/vae/vae_sidearm_beta4/
mv -Tf data.ciirc.cvut.cz/public/groups/incognite/myGym/vae/vae_sidearm_beta4 ./trained_models/vae_sidearm_beta4

wget -r --no-parent --reject "index.html*" https://data.ciirc.cvut.cz/public/groups/incognite/myGym/yolact/weights_yolact_mygym_23/
mv -Tf data.ciirc.cvut.cz/public/groups/incognite/myGym/yolact/weights_yolact_mygym_23 ./trained_models/weights_yolact_mygym_23