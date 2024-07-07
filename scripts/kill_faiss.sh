lsof -i:5000 |  awk '{print $2}' | xargs -I {} kill -9 {}
