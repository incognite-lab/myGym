# Rename all *.txt to *.text
for f in *.STL; do 
    mv -- "$f" "${f%.STL}.stl"
done
