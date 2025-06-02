
# biology galaxies supernovae fluid graphs
listdata="biology galaxies supernovae fluid graphs"
method="PySR"


for i in $listdata; do
    python main.py $i $method
    echo Processing $i $method OK 
done

