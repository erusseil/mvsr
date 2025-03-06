
listdata="biology galaxies graphs supernovae fluid"
method="pyoperon"


for i in $listdata; do
    nohup python main.py $i $method
    echo Processing $i $method OK 
done

