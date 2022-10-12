for model_name in vgg19 densenet121 resnext50_32x4d 
do
    for class in 0 20 40 60 80 100
    do 
        python attack.py --device cuda:0 -targeted --eps 0.0625 --model_name $model_name --target_class $class
    done
done
