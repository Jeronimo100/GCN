#!/bin/bash
# Nucleo sequences | bash script

array=('0' '1' '2' '3' '4' '5' \
         '6' '7' '8' )
array2=('0' '1' '2' '3' '4' '5')
array3=('0')
array4=('5')
array5=('6')
arrayCora=('2709' '2708' '2000' '1800' '1500' '1200' '1000' '900' '700' '600' '500' '400' '300' '250' '200' '150' '140' '130' '120' '110' '100')
arrayCora2=('2704')
#declare -a arrayCora3=("1 2708")
arrayCora3=(1 400\
            3 1800 3 1352 3 1000 3 800 3 600 3 500 3 400 \
            5 1800 5 1352 5 1000 5 800 5 600 5 500 5 400 \
            7 1800 7 1352 7 1000 7 800 7 600 7 500 7 400 \
            9 1800 9 1352 9 1000 9 800 9 600 9 500 9 400 \
            3 1000 5 1000 7 1000 9 1000 14 1000 20 1000 \
            3 1352 5 1352 7 1352 9 1352 \
	    9 800 11 800 12 800 15 800 20 800 100 800)
            #9 600 11 600 12 600 15 600 20 600 100 600 500 600 1000 800 \
            #1000 600)
	    # 1 1800 1 1352 1 1000 1 800 1 600 1 500 1 400 \
            # 10 1200 10 1352 10 1800 10 900
	    #5 1800 5 1352 5 1000 5 800 5 600 5 500 5 400 \
	    #7 1800 7 1352 7 1000 7 800 7 600 7 500 7 400 \
  	    #9 1800 9 1352 9 1000 9 800 9 600 9 500 9 400 \
	    #11 800 15 800 20 800 100 800 500 800 \
	    #11 600 15 600 20 600 100 600 500 600 1000 800 \
            #1000 600 10 900 10 1000 10 1200 10 1350 10 1800)
	    #3 1800 3 1352 3 1000 3 800 3 600 3 500 3 400)
	    #(3 1000 5 1000 7 1000 9 1000 14 1000 20 1000 \
            #3 1352 5 1352 7 1352 9 1352 \
            #10 600 10 700 10 800 10 900 10 1000 10 1200 10 1350 10 1800 \ 
            #9 600 11 600 12 600 20 600 3 800 5 800 7 800 9 800 11 800 20 800 \ 
            #1 1800 1 1352 1 1000 1 800 1 600 1 500 1 400 ) #1 2708 10 400 10 500)
arrayPubMed=('19717' '19000' '15000' '10000' '8000' '6000' '5000' '4000' '3500' '2000' '1800' '1600' '1400' '1200' '1000' '700')
arrayCiteSeer=('3328' '3327' '3000' '2500' '1800' '1500' '1200' '1000' '900' '700' '600' '500' '400' '300' '250' '200' '150' '140' '130' '120' '110' '100')

arraylength=${#arrayCora3[@]}

for (( i=0; i<${arraylength}; i=i+2 ));
do
  python train_vs6_Batching_timed_args_from_RUNScript_RANDOM_5percent.py ${arrayCora3[$i]} ${arrayCora3[$i+1]}
        #train_vs6_Batching_timed_args_from_RUNScript.py ${arrayCora3[$i]} ${arrayCora3[$i+1]}
  #echo $i " / " ${arraylength} " : " ${array[$i-1]}
done


#for element in $arrayCora2
#for element in ${arrayCiteSeer[@]}
#for element in ${arrayPubMed[@]}
#
#do
    #echo $element
    #python train_vs6_Batching_timed_args_from_RUNScript.py $element
    #python train_June_26_vs7c_run_to_be_Called_by_script_Styled_Printin.py $element
    #python train_July_20_vs15_Father_Datasets__Incrd_Batch_size.py $element
    #python train_July_20_v14_Father_Datasets__Same_Batches.py $element
    # python train_June_18_vs5.py $element
    #python train_June_26_vs7_run_to_be_Called_by_script.py $element
    #python train_June_26_vs7c_run_to_be_Called_by_script_Styled_Printin.py $element
    #python train_July_11_vs12_mini_batching.py $element
    #python train_July_11_vs13_Father_Datasets_batching.py
    # python train_July_20_vs14_Father_Datasets__Same_Batches.py
    #python train_June_28_vs9_mini_batching.py $element
    #python train_July_08_vs10_mini_batching.py $element
    #printf "\n"
#done


#for element in ${arrayCiteSeer[@]}  #for element in ${arrayPubMed[@]}
#do
    #python train_July_20_v14_Father_Datasets__Same_Batches.py $element
    #printf "\n"
#done
