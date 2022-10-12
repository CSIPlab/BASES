python GFCS_main.py --model_name vgg19 --n_wb 20 --GFCS --num_step 500 --linf 0.0625 --step_size 0.005
python GFCS_main.py --model_name vgg19 --n_wb 20 --GFCS --num_step 500 --linf 0.0625 --step_size 0.005 --targeted

python GFCS_main.py --model_name vgg19 --n_wb 20 --ODS --num_step 500 --linf 0.0625 --step_size 0.005
python GFCS_main.py --model_name vgg19 --n_wb 20 --ODS --num_step 500 --linf 0.0625 --step_size 0.005 --targeted

python rgf_variants_pytorch.py --model_name vgg19 --method biased --dataprior --n_wb 20 --max_queries 500 --norm linfty --norm_bound 0.0625 --step_size 0.005
python rgf_variants_pytorch.py --model_name vgg19 --method biased --dataprior --n_wb 20 --max_queries 500 --norm linfty --norm_bound 0.0625 --step_size 0.005 --targeted